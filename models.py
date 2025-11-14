import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_span import AdaptiveSpan
from adaptive_io import build_adaptive_io, compute_dummy_loss

# Funciones auxiliares

def _skew(X, pad_value):
# Transforma la matriz de atención para que cada fila esté desplazada correctamente según las posiciones relativas.
# Efecto: Cada fila está desplazada 1 posición más a la derecha que la anterior - esto permite indexación correcta de posiciones relativas.
    """shift every row 1 step to right"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
    X = X.view(B, -1)  # B x ML+MM+M
    X = X[:, :-M]  # B x ML+MM
    X = X.view(B, M, M + L)  # B x M x L+M
    return X


def _unskew(X):
# Operación inversa de `_skew`. Deshace el desplazamiento.
# Después de agregar los position embeddings, necesitamos volver al formato original para aplicar softmax correctamente.
    """reverse _skew operation"""
    # X = B x M x L+M
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)  # B x ML+MM
    X = F.pad(X, (0, M))  # B x ML+MM+M
    X = X.view(B, M, M + L + 1)  # B x M x L+M+1
    X = X[:, :, :L]  # B x M x L
    return X

class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """
  # Implementa self-attention secuencial con span adaptativo.
    def __init__(self, hidden_size, nb_heads, attn_span,
                 dropout, adapt_span_params, pers_mem_params, **kargs):
    # Args:
    # hidden_size: Dimensión de UNA cabeza (ej: si el modelo tiene 512 dims y 8 cabezas, esto es 64)
    # nb_heads: Número de cabezas de atención
    # attn_span: Span máximo permitido
    # adapt_span_params: Dict con configuración del adaptive span
    # pers_mem_params: Para persistent memory (paper diferente, puede ser 0)
                   
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size # size of a single head
        self.attn_span = attn_span
        self.adapt_span_enabled = adapt_span_params['adapt_span_enabled']
        if self.adapt_span_enabled:
            self.adaptive_span = AdaptiveSpan(attn_span=attn_span, nb_heads=nb_heads,
                                              **adapt_span_params, **kargs)
    
        self.persistent_memory = None
    # Opcional (0) para este ejercicio
        if pers_mem_params['pers_mem_size'] > 0:
            self.persistent_memory = PersistentMemory(
                pers_mem_params['pers_mem_size'], nb_heads, hidden_size, dropout)
            if self.adapt_span_enabled:
                self.persistent_memory.adaptive_span = self.adaptive_span

    def forward(self, query, key, value, key_pe):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H
    
    # ENTRADA: 
    # query size = B x M x H
    # key, value sizes = B x (M+L) x H  
    # key_pe = position embeddings
    
    # query: Los M tokens actuales que queremos procesar
    # key, value: Contexto completo = cache (L tokens antiguos) + tokens actuales (M)
    # key_pe: Position embeddings relativos, forma (1, H, L)
    
        if self.adapt_span_enabled:
            # [optional] trim out memory to reduce unnecessary computation
            key, value, key_pe = self.adaptive_span.trim_memory(
                query, key, value, key_pe)
    # Recorta tokens antiguos que nunca serán atendidos según el span actual.
    
        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = _unskew(attn_cont)  # B x M x L
    # Calcula scores de atención basados en el **contenido** (query · key)
    
    
        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn = attn_cont + attn_pos
    # Agrega información de posiciones relativas.
    
        if self.persistent_memory is not None:
            attn, pers_mem_out = self.persistent_memory(query, attn)
        else:
            attn = attn / math.sqrt(self.hidden_size)  # B x M X L_pos
    # Scaling
            attn = F.softmax(attn, dim=-1)
    # Convierte scores a probabilidades
    # Suma a 1 en la dimensión del span
    
    
            if self.adapt_span_enabled:
                # trim attention lengths according to the learned span
    # Cada cabeza enmascara según su span aprendido
                attn = self.adaptive_span(attn)
    
        attn = self.dropout(attn)  # B x M X L_pos
    # Regularización - aleatoriamente pone algunos pesos en 0 durante entrenamiento.
    
    
        attn_cont = _skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H
    # Usa los pesos de atención para combinar los valores.
    
    
        if self.persistent_memory is not None:
            out = out + pers_mem_out
    
        return out


    def get_cache_size(self):
      if self.adapt_span_enabled:
          return self.adaptive_span.get_cache_size()
      else:
          return self.attn_span

 # Al inicializar el modelo, cada capa necesita saber cuánta memoria reservar para el cache.

class MultiHeadSeqAttention(nn.Module):

 # Orquestador que coordina múltiples cabezas de atención en paralelo.
    def __init__(self, hidden_size, nb_heads, **kargs):
        nn.Module.__init__(self)

        # Validación y Setup
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads

        # Crear seq attention
        # Importante, cada cabez opera independiente en su subespacio de 64 dimensiones
        # Seq Attention contiene el adaptive span
        self.attn = SeqAttention(
            hidden_size=self.head_dim, nb_heads=nb_heads, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

       
    def head_reshape(self, x):
    # Reorganiza la dimensión `hidden_size=256` en `(nb_heads=4, head_dim=64)`.
      
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D ### Queremos que la dimensión de las cabezas esté **antes** que la dimensión temporal, para poder procesarlas en paralelo.
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D ### Queremos tratar cada cabeza de cada batch como un ejemplo independiente, para poder procesarlas todas en una sola operación matricial paralela.
        return x
  
    def forward(self, query, key, value, key_pe):

        # Setup
        B = query.size(0)    # batch size
        K = self.nb_heads    # número de cabezas
        D = self.head_dim    # dimensión de cada cabeza
        M = query.size(1)    # número de tokens en query

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        #Aplicar self attention
        out = self.attn(query, key, value, key_pe)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        # Mezcla la información de todas las cabezas 
        return out


class FeedForwardLayer(nn.Module):

  # Red neuronal feedforward de 2 capas que se aplica a cada posición independientemente. 
  # Param:
  # hidden_size: Dimensión del modelo 
  # inner_hidden_size: Dimensión de la capa oculta 
  # dropout: Probabilidad de dropout 
  
    def __init__(self, hidden_size, inner_hidden_size, dropout, **kargs):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class TransformerSeqLayer(nn.Module):

# Capa completa que combina atención + feedforward.
# Arquitectura estándar del Transformer.
    def __init__(self, hidden_size, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttention(hidden_size=hidden_size, **kargs)
        self.norm1 = nn.LayerNorm(hidden_size) # Primera capa de normalización, Mantiene activaciones en rango razonable
        if kargs['pers_mem_params']['pers_mem_size'] > 0:
            # replacing FF with persistent memory
            # Para adpative span siempre se usa feedforward
            self.ff = None
        else:
            self.ff = FeedForwardLayer(hidden_size=hidden_size, **kargs)
            self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, h, h_cache, key_pe):
        # h = B x M x H
        # h_cache = B x L x H
        
        # h: Tokens actuales (B, M, H)
        # h_cache: Historia/cache (B, L, H)
        # key_pe: Position embeddings (1, H_head, L)
        h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H (Concatenar cache con Tokens actuales)
        attn_out = self.attn(h, h_all, h_all, key_pe)
        h = self.norm1(h + attn_out)  # B x M x H (Skip cpnnection)
        if self.ff is not None:
            ff_out = self.ff(h)
            out = self.norm2(h + ff_out)  # B x M x H
        else:
            out = h
        return out

class TransformerSeq(nn.Module):
# Params:
# vocab_size       Tamaño del vocabulario (ej: 27 para text8, 205 para enwik8)
# hidden_size      Dimensión del modelo (ej: 512)
# nb_heads         Número de cabezas de atención (ej: 8)
# nb_layers        Número de capas Transformer (ej: 12)
# attn_span        Span máximo de atención (ej: 8192)
# emb_dropout      Dropout en embeddings (ej: 0.0)
# adapt_io_params  Configuración de Adaptive I/O (puede estar deshabilitado)


    
    def __init__(self, vocab_size, hidden_size, nb_heads, nb_layers,
                 attn_span, emb_dropout, adapt_io_params, **kargs):
        nn.Module.__init__(self)
        # token embeddings
        self.adapt_io = adapt_io_params['adapt_io_enabled']
        if self.adapt_io:
            # Adaptive Input Representations for Neural Language Modeling
            # Tokens frecuentes -> embeddings de alta dimensión (más capacidad)
            # Tokens raros -> embeddings de baja dimensión (menos parámetros)
            self.in_emb, self.out_emb = build_adaptive_io(
                vocab_size, hidden_size, **adapt_io_params)
        else:
            self.in_emb = nn.Embedding(vocab_size, hidden_size)
            self.out_emb = nn.Linear(hidden_size, vocab_size)
        if emb_dropout > 0:
            self.emb_dropout = nn.Dropout(emb_dropout)
        else:
            self.emb_dropout = None
        # position embeddings
        self.key_pe = nn.Parameter(
            torch.randn(1, hidden_size // nb_heads, attn_span))

        self.layers = nn.ModuleList()
        self.layers.extend(
            TransformerSeqLayer(
                hidden_size=hidden_size, nb_heads=nb_heads,
                attn_span=attn_span, **kargs)
            for _ in range(nb_layers))

    def forward(self, x, h_cache, target=None):
        # x:        # Token IDs (B, M) - tokens a procesar
        # h_cache:  # Lista de caches, uno por capa [(B, L, H), (B, L, H), ...]
        # target:   # Token IDs objetivo (B, M) - para calcular loss (opcional)
        
        # x size = B x M
        block_size = x.size(1)
        h = self.in_emb(x)  # B x M x H
        if self.emb_dropout is not None:
            h = self.emb_dropout(h)

        h_cache_next = []
        for l, layer in enumerate(self.layers):
            #Loop sobre capas
            cache_size = layer.attn.attn.get_cache_size()
            # Cada capa puede tener un cache diferente según span actual
            if cache_size > block_size:
            
                h_cache_next_l = torch.cat(
                    [h_cache[l][:, -cache_size + block_size:, :], h],
                    dim=1).detach()
            else:
                h_cache_next_l = h[:, -cache_size:, :].detach()
            h_cache_next.append(h_cache_next_l)
            h = layer(h, h_cache[l], self.key_pe)  # B x M x H

        if self.emb_dropout is not None:
            h = self.emb_dropout(h)
        if self.adapt_io:
            # loss is computed here
            out = self.out_emb(h, target)
            dummy_loss = compute_dummy_loss(self.in_emb, self.out_emb)
        else:
            out = F.log_softmax(self.out_emb(h), dim=-1)
            dummy_loss = None

        return out, h_cache_next, dummy_loss

