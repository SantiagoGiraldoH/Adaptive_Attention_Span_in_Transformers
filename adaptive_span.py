"""
Adaptive Attention Span mechanism
implementación de la función de enmascaramiento
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveSpan(nn.Module):

# Este módulo implementa la función de soft masking que permite a cada cabeza aprednder su span de atención óptimo, reduciendo memoria y computación
# La función es: m_z(x) = min[max(1/R * (R + z - x), 0), 1]
# Donde: x: La distancia desde la posición actual
#        z: El span dinámico [0, S]
#        R: hiperparametro que controla la suavidad de la máscara   
# Args:
#        max_size: Tamaño máximo de atención posible
#        ramp_size: Tamaño de la "rampa" de transición. Controla qué tan gradual es la transición de 1 a 0 en la máscara (típicamente 32)
#        init_val: Valor inicial de la proporción del span 0 = mínimo, 1 = máximo
#        shape: Forma del parámetro aprendible. Permite tener diferentes spans para diferentes cabezas


  def __init__(self, max_size, ramp_size, init_val=0, shape=(1,)):
          nn.Module.__init__(self)
          self._max_size = max_size
          self._ramp_size = ramp_size
          self.current_val = nn.Parameter(torch.zeros(*shape) + init_val) # Crea el parámetro aprendible que determinará el span.
          mask_template = torch.linspace(1 - max_size, 0, steps=max_size) # Crea una plantilla que va de (1-max_size) hasta 0. Esto representa las distancias relativas.
          self.register_buffer('mask_template', mask_template) # Registra la plantilla como buffer (no es parámetro aprendible, pero se guarda con el modelo).
  
  def forward(self, x):
  # Aplica la máscara suave al tensor de entrada
    
          mask = self.mask_template + self.current_val * self._max_size 
  # ejemplo:  Si `current_val = 0.5` y `max_size = 100`, entonces el span efectivo es 50
  # Sumamos a la plantilla: transladamos la función
    
          mask = mask / self._ramp_size + 1
  # Dividimos por `ramp_size` para controlar la pendiente de la transición
  # Sumamos 1 para centrar la función
    
          mask = mask.clamp(0, 1)
          if x.size(-1) < self._max_size:
              # the input could have been trimmed beforehand to save computation
              mask = mask[:, :, -x.size(-1):]
          x = x * mask
              # Multiplicamos elemento a elemento: enmascaramos la entrada
          return x
  
  def get_current_max_size(self, include_ramp=True):
  # Calcula el span máximo actual entre todas las cabezas.
  # Uso: Determinar cuánta memoria se necesita reservar para el caché.
          current_size = math.ceil(self.current_val.max().item() * self._max_size)
          if include_ramp:
              current_size += self._ramp_size
          current_size = max(0, min(self._max_size, current_size))
          return current_size
    
  def get_current_avg_size(self, include_ramp=True):
  # Calcula el span promedio entre todas las cabezas.
  # Uso: Métricas y logging. Permite ver cuál es el span típico que el modelo está usando.
          current_size = math.ceil(self.current_val.mean().item() * self._max_size)
          if include_ramp:
              current_size += self._ramp_size
          current_size = max(0, min(self._max_size, current_size))
          return current_size
  
  def clamp_param(self):
  # Asegura que `current_val` permanezca en el rango válido [0, 1].
  # Es necesario porque durante el entrenamiento con gradient descent, el parámetro podría salirse del rango [0, 1]. Esta función lo restringe.
          """this need to be called after each update"""
          self.current_val.data.clamp_(0, 1)


class AdaptiveSpan(nn.Module):
  # Orquesta el mecanismo completo para un Transformer. 
 
  def __init__(self, attn_span, adapt_span_loss, adapt_span_ramp,
               adapt_span_init, adapt_span_cache, nb_heads, **kargs):
  #  Inicializa el mecanismo de adaptive span con todos sus hiperparámetros.
  #  Args:
  # attn_span: Span máximo permitido (ej: 8192). Es el límite superior.
  # adapt_span_loss: Coeficiente λ para la regularización L1 del span (típicamente ~2e-6)
  # adapt_span_ramp: Tamaño de la rampa de transición suave (típicamente 32)
  # adapt_span_init: Valor inicial del span como proporción (0 = mínimo, 1 = máximo)
  # adapt_span_cache: Booleano. Si True, adapta el tamaño del cache para ahorrar memoria
  # nb_heads: Número de cabezas de atención (típicamente 8)
  
  # Qué almacena:
  
  # self._adapt_cache: Flag para saber si optimizar memoria del cache
  # self._max_span: El límite superior de atención
  # self._loss_coeff: Para la regularización
  # self._nb_heads: Número de cabezas
  # self._mask: La máscara adaptativa con forma (nb_heads, 1, 1) - cada cabeza tiene su propio parámetro de span
  
  # Nota importante: shape=(nb_heads, 1, 1) significa que cada cabeza aprenderá independientemente su span óptimo.
  
        nn.Module.__init__(self)
        self._adapt_cache = adapt_span_cache
        self._max_span = attn_span
        self._loss_coeff = adapt_span_loss
        self._nb_heads = nb_heads
        self._mask = AdaptiveMask(max_size=self._max_span,
                                 ramp_size=adapt_span_ramp,
                                 init_val=adapt_span_init,
                                 shape=(nb_heads, 1, 1))

  def forward(self, attn, normalize=True):
    # Aplica la máscara adaptativa a los pesos de atención.
    # Entrada: attn con forma (B*nb_heads, M, L) donde:
    # B: batch size real
    # nb_heads: número de cabezas
    # M: tamaño del bloque (tokens procesados en paralelo)
    # L: tamaño del span de atención
    
          """mask attention with the right span"""
          # batch and head dimensions are merged together, so separate them first
          B = attn.size(0) # batch size
          M = attn.size(1) # block size
          attn = attn.reshape(B // self._nb_heads, self._nb_heads, M, -1)
    # Separa las dimensiones de batch y heads
  
          attn = self._mask(attn)
    # Aplica la máscara adaptativa
          if normalize:
              attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)  # normalize so sum is 1
    # Después de enmascarar, la suma de los pesos de atención ya no es 1
    # Re-normalizamos para que sumen 1 nuevamente
          attn = attn.view(B, M, -1)
    # Vuelve a fusionar batch y heads: `(batch_size * nb_heads, M, L)`
          return attn

  def get_trim_len(self):
        """how much of memory can be trimmed to reduce computation"""
        L = self._max_span
        trim_len = min(L - 1, L - self._mask.get_current_max_size())
        # too fine granularity might be bad for the memory management
        trim_len = math.floor(trim_len / 64) * 64
        return trim_len
 # **Propósito**: Calcula cuántos tokens antiguos del cache pueden descartarse para ahorrar cómputo.

  def trim_memory(self, query, key, value, key_pe):
      """trim out unnecessary memory beforehand to reduce computation"""
      trim_len = self.get_trim_len()
      cache_size = key.size(1) - query.size(1)
      trim_len_cache = trim_len - (self._max_span - cache_size)
      if trim_len_cache > 0:
          key = key[:, trim_len_cache:, :]
          value = value[:, trim_len_cache:, :]
      elif trim_len_cache < 0:
          # cache is too short! this happens when validation resumes
          # after a lot of updates.
          key = F.pad(key, [0, 0, -trim_len_cache, 0])
          value = F.pad(value, [0, 0, -trim_len_cache, 0])
      if trim_len > 0:
          if key_pe is not None:
              key_pe = key_pe[:, :, trim_len:]
      return key, value, key_pe
  # Recorta la memoria (key, value, position embeddings) eliminando tokens que nunca serán atendidos.

  def get_cache_size(self):
      """determine how long the cache should be"""
      if self._adapt_cache:
          trim_len = self.get_trim_len()
          # give a buffer of 64 steps since a span might increase
          # in future updates
          return min(self._max_span, self._max_span - trim_len + 64)
      else:
          return self._max_span
  # Determina cuántos tokens históricos mantener en el cache.
  def get_loss(self):
      """a loss term for regularizing the span length"""
      return self._loss_coeff * self._max_span * self._mask.current_val.mean()
  # Calcula el término de regularización L1 para penalizar spans grandes
  # L_span = λ * Σ z_i
  # donde z_i es el span de la cabeza i.
  
  
  def get_current_max_span(self):
      return self._mask.get_current_max_size()
  
  def get_current_avg_span(self):
      return self._mask.get_current_avg_size()
  
  def clamp_param(self):
      self._mask.clamp_param()













