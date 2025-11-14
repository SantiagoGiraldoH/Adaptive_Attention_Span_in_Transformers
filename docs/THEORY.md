
AdaptiveMask es el componente fundamental del mecanismo de adaptive attention span. Esta clase implementa una función de enmascaramiento suave (soft masking) que permite aprender de manera diferenciable cuántos tokens del pasado debería atender cada cabeza de atención.
Concepto clave: En lugar de usar un span fijo (ej: siempre ver los últimos 512 tokens), esta máscara aprende un parámetro z que determina dinámicamente cuántos tokens son relevantes. La transición de "atender" a "no atender" es gradual (suave), lo que permite el aprendizaje mediante backpropagation.

**Propósito**: Aplica la máscara suave al tensor de entrada (típicamente pesos de atención).

**Proceso paso a paso**:

1. **mask = self.mask_template + self.current_val * self._max_size`**:
   - self.current_val está en [0, 1], lo multiplicamos por max_size para obtener el span actual
   - Si current_val = 0.5 y max_size = 100, entonces el span efectivo es 50
   - Sumamos a la plantilla: transladamos la función

2. **`mask = mask / self._ramp_size + 1`**:
   - Dividimos por `ramp_size` para controlar la pendiente de la transición
   - Sumamos 1 para centrar la función

3. **`mask = mask.clamp(0, 1)`**:
   - Limitamos los valores entre 0 y 1
   - Valores fuera del span efectivo → 0
   - Valores dentro del span efectivo → 1
   - Valores en la rampa → entre 0 y 1 (transición suave)

4. **Recorte condicional**:
   - Si la entrada es más pequeña que `max_size`, ajustamos la máscara para que coincida

5. **`x = x * mask`**:
   - Multiplicamos elemento a elemento: enmascaramos la entrada

**Ejemplo visual**:

Supongamos max_size=10, ramp_size=2, current_val=0.6 (span=6)

Posiciones:     -9  -8  -7  -6  -5  -4  -3  -2  -1   0
Mask:            0   0   0   0  0.5  1   1   1   1   1


AdaptiveSpan es la clase que orquesta el mecanismo completo de adaptive attention span para un Transformer. Mientras que AdaptiveMask es el componente básico de enmascaramiento, AdaptiveSpan lo integra en el contexto de la atención multi-cabeza, gestionando:

Múltiples máscaras (una por cabeza de atención)
Regularización del span mediante una pérdida auxiliar
Optimización de memoria mediante cache adaptativo
Recorte de memoria innecesaria para reducir cómputo

**Flujo visual**:

Input:  (B*H, M, L)  [batch*heads fusionados]
   ↓
Reshape: (B, H, M, L)  [separar batch y heads]
   ↓
Mask:    (B, H, M, L)  [aplicar máscara por cabeza]
   ↓
Normalize: (B, H, M, L)  [re-normalizar]
   ↓
Reshape: (B*H, M, L)  [fusionar de nuevo]


Regularización L1 para penalizar spans grandes.

L_span = λ * Σ z_i
donde z_i es el span de la cabeza i.
loss = λ * max_span * mean(current_val)

Por qué es importante:

Sin regularización, todas las cabezas tenderían a usar el span máximo
Esta pérdida incentiva spans pequeños (eficiencia)
El modelo encuentra el balance óptimo entre:

Span grande: mejor rendimiento pero más cómputo

## Diagrama de flujo completo
Inicialización:
  AdaptiveSpan crea AdaptiveMask con (nb_heads, 1, 1) parámetros
  Cada cabeza tiene su propio z ∈ [0, 1]
  
Durante Forward:
  1. trim_memory() → recorta cache innecesario
  2. [cómputo de atención en models.py]
  3. forward() → aplica máscaras adaptativas
  4. Cada cabeza enmascara según su span aprendido
  
Durante Backward:
  1. get_loss() → calcula penalización L1
  2. Gradientes fluyen a través de current_val
  3. optimizer.step() → actualiza spans
  4. clamp_param() → asegura z ∈ [0, 1]
  
Gestión de memoria:
  - get_cache_size() → determina memoria a reservar
  - get_trim_len() → optimiza cómputo


Clase SeqAttention(nn.Module)
Esta es la clase más importante - donde ocurre la magia de la atención adaptativa.
Resumen General
SeqAttention implementa self-attention secuencial con span adaptativo. Características clave:

Atiende solo a tokens anteriores (autoregresivo)
Cada token atiende a un número variable y aprendible de pasos previos
Usa position embeddings relativos para capturar orden
Opcionalmente integra persistent memory (para el paper de all-attention networks)


Resumen de SeqAttention
Input: query (tokens actuales), key/value (historia + actuales), key_pe

1. [Opcional] trim_memory() → Optimizar memoria
2. attn_content = query · key → Atención por contenido
3. attn_pos = query · key_pe → Atención por posición
4. attn = attn_content + attn_pos → Combinar
5. attn = softmax(attn / √H) → Normalizar
6. [ADAPTIVE SPAN ] attn = adaptive_span(attn)
7. attn = dropout(attn) → Regularizar
8. output = attn · value → Aplicar atención

Output: representaciones enriquecidas (B, M, H)


## Diagrama Completo del Forward

Input: h (B, M, H), h_cache (B, L, H), key_pe

┌─────────────────────────────────────┐
│ Paso 1: Concatenar                  │
│ h_all = [h_cache | h]               │
│ (B, L+M, H)                         │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ Paso 2: Multi-Head Attention        │
│ attn_out = attn(h, h_all, h_all)   │
│ (B, M, H)                           │
│                                     │
│ Dentro: ⭐ Adaptive Span ⭐         │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ Paso 3: Residual + LayerNorm        │
│ h = norm1(h + attn_out)             │
│                                     │
│     h ──────┐                       │
│     ↓       ↓                       │
│   attn    (+)                       │
│     ↓       ↓                       │
│  attn_out  norm1                    │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ Paso 4: FeedForward                 │
│ ff_out = ff(h)                      │
│ (B, M, H)                           │
│                                     │
│ H → 4H → H con ReLU                 │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ Paso 5: Residual + LayerNorm        │
│ out = norm2(h + ff_out)             │
│                                     │
│     h ──────┐                       │
│     ↓       ↓                       │
│    ff      (+)                      │
│     ↓       ↓                       │
│  ff_out   norm2                     │
└─────────────────────────────────────┘
           ↓
      Output (B, M, H)




TransformerSeq es el modelo completo end-to-end que:

Convierte tokens (IDs) en embeddings
Añade position embeddings
Procesa a través de N capas de Transformer (con adaptive span)
Genera probabilidades de siguiente token
Maneja el cache para procesamiento secuencial eficiente

Es lo que entrenaremos y usaremos para generación de texto.

## Flujo Completo del Forward

Input:
  x: (2, 512)           # Token IDs
  h_cache: [12 tensors] # Caches (uno por capa)
  target: (2, 512)      # Targets (opcional)

↓ [Embedding]
h = in_emb(x)
h: (2, 512, 512)        # Vectores densos

↓ [Capa 0]
h = layer_0(h, h_cache[0], key_pe)
h_cache_next[0] = actualizar_cache(h, h_cache[0])

↓ [Capa 1]
h = layer_1(h, h_cache[1], key_pe)
h_cache_next[1] = actualizar_cache(h, h_cache[1])

...

↓ [Capa 11]
h = layer_11(h, h_cache[11], key_pe)
h_cache_next[11] = actualizar_cache(h, h_cache[11])

↓ [Output]
logits = out_emb(h)     # (2, 512, 27)
out = log_softmax(logits, dim=-1)

↓ [Return]
return out, h_cache_next, None


### Flujo Completo

main.py
  └─> train_iteration(eval_only=False, nb_batches=1000)
       │
       ├─> Loop: 1000 batches
       │    │
       │    └─> _train_batch()
       │         │
       │         ├─> optimizer.zero_grad()
       │         │
       │         └─> _train_step()
       │              │
       │              ├─> model.forward(X, h_cache, target=Y)
       │              │    └─> Devuelve: (out, h_cache_next, dummy_loss)
       │              │
       │              ├─> Calcula loss principal
       │              │
       │              ├─> loss += adaptive_span_loss  ← SOLO EN TRAIN
       │              │    └─> Penaliza spans largos
       │              │
       │              └─> loss.backward()  ← GRADIENTES
       │
       └─> optimizer.step()  ← ACTUALIZA PESOS
       └─> adaptive_span.clamp_param()  ← z ∈ [0,1]


### 🎯 RESUMEN CONCEPTUAL

 **Training:**

Procesar bloque → Calcular loss → Agregar adaptive_span_loss → 
Backward → Actualizar pesos → Clamp parámetros → Siguiente bloque


 **Eval (rápido):**
Procesar 10% de bloques → Calcular loss → 
Log para monitorear → (NO tocar pesos)


### Data

## 📊 FLUJO COMPLETO DE DATOS

1. get_train_val_test_data()
    ↓
2. _build_corpus()
    ↓
3. Dictionary('train.txt', sort_dict)
    - Construye vocabulario
    ↓
4. _tokenize() para train/val/test
    - Convierte texto → índices
    ↓
5. _batchify()
    - Reorganiza: (seq_len,) → (batch_size, seq_per_batch)
    ↓
6. División para distributed (si aplica)
    - Cada GPU toma su slice
    ↓
7. .to(device)
    - Mover a GPU
    ↓
8. Retornar tensors listos para entrenamiento


## 🎯 FLUJO COMPLETO RESUMIDO

1. Parse argumentos (config.py)
    ↓
2. Setup environment (GPUs, distributed)
    ↓
3. Load data (tokenize, batchify, to device)
    ↓
4. Create model (TransformerSeq con adaptive span)
    ↓
5. Wrap con DistributedDataParallel
    ↓
6. Create optimizer (Adagrad) y scheduler (warmup)
    ↓
7. Create logger
    ↓
8. Load checkpoint (si existe)
    ↓
9. Initialize cache (zeros)
    ↓
10. Training loop:
    ├─> Train iteration (1000 batches)
    │   ├─> Forward
    │   ├─> Backward
    │   ├─> Optimizer step
    │   └─> Update cache
    │
    ├─> Val iteration (100 batches)
    │   ├─> Forward (sin backward)
    │   └─> Update cache
    │
    ├─> Aggregate results (distributed)
    ├─> Log metrics
    └─> Save checkpoint
    ↓
11. (Opcional) Full eval en test set


### FLOPS

Los FLOPS (Floating Point Operations) son operaciones matemáticas de punto flotante que una computadora realiza. En el contexto de redes neuronales, se usan para medir el costo computacional de un modelo.
¿Qué son los FLOPS?
Son operaciones básicas como:

Sumas
Multiplicaciones
Divisiones
Exponenciales

Por ejemplo, multiplicar dos matrices de tamaño n x n requiere aproximadamente 2n³ FLOPS (n³ multiplicaciones + n³ sumas).
¿Cómo se calculan en Transformers?
En el paper, los FLOPS se calculan principalmente para dos componentes:
1. Self-Attention Layer
Para cada token en una secuencia de longitud M con span S:

Query-Key similarity: 2 × M × S × d FLOPS

M queries × S keys × d dimensiones × 2 (mult + suma)


Attention-Value: 2 × M × S × d FLOPS
Total por cabeza: 4 × M × S × d
Total con H cabezas: 4 × M × S × d × H

2. Feed-Forward Layer

Primera capa lineal: 2 × M × d × d_ff
Segunda capa lineal: 2 × M × d_ff × d
Total: 4 × M × d × d_ff


