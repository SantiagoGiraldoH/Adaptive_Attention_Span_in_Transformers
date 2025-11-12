# Adaptive_Attention_Span_in_Transformers
Este proyecto replica la implementación del paper "Adaptive Attention Span in Transformers" (Sukhbaatar et al., 2019) de Facebook AI Research. El paper propone un mecanismo innovador de auto-atención que permite a cada cabeza de atención aprender su span óptimo de forma independiente, reduciendo significativamente el costo computacional y de memoria mientras mantiene o mejora el rendimiento.

Objetivos específicos:
1. Implementar soft mask function
2. Entrenar parámetros z(i) por cabeza de atención
3. Analizar span aprendido en diferentes capas
4. Implementar regularización L1
5. Evaluar reducción de FLOPs manteniendo calidad

Link del trabajo original: https://arxiv.org/abs/1905.07799
