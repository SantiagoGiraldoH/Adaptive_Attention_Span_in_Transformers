"""
Configuración SMALL para Colab - Experimento real pero escalado
Entrena en ~2-3 horas en Colab, resultados razonables esperados
"""

PARAMS_CONFIG = {
    # Environment
    'env_params': {
        '--distributed': {
            'action': 'store_true',
            'default': False,
            'help': 'enable distributed training',
            'dest': 'distributed'
        },
        '--local_rank': {
            'type': int,
            'default': 0,
            'help': 'used in distributed training',
            'dest': 'local_rank'
        },
    },
    
    # Data
    'data_params': {
        '--data': {
            'type': str,
            'default': 'data/text8',
            'help': 'data location',
            'dest': 'data_path'
        },
        '--data-unit': {
            'type': str,
            'default': 'bpc',
            'choices': ['bpc', 'ppl'],
            'help': 'loss unit to log',
            'dest': 'data_unit'
        },
    },
    
    # Model - SMALL VERSION (similar al paper pero escalado)
    'model_params': {
        '--hid-sz': {
            'type': int,
            'default': 256,  # ← Paper usa 512, pero 256 es razonable
            'help': 'hidden size',
            'dest': 'hidden_size'
        },
        '--inner-hid-sz': {
            'type': int,
            'default': 1024,  # ← Mismo ratio que el paper (4x)
            'help': 'inner hidden size of FF layer',
            'dest': 'inner_hidden_size'
        },
        '--nlayers': {
            'type': int,
            'default': 8,  # ← Mismo que "small" del paper
            'help': 'number of layers',
            'dest': 'nb_layers'
        },
        '--block-sz': {
            'type': int,
            'default': 256,  # ← Paper usa 512, reducido para memoria
            'help': 'block size',
            'dest': 'block_size'
        },
        '--nheads': {
            'type': int,
            'default': 4,  # ← Paper usa 8, reducido
            'help': 'number of attention heads',
            'dest': 'nb_heads'
        },
        '--attn-span': {
            'type': int,
            'default': 2048,  # ← Paper usa 8192, reducido para memoria
            'help': 'max attention span',
            'dest': 'attn_span'
        },
        '--dropout': {
            'type': float,
            'default': 0.3,  # ← Mismo que el paper
            'help': 'dropout rate',
            'dest': 'dropout'
        },
        '--emb-dropout': {
            'type': float,
            'default': 0.,
            'help': 'embedding dropout rate',
            'dest': 'emb_dropout'
        },
    },
    
    # Optimization
    'optim_params': {
        '--lr': {
            'type': float,
            'default': 0.07,
            'help': 'learning rate',
            'dest': 'lr'
        },
        '--momentum': {
            'type': float,
            'default': 0,
            'help': 'SGD momentum',
            'dest': 'momentum'
        },
        '--optim': {
            'type': str,
            'default': 'adagrad',
            'help': 'optimizer: sgd | adagrad | adam',
            'dest': 'optim'
        },
        '--lr-warmup': {
            'type': int,
            'default': 8000,  # ← Paper usa 32000, escalado proporcionalmente
            'help': 'warmup steps',
            'dest': 'lr_warmup'
        },
        '--grad-clip': {
            'type': float,
            'default': 0.03,
            'help': 'gradient clipping',
            'dest': 'grad_clip'
        },
    },
    
    # Training - SMALL VERSION
    'trainer_params': {
        '--batch-sz': {
            'type': int,
            'default': 32,  # ← Paper usa 64, reducido para Colab
            'help': 'batch size',
            'dest': 'batch_size'
        },
        '--batch-split': {
            'type': int,
            'default': 2,  # ← Dividir batch para caber en memoria
            'help': 'split batch into smaller pieces',
            'dest': 'batch_split'
        },
        '--nbatches': {
            'type': int,
            'default': 500,  # ← Paper usa 1000, reducido
            'help': 'batches per iteration',
            'dest': 'nb_batches_per_iter'
        },
        '--niter': {
            'type': int,
            'default': 200,  # ← Paper usa 600, escalado
            'help': 'number of iterations',
            'dest': 'nb_iter'
        },
        '--checkpoint': {
            'type': str,
            'default': 'checkpoints/small_model.pt',
            'help': 'checkpoint path',
            'dest': 'checkpoint_path'
        },
        '--full-eval-mode': {
            'action': 'store_true',
            'default': False,
            'help': 'full evaluation mode',
            'dest': 'full_eval_mode'
        },
    },
    
    # Adaptive I/O - Desactivado para simplificar
    'adapt_io_params': {
        '--adapt-io': {
            'action': 'store_true',
            'default': False,
            'help': 'enable adaptive I/O',
            'dest': 'adapt_io_enabled'
        },
        '--adapt-io-tied': {
            'action': 'store_true',
            'default': False,
            'help': 'tie input/output parameters',
            'dest': 'adapt_io_tied'
        },
        '--adapt-io-divval': {
            'type': int,
            'default': 4,
            'help': 'dimension division value',
            'dest': 'adapt_io_divval'
        },
        '--adapt-io-cutoffs': {
            'type': list,
            'default': [20000, 40000, 200000],
            'help': 'cutoff values',
            'dest': 'adapt_io_cutoffs'
        },
    },
    
    # Adaptive Span
    'adapt_span_params': {
        '--adapt-span': {
            'action': 'store_true',
            'default': True,
            'help': 'enable adaptive attention span',
            'dest': 'adapt_span_enabled'
        },
        '--adapt-span-loss': {
            'type': float,
            'default': 0.000002,
            'help': 'loss coefficient for span lengths',
            'dest': 'adapt_span_loss'
        },
        '--adapt-span-ramp': {
            'type': int,
            'default': 32,
            'help': 'ramp size for soft masking',
            'dest': 'adapt_span_ramp'
        },
        '--adapt-span-init': {
            'type': float,
            'default': 0,
            'help': 'initial span ratio',
            'dest': 'adapt_span_init'
        },
        '--adapt-span-cache': {
            'action': 'store_true',
            'default': True,
            'help': 'adapt cache size to reduce memory',
            'dest': 'adapt_span_cache'
        },
    },
    
    # Persistent Memory
    'pers_mem_params': {
        '--pers-mem-size': {
            'type': int,
            'default': 0,
            'help': 'number of persistent memory vectors',
            'dest': 'pers_mem_size'
        },
    },
}

# Resumen
"""
SMALL CONFIG:
- Modelo: 8 capas, 256 hidden, 4 heads
- Atención: span=2048 con adaptive span
- Training: 200 iters × 500 batches = 100,000 steps
- Tiempo estimado: ~2-3 horas en Colab (T4 GPU)
- Propósito: Entrenar un modelo funcional con resultados razonables

Parámetros totales: ~8-10M (vs 38M del paper)
Resultado esperado: ~1.20-1.30 bpc en text8 (vs 1.11 del paper)
"""
