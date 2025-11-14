"""
Configuración TINY para Colab - Solo para testing rápido
Entrena en ~10 minutos, no esperes buenos resultados
"""

PARAMS_CONFIG = {
    # Environment
    'env_params': {
        '--distributed': {
            'action': 'store_true',
            'default': False,  # ← Colab = 1 GPU, no distributed
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
            'default': 'data/text8',  # ← text8 es más pequeño que enwik8
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
    
    # Model - TINY VERSION
    'model_params': {
        '--hid-sz': {
            'type': int,
            'default': 128,  # ← Original: 256-512, reducido para velocidad
            'help': 'hidden size',
            'dest': 'hidden_size'
        },
        '--inner-hid-sz': {
            'type': int,
            'default': 512,  # ← Original: 1024-2048
            'help': 'inner hidden size of FF layer',
            'dest': 'inner_hidden_size'
        },
        '--nlayers': {
            'type': int,
            'default': 4,  # ← Original: 8-24, muy reducido
            'help': 'number of layers',
            'dest': 'nb_layers'
        },
        '--block-sz': {
            'type': int,
            'default': 128,  # ← Original: 512, reducido para velocidad
            'help': 'block size',
            'dest': 'block_size'
        },
        '--nheads': {
            'type': int,
            'default': 4,  # ← Original: 8
            'help': 'number of attention heads',
            'dest': 'nb_heads'
        },
        '--attn-span': {
            'type': int,
            'default': 512,  # ← Original: 1024-8192
            'help': 'max attention span',
            'dest': 'attn_span'
        },
        '--dropout': {
            'type': float,
            'default': 0.2,  # ← Original: 0.3
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
            'default': 0.07,  # ← Mismo que el paper
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
            'default': 'adagrad',  # ← Mismo que el paper
            'help': 'optimizer: sgd | adagrad | adam',
            'dest': 'optim'
        },
        '--lr-warmup': {
            'type': int,
            'default': 2000,  # ← Original: 32000, reducido
            'help': 'warmup steps',
            'dest': 'lr_warmup'
        },
        '--grad-clip': {
            'type': float,
            'default': 0.03,  # ← Mismo que el paper
            'help': 'gradient clipping',
            'dest': 'grad_clip'
        },
    },
    
    # Training - TINY VERSION
    'trainer_params': {
        '--batch-sz': {
            'type': int,
            'default': 16,  # ← Original: 64, reducido para caber en Colab
            'help': 'batch size',
            'dest': 'batch_size'
        },
        '--batch-split': {
            'type': int,
            'default': 1,  # ← No dividir batch (modelo pequeño)
            'help': 'split batch into smaller pieces',
            'dest': 'batch_split'
        },
        '--nbatches': {
            'type': int,
            'default': 100,  # ← Original: 1000, MUY reducido
            'help': 'batches per iteration',
            'dest': 'nb_batches_per_iter'
        },
        '--niter': {
            'type': int,
            'default': 50,  # ← Original: 600-900, MUY reducido
            'help': 'number of iterations',
            'dest': 'nb_iter'
        },
        '--checkpoint': {
            'type': str,
            'default': 'checkpoints/tiny_model.pt',
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
            'default': False,  # ← Desactivado
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
    
    # Adaptive Span - ACTIVADO
    'adapt_span_params': {
        '--adapt-span': {
            'action': 'store_true',
            'default': True,  # ← ACTIVADO por default
            'help': 'enable adaptive attention span',
            'dest': 'adapt_span_enabled'
        },
        '--adapt-span-loss': {
            'type': float,
            'default': 0.000002,  # ← Mismo que paper (small model)
            'help': 'loss coefficient for span lengths',
            'dest': 'adapt_span_loss'
        },
        '--adapt-span-ramp': {
            'type': int,
            'default': 32,  # ← Mismo que el paper
            'help': 'ramp size for soft masking',
            'dest': 'adapt_span_ramp'
        },
        '--adapt-span-init': {
            'type': float,
            'default': 0,  # ← Mismo que el paper
            'help': 'initial span ratio',
            'dest': 'adapt_span_init'
        },
        '--adapt-span-cache': {
            'action': 'store_true',
            'default': True,  # ← ACTIVADO
            'help': 'adapt cache size to reduce memory',
            'dest': 'adapt_span_cache'
        },
    },
    
    # Persistent Memory - Desactivado
    'pers_mem_params': {
        '--pers-mem-size': {
            'type': int,
            'default': 0,  # ← Desactivado (0 = no usar)
            'help': 'number of persistent memory vectors',
            'dest': 'pers_mem_size'
        },
    },
}

# Resumen de la configuración
"""
TINY CONFIG:
- Modelo: 4 capas, 128 hidden, 4 heads
- Atención: span=512 con adaptive span
- Training: 50 iters × 100 batches = 5,000 steps
- Tiempo estimado: ~10 minutos en Colab (T4 GPU)
- Propósito: Verificar que el código funciona

Parámetros totales: ~2-3M (vs 38M del paper)
"""
