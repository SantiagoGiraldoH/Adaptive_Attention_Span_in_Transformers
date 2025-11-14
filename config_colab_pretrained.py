
3️⃣ config_colab_pretrained.py - Para evaluar modelos pre-entrenados
python"""
Configuración para evaluar modelos PRE-ENTRENADOS del paper
NO entrena, solo evalúa
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
            'default': 'data/text8',  # O 'data/enwik8'
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
    
    # Model - DEBE COINCIDIR CON EL CHECKPOINT
    'model_params': {
        '--hid-sz': {
            'type': int,
            'default': 512,  # ← Small model del paper
            'help': 'hidden size',
            'dest': 'hidden_size'
        },
        '--inner-hid-sz': {
            'type': int,
            'default': 2048,
            'help': 'inner hidden size of FF layer',
            'dest': 'inner_hidden_size'
        },
        '--nlayers': {
            'type': int,
            'default': 12,  # ← Small model del paper
            'help': 'number of layers',
            'dest': 'nb_layers'
        },
        '--block-sz': {
            'type': int,
            'default': 512,
            'help': 'block size',
            'dest': 'block_size'
        },
        '--nheads': {
            'type': int,
            'default': 8,
            'help': 'number of attention heads',
            'dest': 'nb_heads'
        },
        '--attn-span': {
            'type': int,
            'default': 8192,  # ← Small model del paper
            'help': 'max attention span',
            'dest': 'attn_span'
        },
        '--dropout': {
            'type': float,
            'default': 0.3,
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
    
    # Optimization - No importa para eval
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
            'help': 'optimizer',
            'dest': 'optim'
        },
        '--lr-warmup': {
            'type': int,
            'default': 32000,
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
    
    # Training - Solo eval
    'trainer_params': {
        '--batch-sz': {
            'type': int,
            'default': 8,  # ← Batch pequeño para eval completo
            'help': 'batch size',
            'dest': 'batch_size'
        },
        '--batch-split': {
            'type': int,
            'default': 1,
            'help': 'split batch',
            'dest': 'batch_split'
        },
        '--nbatches': {
            'type': int,
            'default': 1000,  # ← No importa en eval mode
            'help': 'batches per iteration',
            'dest': 'nb_batches_per_iter'
        },
        '--niter': {
            'type': int,
            'default': 1,  # ← No importa en eval mode
            'help': 'number of iterations',
            'dest': 'nb_iter'
        },
        '--checkpoint': {
            'type': str,
            'default': 'checkpoints/text8.pt',  # ← Modelo descargado
            'help': 'checkpoint path',
            'dest': 'checkpoint_path'
        },
        '--full-eval-mode': {
            'action': 'store_true',
            'default': True,  # ← ACTIVADO por default
            'help': 'full evaluation mode',
            'dest': 'full_eval_mode'
        },
    },
    
    # Adaptive I/O
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
            'default': 0.0000005,  # ← Valor del paper
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
            'help': 'adapt cache size',
            'dest': 'adapt_span_cache'
        },
    },
    
    # Persistent Memory
    'pers_mem_params': {
        '--pers-mem-size': {
            'type': int,
            'default': 0,
            'help': 'persistent memory size',
            'dest': 'pers_mem_size'
        },
    },
}

# Resumen
"""
PRETRAINED CONFIG:
- Carga modelo pre-entrenado del paper
- Evalúa en test set completo
- NO entrena, solo inferencia
- Tiempo: ~10-15 minutos
- Propósito: Verificar reproducibilidad del paper

Resultado esperado:
- text8.pt: ~1.11 bpc en test
- enwik8.pt: ~1.02 bpc en test
"""
