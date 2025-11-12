"""
Adaptive Attention Span in Transformers
Implementation for learning purposes
"""

__version__ = "0.1.0"
__author__ = "Santiago Giraldo Henao"

from .adaptive_span import AdaptiveSpan
from .models import AdaptiveTransformer

__all__ = [
    "AdaptiveSpan",
    "AdaptiveTransformer",
]
