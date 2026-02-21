from .configuration_graphmert import GraphMertConfig
from .collating_graphmert import (
    GraphMertDataCollator,
    GraphMertDataCollatorForLanguageModeling,
    GraphMertDataCollatorForMultipleChoice,
)

try:
    _torch_available = True
except ImportError:
    _torch_available = False

if _torch_available:
    from .modeling_graphmert import (
        GraphMertForMultipleChoice,
        GraphMertForSequenceClassification,
        GraphMertForMaskedLM,
        GraphMertModel,
        GraphMertPreTrainedModel,
    )

__all__ = [
    "GraphMertConfig",
    "GraphMertDataCollator",
    "GraphMertDataCollatorForLanguageModeling",
    "GraphMertDataCollatorForMultipleChoice",
]

if _torch_available:
    __all__.extend([
        "GraphMertForMultipleChoice",
        "GraphMertForSequenceClassification",
        "GraphMertForMaskedLM",
        "GraphMertModel",
        "GraphMertPreTrainedModel",
    ])
