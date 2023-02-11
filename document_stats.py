from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DocumentStats:
    word_freq: Dict[str, int] = field(default_factory=dict)
    co_freq: Dict[str, Dict[str, int]] = field(default_factory=dict)
    n_sentences: int = 0
