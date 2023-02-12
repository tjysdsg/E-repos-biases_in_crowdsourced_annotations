from dataclasses import dataclass, field
from typing import Dict, Tuple, Set


@dataclass
class DocumentStats:
    word_freq: Dict[str, int] = field(default_factory=dict)
    co_freq: Dict[Tuple[str, str], int] = field(default_factory=dict)
    vocab: Set[str] = field(default_factory=set)
    n_sentences: int = 0
