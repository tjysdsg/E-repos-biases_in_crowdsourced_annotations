from typing import Optional, Iterable, Tuple
from document_stats import DocumentStats
import math


class PMI:
    def __init__(self, doc_stats: DocumentStats):
        self.doc_stats = doc_stats

    def __call__(self, src_words: Iterable[str], target_words: Optional[Iterable[str]] = None, threshold=10,
                 include_scores=False, top_k=20):
        res = {}
        for src in src_words:
            src = src.lower()

            if src not in self.doc_stats.word_freq:
                continue
            res[src] = {}

            targets = target_words if target_words is not None else list(self.doc_stats.vocab)
            for tgt in targets:
                tgt = tgt.lower()

                if src == tgt:
                    continue

                p = tuple(sorted([src, tgt]))  # type: Tuple[str, str]
                if p not in self.doc_stats.co_freq:
                    continue

                co_count = self.doc_stats.co_freq[p]
                src_count = self.doc_stats.word_freq[src]
                tgt_count = self.doc_stats.word_freq[tgt]
                if src_count < threshold or tgt_count < threshold:
                    continue

                pmi = math.log2(
                    self.doc_stats.n_sentences * co_count / (src_count * tgt_count)
                )
                res[src][tgt] = pmi

        # sort targets by pmi
        ret = {}
        for src in res:
            d = sorted(res[src].items(), key=lambda x: x[1], reverse=True)
            d = d[:top_k]
            if include_scores:
                ret[src] = dict(d)
            else:
                ret[src] = [x[0] for x in d]
        return ret
