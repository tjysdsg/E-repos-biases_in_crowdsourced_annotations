from config import SNLI_DATA_FILES, PREMISE_KEY, HYPOTHESIS_KEY, PREMISE_ID_KEY, HYPOTHESIS_ID_KEY
from typing import Literal, Dict, List, Set, Tuple
import jsonlines as jsonl
from spacy.lang.en import English
import string
from document_stats import DocumentStats


class UnigramSNLIData:

    def __init__(self, split: Literal["train", "dev", "test"] = 'train'):
        self.split = split
        self.file = SNLI_DATA_FILES[split]

        self._read()

    def _read(self):
        nlp = English()

        self.data = {}  # type: Dict[str, List[List[str]]]
        self.unique_ids = {}  # type: Dict[str, Set[str]]
        with jsonl.open(self.file) as reader:
            for obj in reader:
                for sent_key, id_key in zip([PREMISE_KEY, HYPOTHESIS_KEY], [PREMISE_ID_KEY, HYPOTHESIS_ID_KEY]):
                    # only append unique sentences
                    self.unique_ids.setdefault(id_key, set())
                    sent_id = obj[id_key]
                    if sent_id in self.unique_ids[id_key]:
                        continue
                    self.unique_ids[id_key].add(sent_id)

                    # tokenize and remove punctuations
                    tokens = nlp(obj[sent_key])
                    tokens = [t.text for t in tokens if t.text not in string.punctuation]

                    self.data.setdefault(sent_key, []).append(tokens)

        # import json
        # with open('tmp.json', 'w') as f:
        #     json.dump(self.data, f, indent=2)

    def collect_stats(self, key=None) -> DocumentStats:
        """
        :param key: The sentence key, collect sentences from the entire corpus if None
        """
        res = DocumentStats()

        if key is None:
            sentences = self.data[PREMISE_KEY] + self.data[HYPOTHESIS_KEY]
        else:
            assert key in self.data
            sentences = self.data[key]

        res.n_sentences = len(sentences)

        for tokens in sentences:
            pairs = set()
            for src in tokens:
                src = src.lower()
                res.word_freq.setdefault(src, 0)
                res.word_freq[src] += 1

                res.vocab.add(src)

                for tgt in tokens:
                    tgt = tgt.lower()
                    if src == tgt:
                        continue

                    # multiple co-occurrence in the same sentence is counted as one
                    p = tuple(sorted([src, tgt]))  # type: Tuple[str, str]
                    if p in pairs:
                        continue
                    pairs.add(p)

                    res.co_freq.setdefault(p, 0)
                    res.co_freq[p] += 1

        return res
