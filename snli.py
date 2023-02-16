from config import SNLI_DATA_FILES, PREMISE_KEY, HYPOTHESIS_KEY, PREMISE_ID_KEY, HYPOTHESIS_ID_KEY
from typing import Literal, Dict, List, Set, Tuple
import jsonlines as jsonl
import spacy
import string
from document_stats import DocumentStats
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex


def get_tokenizer():
    nlp = spacy.load("en_core_web_sm")

    # prevent spacy from splitting hyphens
    # see spacy.lang.punctuation
    infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS), <---
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
    )
    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer
    return nlp


class UnigramSNLIData:

    def __init__(self, split: Literal["train", "dev", "test"] = 'train'):
        self.split = split
        self.file = SNLI_DATA_FILES[split]

        self._read()

    def _read(self):
        nlp = get_tokenizer()

        sentences = {}  # type: Dict[str, List[str]]
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

                    sentences.setdefault(sent_key, []).append(obj[sent_key])

        # batched tokenization and punctuation removal
        self.data = {}  # type: Dict[str, List[List[str]]]
        for sent_key in [PREMISE_KEY, HYPOTHESIS_KEY]:
            for doc in nlp.pipe(
                    sentences[sent_key],
                    n_process=8,
                    batch_size=10000,
                    disable=["tok2vec", "parser", "ner", "entity_linker", "entity_ruler"],
            ):
                tokens = [t.text for t in doc if t.text not in string.punctuation and not t.is_stop]
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
