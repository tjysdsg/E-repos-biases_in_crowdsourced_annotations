from config import SNLI_DATA_FILES, PREMISE_KEY, HYPOTHESIS_KEY
from typing import Literal, Dict, List, Set, Tuple
import jsonlines as jsonl
import spacy
from spacy.tokens import Token
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
        with jsonl.open(self.file) as reader:
            for obj in reader:
                for sent_key in [PREMISE_KEY, HYPOTHESIS_KEY]:
                    sentences.setdefault(sent_key, []).append(obj[sent_key])

        # remove duplicated sentences
        for key in sentences:
            sentences[key] = list(set(sentences[key]))

        # batched tokenization and punctuation removal
        self.data = {}  # type: Dict[str, List[List[Token]]]
        for sent_key in [PREMISE_KEY, HYPOTHESIS_KEY]:
            for doc in nlp.pipe(
                    sentences[sent_key],
                    # n_process=8,
                    # batch_size=10000,
                    disable=["tok2vec", "parser", "ner", "entity_linker", "entity_ruler"],
            ):
                tokens = [t for t in doc]
                self.data.setdefault(sent_key, []).append(tokens)

    def collect_stats(self, key=None, bigram=False) -> DocumentStats:
        """
        :param key: The sentence key, collect sentences from the entire corpus if None
        :param bigram: Collect bi-gram information
        """
        res = DocumentStats()

        if key is None:
            sentences = self.data[PREMISE_KEY] + self.data[HYPOTHESIS_KEY]
        else:
            assert key in self.data
            sentences = self.data[key]

        res.n_sentences = len(sentences)

        def add_vocab(word: str):
            res.word_freq.setdefault(word, 0)
            res.word_freq[word] += 1
            res.vocab.add(word)
            # print(f"WORD: {word}")

        def add_pair(word1: str, word2: str):
            # shouldn't have overlapping unigrams
            s1 = set(word1.split())
            s2 = set(word2.split())
            if len(s1 & s2) > 0:
                return

            # multiple co-occurrence in the same sentence is counted as one
            p = tuple(sorted([word1, word2]))  # type: Tuple[str, str]

            if p not in pairs:
                pairs.add(p)
                res.co_freq.setdefault(p, 0)
                res.co_freq[p] += 1
                # print(f"PAIR: {word1} + {word2}")

        def is_word(word: Token):
            return not word.is_stop and not word.is_punct

        def tok2text(word: Token):
            return word.text.lower()

        for tokens in sentences:
            # start with unigram
            ngrams: List[str] = [tok2text(t) for t in tokens if is_word(t)]
            if bigram:  # add bigrams
                for i, t in enumerate(tokens):
                    if i == 0:
                        continue

                    prev = tokens[i - 1]
                    if is_word(prev) and is_word(t):
                        ngrams.append(f'{tok2text(prev)} {tok2text(t)}')

            pairs = set()
            for i, src in enumerate(ngrams):
                add_vocab(src)

                if i + 1 >= len(ngrams):
                    break

                for tgt in ngrams[i + 1:]:
                    add_pair(src, tgt)

        return res
