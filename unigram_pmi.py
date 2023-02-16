import json
from snli import UnigramSNLIData
from config import IDENTITY_LABEL_FILE, PREMISE_KEY, HYPOTHESIS_KEY
from pmi import PMI


def main():
    identity_labels = []
    with open(IDENTITY_LABEL_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n').strip()
            if len(line) > 0:
                identity_labels.append(line)

    data = UnigramSNLIData()

    doc_stats = data.collect_stats()
    pmi = PMI(doc_stats)
    pmi_stats = pmi(identity_labels)
    with open('pmi.json', 'w', encoding='utf-8') as f:
        json.dump(pmi_stats, f, indent=2)

    doc_stats = data.collect_stats(key=PREMISE_KEY)
    pmi = PMI(doc_stats)
    pmi_stats = pmi(identity_labels)
    with open('pmi_premise.json', 'w', encoding='utf-8') as f:
        json.dump(pmi_stats, f, indent=2)

    doc_stats = data.collect_stats(key=HYPOTHESIS_KEY)
    pmi = PMI(doc_stats)
    pmi_stats = pmi(identity_labels)
    with open('pmi_hypothesis.json', 'w', encoding='utf-8') as f:
        json.dump(pmi_stats, f, indent=2)


if __name__ == '__main__':
    main()
