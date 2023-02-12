import json
from snli import UnigramSNLIData
from config import IDENTITY_LABEL_FILE
from pmi import PMI
from dataclasses import asdict


def main():
    identity_labels = []
    with open(IDENTITY_LABEL_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n').strip()
            if len(line) > 0:
                identity_labels.append(line)

    data = UnigramSNLIData()
    doc_stats = data.collect_stats()

    # with open('document_stats.json', 'w', encoding='utf-8') as f:
    #     json.dump(asdict(doc_stats), f, indent=2)

    pmi = PMI(doc_stats)
    pmi_stats = pmi(identity_labels)

    with open('pmi.json', 'w', encoding='utf-8') as f:
        json.dump(pmi_stats, f, indent=2)


if __name__ == '__main__':
    main()
