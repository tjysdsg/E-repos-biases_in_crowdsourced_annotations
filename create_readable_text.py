from config import PREMISE_KEY, HYPOTHESIS_KEY, SNLI_TRAIN_FILE, LABEL_KEY
import jsonlines as jsonl


def main():
    data = []
    with jsonl.open(SNLI_TRAIN_FILE) as reader:
        for obj in reader:
            data.append([
                obj[PREMISE_KEY],
                obj[HYPOTHESIS_KEY],
                obj[LABEL_KEY],
            ])

    with open('data.txt', 'w', encoding='utf-8') as f:
        for premise, hyp, label in data:
            f.write(f'\n{premise}\n{hyp}\n{label}\n')


if __name__ == '__main__':
    main()
