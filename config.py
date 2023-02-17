import os

IDENTITY_LABEL_FILE = 'identity_labels.txt'

SNLI_DATA_DIR = 'snli_1.0'

SNLI_TRAIN_FILE = os.path.join(SNLI_DATA_DIR, 'snli_1.0_train.jsonl')
SNLI_DEV_FILE = os.path.join(SNLI_DATA_DIR, 'snli_1.0_dev.jsonl')
SNLI_TEST_FILE = os.path.join(SNLI_DATA_DIR, 'snli_1.0_test.jsonl')

SNLI_DATA_FILES = {
    'train': SNLI_TRAIN_FILE,
    'dev': SNLI_DEV_FILE,
    'test': SNLI_TEST_FILE,
}

PREMISE_KEY = 'sentence1'
HYPOTHESIS_KEY = 'sentence2'
LABEL_KEY = 'gold_label'
PREMISE_ID_KEY = 'captionID'
HYPOTHESIS_ID_KEY = 'pairID'
