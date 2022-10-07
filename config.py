import os
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parent
DATA_DIR = os.path.join(ROOT_PATH, 'data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train_data.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test_data.csv')
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, 'train_labels.csv')
MODEL_PATH = os.path.join(ROOT_PATH, 'models')

# Kaggle
# ROOT_PATH = Path(__file__).absolute().parent.parent
# DATA_DIR = os.path.join(ROOT_PATH, 'amex-parquet')
# TRAIN_PATH = os.path.join(DATA_DIR, 'train_fe.parquet')
# TEST_PATH = os.path.join(DATA_DIR, 'test_fe.parquet')
# TRAIN_LABELS_PATH = os.path.join(DATA_DIR, 'train_labels.csv')
# MODEL_PATH = os.path.join(ROOT_PATH, 'amex-models')

n_folds = 5
seed = 42
target = 'target'
