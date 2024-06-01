import os

LUCKY_SEED = 42
TRAIN_FILE_COUNT = 43
VAL_FILE_COUNT = 12

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

OBJECTS_DIR = os.path.join(ROOT_DIR, "objects")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DATA_DIR = os.path.join(ROOT_DIR, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
ORIG_DATA_DIR = os.path.join(RAW_DATA_DIR, "sa-emotions")
OTHERS_RAW_DATA = os.path.join(RAW_DATA_DIR, "others")

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
COMPLEX_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "complex")
SIMPLE_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "simple")

TEST_DATA_DIR = os.path.join(DATA_DIR, "testing_data")

TRAIN_DATA_DIR = os.path.join(DATA_DIR, "training_data")
TRAIN_DATA_DIR_WI = os.path.join(TRAIN_DATA_DIR, "word_2_index")
TRAIN_DATA_DIR_TF_IDF = os.path.join(TRAIN_DATA_DIR, "tf_idf")

VAL_DATA_DIR = os.path.join(DATA_DIR, "validation_data")
VAL_DATA_DIR_WI = os.path.join(VAL_DATA_DIR, "word_2_index")
VAL_DATA_DIR_TF_IDF = os.path.join(VAL_DATA_DIR, "tf_idf")

SPACY_MEDIUM_MODEL = "en_core_web_md"
SPACY_LARGE_MODEL = "en_core_web_lg"
TF_HUB_EMBEDDING_MODELS = [
    "https://tfhub.dev/google/nnlm-en-dim128/2",
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
]

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(filename)s | %(lineno)d | %(message)s"
)
LOG_LEVEL = "DEBUG"
LOG_FILE = os.path.join(LOGS_DIR, "sentiment_analysis.log")
LOG_FILE_MAX_BYTES = 1048576
LOG_FILE_BACKUP_COUNT = 2
