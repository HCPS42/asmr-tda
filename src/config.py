# Random seed

SEED = 42

# Paths and filenames

DATA_PATH = '../data'
EEG_PATH = f'{DATA_PATH}/osfstorage-archive/EEG Data'
ID_PATH = f'{DATA_PATH}/id.joblib'
INTERVAL_PATH = f'{DATA_PATH}/interval.joblib'
POINT_CLOUD_PATH = f'{DATA_PATH}/point_cloud.joblib'
LABEL_PATH = f'{DATA_PATH}/label.joblib'

# EEG settings

N_EEG_CHANNELS = 64

# Embedding settings

TIME_DELAY = 10
DIMENSION = 3
STRIDE = 10

REWRITE_PROCESSED_DATA = False # True if the emdedding settings have changed

# Training settings

N_SAMPLES = 10
TARGET = 'ASMR'

HOMOLOGY_DIMENSIONS = [0, 1, 2]