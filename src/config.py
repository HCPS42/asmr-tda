# Random seed

SEED = 42

# Paths and filenames

DATA_PATH = '../data'
EEG_PATH = f'{DATA_PATH}/osfstorage-archive/EEG Data'

# EEG settings

N_EEG_CHANNELS = 64

# Embedding settings

TIME_DELAY = 10
DIMENSION = 3
STRIDE = 10

# Training settings

N_SAMPLES = 10
TARGET = 'label'
TEST_SIZE = 0.1

HOMOLOGY_DIMENSIONS = [0, 1, 2]