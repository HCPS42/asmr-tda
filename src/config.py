# Random seed

SEED = 42

# Paths and filenames

DATA_PATH = '../data'
EEG_PATH = f'{DATA_PATH}/osfstorage-archive/EEG Data'
IMAGE_PATH = '../images'
INTERACTIVE = False

# EEG settings

N_EEG_CHANNELS = 64
EEG_CHANNELS = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
                'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
                'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
                'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
                'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
                'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
                'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

# Embedding settings

TIME_DELAY = 10
DIMENSION = 3
STRIDE = 10

# Training settings

N_SAMPLES = 10
TARGET = 'label'
TEST_SIZE = 0.1

HOMOLOGY_DIMENSIONS = [0, 1, 2]
