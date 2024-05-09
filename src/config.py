# Random seed

SEED = 42

# Paths and filenames

DATA_PATH = '../data'
EEG_PATH = f'{DATA_PATH}/osfstorage-archive/EEG Data'
IMAGE_PATH = '../images'
INTERACTIVE = False

# EEG settings

ALL_EEG_CHANNELS = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
                'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
                'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
                'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
                'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
                'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
                'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'] # 64 channels

# Embedding settings

TIME_DELAY = 10
DIMENSION = 3
STRIDE = 10

# Training settings

#CHANNELS = ['T7', 'T8', 'TP7', 'TP8', 'P3', 'P4', 'Pz', 'O1', 'O2', 'F3', 'F4', 'Fpz']
CHANNELS = ['Fp1']

TARGET_ID = 1
ALL_TARGETS = ['label', 'ASMR']

TARGET = ALL_TARGETS[TARGET_ID]
N_INTERVALS_PER_PERSON_PER_CLASS = 10
TRAIN_VAL_SAME_PEOPLE = True
N_PEOPLE = 1

HOMOLOGY_DIMENSIONS = [0, 1, 2]
