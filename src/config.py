import torch


# Random seed

SEED = 42

# Paths and filenames

DATA_PATH = '../data'
EEG_PATH = f'{DATA_PATH}/osfstorage-archive/EEG Data'
CHECKPOINTS_PATH = '../checkpoints'

SORTED_IDS = ['067', '061', '007', '039', '057',
           '064', '013', '042', '062', '048',
           '081', '053', '044', '038', '002',
           '056', '051', '069', '055', '023',
           '016', '040', '059', '018', '017'] # IDs sorted by minimum counts of ASMR True/False intervals

# Embedding settings

DIMENSION = 6
TIME_DELAY = 15
STRIDE = 10

# Training settings

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VAL_SIZE = 0.2
BATCH_SIZE = 32
NUM_EPOCHS = 20
