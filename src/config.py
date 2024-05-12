from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Random seed

SEED = 42

# Paths and filenames

DATA_PATH = '../data'
EEG_PATH = f'{DATA_PATH}/osfstorage-archive/EEG Data'
IMAGE_PATH = '../images'
EMBEDDING_STATISTICS = 'embedding_stats.pkl'
EXPERIMENTS_RESULTS_FILE = 'results.pkl'

SORTED_IDS = ['067', '061', '007', '039', '057',
           '064', '013', '042', '062', '048',
           '081', '053', '044', '038', '002',
           '056', '051', '069', '055', '023',
           '016', '040', '059', '018', '017'] # IDs sorted by minimum counts of ASMR True/False intervals

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

DIMENSION = 6
TIME_DELAY = 15
STRIDE = 10

# Training settings

HOMOLOGY_DIMENSIONS = [0, 1, 2]

ALL_TARGETS = ['label', 'ASMR']
TARGET_ID = 1
TARGET = ALL_TARGETS[TARGET_ID]

CHANNELS = ALL_EEG_CHANNELS

N_INTERVALS_PER_PERSON_PER_CLASS = 10
TRAIN_VAL_SAME_PEOPLE = True
N_PEOPLE = 1

ALL_CLASSIFIERS = {
    'LogisticRegression': LogisticRegression(random_state=SEED),
    'SVC': SVC(probability=True),
    'KNeighbors': KNeighborsClassifier(),
    'RandomForest': RandomForestClassifier(random_state=SEED),
    'GradientBoosting': GradientBoostingClassifier(random_state=SEED),
    'AdaBoost': AdaBoostClassifier(algorithm='SAMME', random_state=SEED),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=SEED)
}
CLASSIFIER_NAME = 'LogisticRegression'
CLASSIFIER = ALL_CLASSIFIERS[CLASSIFIER_NAME]

# Experiment goal

ALL_EXPERIMENT_GOALS = [
    'Example',
    'Try different strides',
    'Fix some pairs of (train, validation) sets',
    'Compare classifiers',
    'Determine the most informative channels',
]
EXPERIMENT_GOAL_ID = 0
EXPERIMENT_GOAL = ALL_EXPERIMENT_GOALS[EXPERIMENT_GOAL_ID]
