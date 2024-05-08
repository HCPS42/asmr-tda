import mne
import glob
import os
import warnings
import pandas as pd
import numpy as np
from gtda.time_series import TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from sklearn.model_selection import train_test_split

from config import SEED
from config import EEG_PATH
from config import N_EEG_CHANNELS
from config import TIME_DELAY, DIMENSION, STRIDE
from config import N_SAMPLES, TARGET, HOMOLOGY_DIMENSIONS, TEST_SIZE


def read_raw_data(n_people=None):
    file_paths = glob.glob(f'{EEG_PATH}/*.set')
    filenames = [os.path.basename(file).replace('P', 'P') for file in file_paths]
    if n_people is None:
        n_people = len(filenames)
    filenames = filenames[:n_people]
    raw_data = []
    for path, filename in zip(file_paths, filenames):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            raw = mne.read_epochs_eeglab(path, verbose=False)
        idx = filename[1:4]
        raw_data.append((idx, raw))
    return raw_data

def get_processed_data():    
    raw_data = read_raw_data()

    all_ids = []
    all_intervals = []
    all_labels = []

    for idx, raw in raw_data:
        ids = [idx] * len(raw)

        intervals = raw.get_data(copy=True)[:, :-3] # Exclude non-EEG channels
        intervals = [intervals[i] for i in range(intervals.shape[0])]

        inverse_event_id = {v: k[k.find('B'):k.find(')')+1] for k, v in raw.event_id.items()}
        inverse_event_id_func = np.vectorize(inverse_event_id.get)
        labels = raw.events[:, -1]
        labels = inverse_event_id_func(labels)

        all_ids.extend(ids)
        all_intervals.extend(intervals)
        all_labels.extend(labels)
        
    df = pd.DataFrame({'id': all_ids, 'interval': all_intervals, 'label': all_labels})

    TE = TakensEmbedding(time_delay=TIME_DELAY, dimension=DIMENSION, stride=STRIDE)
    TE.fit([])
    df['point_cloud'] = df.apply(lambda row: TE.transform(row['interval']), axis=1)
    df = df[['id', 'interval', 'point_cloud', 'label']]
    return df

def get_training_data():
    df = get_processed_data()

    df = df.drop(columns=['interval'])
    df = df.sample(N_SAMPLES)
    df = df.explode('point_cloud').reset_index(drop=True)
    df['channel'] = df.index % N_EEG_CHANNELS
    df['ASMR'] = np.char.find(df['label'].values.astype(str), 'ASMR') >= 0

    point_clouds = np.stack(df['point_cloud'].values)

    VR = VietorisRipsPersistence(homology_dimensions=HOMOLOGY_DIMENSIONS)
    diagrams = VR.fit_transform(point_clouds)

    # TODO: add features https://giotto-ai.github.io/gtda-docs/latest/modules/generated/diagrams/features/gtda.diagrams.Amplitude.html
    
    PE = PersistenceEntropy()
    features = PE.fit_transform(diagrams)
    features_df = pd.DataFrame(features, columns=[f'PE_{i}' for i in range(features.shape[1])])
    df = pd.concat([df, features_df], axis=1)
    
    df = df.drop(columns=['point_cloud'])
    return df

def split_training_data(df):
    X = df.drop(columns=['label', 'ASMR'])
    y = df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    return X_train, X_val, y_train, y_val
