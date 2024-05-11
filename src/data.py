import mne
import glob
import os
import warnings
import pandas as pd
import numpy as np
from gtda.time_series import TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude, PersistenceEntropy, NumberOfPoints
from sklearn.decomposition import PCA
import hashlib
import pickle

from config import (
    SEED,
    DATA_PATH, EEG_PATH, ALL_EEG_CHANNELS,
    TIME_DELAY, DIMENSION, STRIDE,
    CHANNELS, ALL_TARGETS, TARGET, N_INTERVALS_PER_PERSON_PER_CLASS, TRAIN_VAL_SAME_PEOPLE, N_PEOPLE,
    HOMOLOGY_DIMENSIONS
)

def read_raw_data(n_people=None):
    file_paths = glob.glob(f'{EEG_PATH}/*.set')
    filenames = [os.path.basename(file).replace('P', 'P') for file in file_paths]
    if n_people is None:
        if TRAIN_VAL_SAME_PEOPLE:
            n_people = N_PEOPLE
        else:
            n_people = 2 * N_PEOPLE
    if n_people == -1:
        n_people = len(filenames)
    else:
        np.random.seed(SEED)
        filenames = np.random.choice(filenames, size=n_people, replace=False)
    raw_data = []
    for path, filename in zip(file_paths, filenames):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            raw = mne.read_epochs_eeglab(path, verbose=False)
        idx = filename[1:4]
        raw_data.append((idx, raw))
    return raw_data

def extract_intervals(raw_data=None):
    if raw_data is None:
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

    return df

def compute_point_clouds(intervals=None):
    if intervals is None:
        df = extract_intervals()
    else:
        df = intervals.copy(deep=True)

    TE = TakensEmbedding(time_delay=TIME_DELAY, dimension=DIMENSION, stride=STRIDE)
    TE.fit([])
    df['point_cloud'] = df.apply(lambda row: TE.transform(row['interval']), axis=1)

    pca = PCA(n_components=3)

    def transform_point_cloud(point_clouds):
        projected_clouds = []
        for i in range(point_clouds.shape[0]):
            point_cloud = point_clouds[i]
            projected_cloud = pca.fit_transform(point_cloud)
            projected_clouds.append(projected_cloud)
        return np.stack(projected_clouds)
    
    df['point_cloud'] = df['point_cloud'].apply(transform_point_cloud)

    df = df.drop(columns=['interval'])

    return df

def prepare_data(df, 
                 step, 
                 vietoris_rips,
                 amplitude,
                 persistence_entropy,
                 number_of_points):
    
    df = df.explode('point_cloud').reset_index(drop=True)
    df['channel'] = np.tile(ALL_EEG_CHANNELS, df.shape[0] // len(ALL_EEG_CHANNELS))
    df = df[df['channel'].isin(CHANNELS)].reset_index(drop=True)

    point_clouds = np.stack(df['point_cloud'].values)
    df = df.drop(columns=['point_cloud'])

    if step == 'train':
        diagrams = vietoris_rips.fit_transform(point_clouds)
        amplitude_features = amplitude.fit_transform(diagrams)
        persistence_entropy_features = persistence_entropy.fit_transform(diagrams)
        betti_features = number_of_points.fit_transform(diagrams)
    elif step == 'val':
        diagrams = vietoris_rips.transform(point_clouds)
        amplitude_features = amplitude.transform(diagrams)
        persistence_entropy_features = persistence_entropy.transform(diagrams)
        betti_features = number_of_points.transform(diagrams)

    amplitude_features_df = pd.DataFrame(amplitude_features,
                                         columns=[f'amplitude_{i}' for i in range(amplitude_features.shape[1])])

    persistence_entropy_features_df = pd.DataFrame(persistence_entropy_features,
                                                   columns=[f'entropy_{i}' for i in range(persistence_entropy_features.shape[1])])

    betti_features_df = pd.DataFrame(betti_features,
                                     columns=[f'Betti_{i}' for i in range(betti_features.shape[1])])

    df = pd.concat([df,
                    amplitude_features_df,
                    persistence_entropy_features_df,
                    betti_features_df], axis=1)

    return df

def generate_filename(step):
    channels_str = str(CHANNELS).replace(' ', '')
    tvsp = 'same' if TRAIN_VAL_SAME_PEOPLE else 'diff'
    concat_str = f"{TIME_DELAY}_{DIMENSION}_{STRIDE}_{channels_str}_{TARGET}_{N_INTERVALS_PER_PERSON_PER_CLASS}_{tvsp}_{N_PEOPLE}"
    hash_object = hashlib.sha256(concat_str.encode())
    hash_integer = int(hash_object.hexdigest(), 16)
    filename = f'{DATA_PATH}/df_{step}_{hash_integer}.pkl'
    return filename

def get_training_data(df=None):
    train_file = generate_filename('train')
    val_file = generate_filename('val')

    if os.path.exists(train_file):
        df_train = pd.read_pickle(train_file)
        df_val = pd.read_pickle(val_file)
    else:
        if df is None:
            df = extract_intervals()
        else:
            df = df.copy(deep=True)

        df['ASMR'] = np.char.find(df['label'].values.astype(str), 'ASMR') >= 0

        np.random.seed(SEED)

        if TRAIN_VAL_SAME_PEOPLE:
            unique_people = df['id'].unique()
            selected_people = np.random.choice(unique_people, size=N_PEOPLE, replace=False)

            df_train = pd.DataFrame()
            df_val = pd.DataFrame()

            for person in selected_people:
                df_person = df[df['id'] == person]
                
                grouped = df_person.groupby(TARGET)
                df_sampled = pd.DataFrame()
                
                for _, group in grouped:
                    if len(group) < 2 * N_INTERVALS_PER_PERSON_PER_CLASS:
                        raise RuntimeError('Not enough intervals')
                    
                    sample = group.sample(2 * N_INTERVALS_PER_PERSON_PER_CLASS, replace=False)
                    
                    df_sampled = pd.concat([df_sampled, sample], ignore_index=True)
                    train, val = np.split(sample, 2)
                        
                    df_train = pd.concat([df_train, train], ignore_index=True)
                    df_val = pd.concat([df_val, val], ignore_index=True)

        else:
            unique_people = df['id'].unique()
            np.random.shuffle(unique_people)
            
            train_people = unique_people[:N_PEOPLE]
            val_people = unique_people[N_PEOPLE:N_PEOPLE * 2]
            
            df_train = pd.DataFrame()
            df_val = pd.DataFrame()
            
            for person in train_people:
                df_person = df[df['id'] == person]
                grouped = df_person.groupby(TARGET)
                
                for _, group in grouped:
                    if len(group) < N_INTERVALS_PER_PERSON_PER_CLASS:
                        raise RuntimeError(f'Not enough intervals')
                    
                    train_sample = group.sample(N_INTERVALS_PER_PERSON_PER_CLASS, replace=False)
                    df_train = pd.concat([df_train, train_sample], ignore_index=True)
            
            for person in val_people:
                df_person = df[df['id'] == person]
                grouped = df_person.groupby(TARGET)
                
                for _, group in grouped:
                    if len(group) < N_INTERVALS_PER_PERSON_PER_CLASS:
                        raise RuntimeError(f'Not enough intervals')
                    
                    val_sample = group.sample(N_INTERVALS_PER_PERSON_PER_CLASS, replace=False)
                    df_val = pd.concat([df_val, val_sample], ignore_index=True)

        df_train = compute_point_clouds(df_train)
        df_val = compute_point_clouds(df_val)

        vietoris_rips = VietorisRipsPersistence(homology_dimensions=HOMOLOGY_DIMENSIONS)
        amplitude = Amplitude()
        persistence_entropy = PersistenceEntropy()
        number_of_points = NumberOfPoints()

        df_train = prepare_data(df_train, 
                                'train',
                                vietoris_rips,
                                amplitude,
                                persistence_entropy,
                                number_of_points)

        df_val = prepare_data(df_val,
                              'val',
                              vietoris_rips,
                              amplitude,
                              persistence_entropy,
                              number_of_points)

        df_train.to_pickle(train_file)
        df_val.to_pickle(val_file)

    X_train = df_train.drop(columns=ALL_TARGETS)
    y_train = df_train[TARGET]

    X_val = df_val.drop(columns=ALL_TARGETS)
    y_val = df_val[TARGET]

    return X_train, X_val, y_train, y_val
