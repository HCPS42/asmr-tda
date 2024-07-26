import mne
import glob
import os
import warnings
import pandas as pd
import numpy as np
from persim import PersistenceImager
from ripser import Rips
import torch
from torch.utils.data import Dataset
from PIL import Image

from config import (
    DATA_PATH, EEG_PATH, SORTED_IDS,
    TIME_DELAY, DIMENSION, STRIDE
)


def read_raw_data(n_people=None):
    file_paths = glob.glob(f'{EEG_PATH}/*.set')
    filenames = [os.path.basename(file) for file in file_paths]
    if n_people is not None:
        filenames = sorted(filenames, key=lambda filename: SORTED_IDS.index(filename[1:4]))
        filenames = filenames[:n_people]
        file_paths = [f'{EEG_PATH}/{filename}' for filename in filenames]
    raw_data = []
    for path, filename in zip(file_paths, filenames):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            raw = mne.read_epochs_eeglab(path, verbose=False)
        idx = filename[1:4]
        raw_data.append((idx, raw))
    return raw_data

def extract_intervals(raw_data):
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

def load_intervals():
    filename = f'{DATA_PATH}/intervals.pkl'
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    raw_data = read_raw_data()
    df = extract_intervals(raw_data)
    df.to_pickle(filename)
    return df

def takens_embedding(arr, time_delay=TIME_DELAY, dimension=DIMENSION, stride=STRIDE):
    embedded_rows = []
    for row in arr:
        length = len(row)
        num_embeddings = (length - (dimension - 1) * time_delay) // stride
        row_embed = np.lib.stride_tricks.as_strided(
            row,
            shape=(num_embeddings, dimension),
            strides=(row.strides[0] * stride, row.strides[0] * time_delay)
        )
        embedded_rows.append(row_embed)
    return np.array(embedded_rows)

def load_point_clouds():
    filename = f'{DATA_PATH}/point_clouds.pkl'
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    df = load_intervals()
    df['point_cloud'] = df['interval'].apply(lambda x: takens_embedding(x))
    df = df[['id', 'point_cloud', 'label']]
    df.to_pickle(filename)
    return df

def process_point_cloud(point_cloud):
    all_diagrams = []
    for i in range(point_cloud.shape[0]):
        channel = point_cloud[i]
        rips = Rips(maxdim=2, verbose=False)
        diagrams = rips.fit_transform(channel)
        all_diagrams.append(diagrams)    
    return all_diagrams

def load_diagrams():
    filename = f'{DATA_PATH}/diagrams.pkl'
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    df = load_point_clouds()
    df['diagram'] = df['point_cloud'].apply(lambda x: process_point_cloud(x))
    df = df[['id', 'diagram', 'label']]
    df.to_pickle(filename)
    return df

def process_diagram(diagram):
    diagram = diagram.copy()
    all_features = []
    for channel in diagram:
        homologies = channel[1]
        if homologies.size == 0:
            homologies = np.array([[0, 0]])
        differences = homologies[:, 1] - homologies[:, 0]
        sorted_indices = np.argsort(-differences)
        sorted_homologies = homologies[sorted_indices]
        if len(sorted_homologies) >= 30:
            features = sorted_homologies[:30]
        else:
            repeats = 30 // len(sorted_homologies) + 1
            bootstrapped = np.tile(sorted_homologies, (repeats, 1))
            features = bootstrapped[:30]
        all_features.append(features)
    return np.array(all_features)

def load_features():
    filename = f'{DATA_PATH}/features.pkl'
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    df = load_diagrams()
    df['features'] = df['diagram'].apply(lambda x: process_diagram(x))
    df = df[['id', 'features', 'label']]
    df.to_pickle(filename)
    return df

class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        features = self.df.iloc[idx]['features']
        label = self.df.iloc[idx]['ASMR']
        label = torch.tensor(label, dtype=torch.float32)
        return features, label
