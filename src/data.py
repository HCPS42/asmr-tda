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
    df['ASMR'] = np.char.find(df['label'].values.astype(str), 'ASMR') >= 0
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
    df.to_pickle(filename)
    return df

def normalize_diagrams(diagrams):
    diagrams = diagrams.copy()
    global_max = 0
    for diagram in diagrams:
        max_non_inf = np.max(diagram[np.isfinite(diagram)])
        diagram[np.isinf(diagram)] = max_non_inf
        global_max = max(global_max, max_non_inf)
    normalized = []
    for diagram in diagrams:
        diagram /= global_max
        normalized.append(diagram)
    return normalized

def process_point_cloud(point_cloud):
    h1_diagrams = []
    for i in range(point_cloud.shape[0]):
        channel = point_cloud[i]
        rips = Rips(maxdim=1, verbose=False)
        diagrams = rips.fit_transform(channel)
        diagrams = normalize_diagrams(diagrams)
        h1_diagrams.append(diagrams[1])    
    pimgr = PersistenceImager(pixel_size=0.05)
    images = pimgr.transform(h1_diagrams)
    images = np.array(images)
    reshaped_images = images.reshape(8, 8, 20, 20)
    grid_image = np.block([[reshaped_images[i, j] for j in range(8)] for i in range(8)])
    return grid_image

def load_diagram_images():
    filename = f'{DATA_PATH}/diagram_images.pkl'
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    df = load_point_clouds()
    df['diagram_image'] = df['point_cloud'].apply(lambda x: process_point_cloud(x))
    df = df[['diagram_image', 'ASMR']]
    df.to_pickle(filename)
    return df

class CustomDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = self.df.iloc[idx]['diagram_image']
        image = (image / image.max() * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = self.transform(image)
        label = self.df.iloc[idx]['ASMR']
        label = torch.tensor(label, dtype=torch.float32)
        return image, label
