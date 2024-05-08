import mne
import glob
import os
import warnings
import pandas as pd
import numpy as np

from config import DATA_PATH


def read_raw_data():
    file_paths = glob.glob(f'{DATA_PATH}/*.set')
    file_names = [os.path.basename(file).replace('P', 'P') for file in file_paths]
    raw_data = []
    for path, filename in zip(file_paths, file_names):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            raw = mne.read_epochs_eeglab(path, verbose=False)
        idx = filename[1:4]
        raw_data.append((idx, raw))
    return raw_data

def process_raw_data(raw_data):
    all_ids = []
    all_segments = []
    all_labels = []

    for idx, raw in raw_data:
        ids = [idx] * len(raw)

        segments = raw.get_data(copy=True)[:, :-3] # Exclude non-EEG channels
        segments = [segments[i] for i in range(segments.shape[0])]

        inverse_event_id = {v: k[k.find('B'):k.find(')')+1] for k, v in raw.event_id.items()}
        inverse_event_id_func = np.vectorize(inverse_event_id.get)
        labels = raw.events[:, -1]
        labels = inverse_event_id_func(labels)

        all_ids.extend(ids)
        all_segments.extend(segments)
        all_labels.extend(labels)
        
    return pd.DataFrame({'id': all_ids, 'segment': all_segments, 'label': all_labels})
