import mne
import glob
import os
import warnings

from config import DATA_PATH


def get_raw_data():
    file_paths = glob.glob(f'{DATA_PATH}/*.set')
    file_names = [os.path.basename(file).replace('P', 'P') for file in file_paths]
    raw_data = []
    for path, name in zip(file_paths, file_names):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            raw = mne.io.read_epochs_eeglab(path, verbose=False)
        raw_data.append((name, raw))
    return raw_data
