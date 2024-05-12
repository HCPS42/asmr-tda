import pandas as pd
from gtda.time_series import SingleTakensEmbedding
from tqdm import tqdm
import os
import numpy as np

from data import read_raw_data
from data import extract_intervals
from config import SEED, EMBEDDING_STATISTICS, ALL_EEG_CHANNELS


def get_embedding_statistics():
    filename = EMBEDDING_STATISTICS
    if os.path.exists(filename):
        return pd.read_pickle(filename)

    raw_data = read_raw_data()
    df = extract_intervals(raw_data)

    df = df.groupby(['id', 'label']).sample(2, random_state=SEED).reset_index(drop=True)
    df = df.explode('interval').reset_index(drop=True)
    df['channel'] = np.tile(ALL_EEG_CHANNELS, df.shape[0] // len(ALL_EEG_CHANNELS))
    
    max_embedding_dimension = 10
    max_time_delay = 90
    stride = 1

    embedder = SingleTakensEmbedding(
        parameters_type="search",
        time_delay=max_time_delay,
        dimension=max_embedding_dimension,
        stride=stride,
    )

    results = []

    for i, row in tqdm(df.iterrows()):
        interval = row['interval']
        interval_embedded = embedder.fit_transform(interval)
        
        results.append({
            'id': row['id'],
            'channel': row['channel'],
            'label': row['label'],
            'dimension': embedder.dimension_, 
            'time_delay': embedder.time_delay_
        })

    results_df = pd.DataFrame(results)

    results_df.to_pickle(filename)

    return results_df
