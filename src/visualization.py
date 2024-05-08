import mne
import matplotlib.pyplot as plt
import warnings

from config import IMG_PATH


def plot_eeg(raw_data, show=True, verbose=True):
    idx, raw = raw_data
    img_path = f'{IMG_PATH}/{idx}'

    plt.close('all')

    if verbose:
        print('Data loaded successfully. Data info:')
        print(raw.info)

    fig_raw = raw.plot(show=False)
    fig_raw.savefig(f'{img_path}_raw_data_plot.png')
    if verbose:
        print(f"Raw data plot saved as '{img_path}_raw_data_plot.png'.")

    if show:
        plt.show()
    else:
        plt.close('all')

def plot_psd(raw_data, show=True, verbose=True):
    idx, raw = raw_data
    img_path = f'{IMG_PATH}/{idx}'

    plt.close('all')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        with mne.utils.use_log_level('ERROR'):
            fig_psd = raw.compute_psd(verbose=False).plot(show=False)

    fig_psd.savefig(f'{img_path}_psd_plot.png')
    if verbose:
        print(f"PSD plot saved as '{img_path}_psd_plot.png'.")

    if show:
        plt.show()
    else:
        plt.close('all')

def plot_topomap(raw_data, times, show=True, verbose=True):
    idx, raw = raw_data
    img_path = f'{IMG_PATH}/{idx}'

    plt.close('all')

    evoked = raw.average()
    fig_topo = evoked.plot_topomap(times, size=3, show=False)
    fig_topo.savefig(f'{img_path}_topomap.png')
    if verbose:
        print(f"Topomap saved as '{img_path}_topomap.png'.")

    if show:
        plt.show()
    else:
        plt.close('all')
