import mne
import matplotlib.pyplot as plt
import warnings

from config import IMG_PATH


def visualize_eeg(raw_data, times=None, show_plots=True, verbose=True):
    file_name, raw = raw_data
    img_path = f'{IMG_PATH}/{file_name}'

    plt.close('all')

    if verbose:
        print('Data loaded successfully. Data info:')
        print(raw.info)

    fig_raw = raw.plot(show=False)
    fig_raw.savefig(f'{img_path}_raw_data_plot.png')
    if verbose:
        print(f"Raw data plot saved as '{img_path}_raw_data_plot.png'.")

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        with mne.utils.use_log_level('ERROR'):
            fig_psd = raw.compute_psd(verbose=False).plot(show=False)

    fig_psd.savefig(f'{img_path}_psd_plot.png')
    if verbose:
        print(f"PSD plot saved as '{img_path}_psd_plot.png'.")

    if times is not None:
        evoked = raw.average()
        fig_topo = evoked.plot_topomap(times, size=3, show=False)
        fig_topo.savefig(f'{img_path}_topomap.png')
        if verbose:
            print(f"Topomap saved as '{img_path}_topomap.png'.")

    if show_plots:
        plt.show()
    else:
        plt.close('all')
