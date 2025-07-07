import numpy as np
from config import get_config
import pandas as pd
import math

cfg = get_config('./config/config.yaml')
window_length = math.ceil(cfg['fs'] * cfg['length'])


def read_prepared_data(name):
    data = []  # data
    target = []  # label
    for l in range(len(cfg['ConType'])):
        label = pd.read_csv(cfg['data_path'] + "/csv/" + name + cfg['ConType'][l] + ".csv", header=None)
        for k in range(cfg['trail_number']):
            filename = cfg['data_path'] + "/" + cfg['ConType'][l] + "/" + name + "Tra" + str(k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)
            eeg_data = data_pf.iloc[:, :]
            data.append(eeg_data)
            target.append(label.iloc[k, 0])
    return data, target


def sliding_window(eeg_datas, labels, out_channels):
    window_size = window_length
    stride = int(window_size * (1 - cfg['overlap']))

    train_eeg = []
    test_eeg = []
    train_label = []
    test_label = []

    for m in range(len(labels)):
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            window = eeg[i:i + window_size, :]
            windows.append(window)
            new_label.append(label)
        train_eeg.append(np.array(windows)[:int(len(windows) * 0.9)])
        test_eeg.append(np.array(windows)[int(len(windows) * 0.9):])
        train_label.append(np.array(new_label)[:int(len(windows) * 0.9)])
        test_label.append(np.array(new_label)[int(len(windows) * 0.9):])

    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels)
    test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)
    test_label = np.stack(test_label, axis=0).reshape(-1, 1)

    return train_eeg, test_eeg, train_label, test_label
