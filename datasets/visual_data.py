import numpy as np
import torch
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from config import get_config
from utils import hypergraph_utils as hgut
from utils.data_utils import sliding_window, read_prepared_data
from tqdm import tqdm

cfg = get_config('./config/config.yaml')


def cat_egg_data(test_idx=1):

    test_person_idx = test_idx
    data, label = read_prepared_data("S" + str(test_person_idx))
    data = np.vstack(data)
    data = data.reshape([cfg['trail_number'], -1, cfg['eeg_channel']])
    label = np.vstack(label)

    train_eeg, test_eeg, train_label, test_label = sliding_window(data, label, cfg['eeg_channel'])
    train_eeg = train_eeg.transpose(0, 2, 1)
    test_eeg = test_eeg.transpose(0, 2, 1)
    train_label = np.array(train_label)
    train_label = np.squeeze(train_label)
    test_label = np.array(test_label)
    test_label = np.squeeze(test_label)

    csp = CSP(n_components=cfg['eeg_channel'], reg=None, log=None, cov_est='concat', transform_into='csp_space',
              norm_trace=True)
    train_eeg = train_eeg.transpose(0, 2, 1)
    test_eeg = test_eeg.transpose(0, 2, 1)
    train_eeg = torch.tensor(train_eeg, dtype=torch.double)
    test_data = torch.tensor(test_eeg, dtype=torch.double)
    train_label = torch.tensor(train_label, dtype=torch.long)
    train_data, valid_data, train_label, valid_label = train_test_split(train_eeg, train_label,
                                                                                test_size=0.1, shuffle=True)
    train_data = train_data.transpose(2, 1)
    valid_data = valid_data.transpose(2, 1)
    test_data = test_data.transpose(2, 1)

    train_data = train_data.numpy()
    valid_data = valid_data.numpy()
    test_data = test_data.numpy()
    train_data = csp.fit_transform(train_data, train_label)
    test_data = csp.transform(test_data)
    valid_data = csp.transform(valid_data)

    train_data = torch.tensor(train_data, dtype=torch.float)
    valid_data = torch.tensor(valid_data, dtype=torch.float)
    test_data = torch.tensor(test_data, dtype=torch.float)
    train_label = torch.tensor(train_label, dtype=torch.long)
    valid_label = torch.tensor(valid_label, dtype=torch.long)
    test_label = torch.tensor(test_label, dtype=torch.long)
    train_data = train_data.transpose(2, 1)
    valid_data = valid_data.transpose(2, 1)
    test_data = test_data.transpose(2, 1)
    return train_data, valid_data, test_data, train_label, valid_label, test_label


def cal_graph(data, K_neigs, is_probH=True, m_prob=1):
    G = []
    for i in tqdm(range(data.shape[0])):
        tmp = hgut.construct_H_with_KNN(data[i], K_neigs=K_neigs, is_probH=is_probH, m_prob=m_prob)
        g = hgut.generate_G_from_H(tmp)
        g = torch.tensor(g)
        G.append(g)
    return G


def load_feature_construct_H(K_neigs=[2], K_neigs_sp=[2], test_idx=14, is_probH=True, m_prob=1):
    train_data, valid_data, test_data, train_label, valid_label, test_label = cat_egg_data(test_idx)

    # construct temporal hypergraph G
    G_train = cal_graph(train_data, K_neigs, is_probH=is_probH, m_prob=m_prob)
    G_valid = cal_graph(valid_data, K_neigs, is_probH=is_probH, m_prob=m_prob)
    G_test = cal_graph(test_data, K_neigs, is_probH=is_probH, m_prob=m_prob)

    train_data = train_data.permute(0, 2, 1)
    valid_data = valid_data.permute(0, 2, 1)
    test_data = test_data.permute(0, 2, 1)

    # construct spatial hypergraph G_sp
    G_train_sp = cal_graph(train_data, K_neigs_sp, is_probH=is_probH, m_prob=m_prob)
    G_valid_sp = cal_graph(valid_data, K_neigs_sp, is_probH=is_probH, m_prob=m_prob)
    G_test_sp = cal_graph(test_data, K_neigs_sp, is_probH=is_probH, m_prob=m_prob)

    return train_data, valid_data, test_data, train_label, valid_label, test_label, \
        G_train, G_valid, G_test, G_train_sp, G_valid_sp, G_test_sp
