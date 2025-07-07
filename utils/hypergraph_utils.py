import numpy as np
import torch


def Eu_dis(x):
    x = torch.tensor(x, dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    row_norms_squared = torch.sum(x * x, dim=1).unsqueeze(1)
    dist_mat = row_norms_squared + row_norms_squared.t() - 2 * torch.mm(x, x.t())
    dist_mat = torch.clamp(dist_mat, min=0)
    dist_mat = torch.sqrt(dist_mat)
    dist_mat = torch.maximum(dist_mat, dist_mat.t())
    return dist_mat


def construct_H_with_KNN_from_distance(dis_mat, k_neig, m_prob=1.0, is_probH=True):
    n_obj = dis_mat.shape[0]
    H = np.zeros((n_obj, n_obj))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx].cpu()
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx
        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs, is_probH=True, m_prob=1):
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])
    if isinstance(K_neigs, int):
        K_neigs = [K_neigs]
    dis_mat = Eu_dis(X)
    H = []
    for k in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k, m_prob, is_probH)
        if len(H) == 0:
            H = H_tmp
        else:
            H = np.hstack((H, H_tmp))
    return H


def generate_G_from_H(H):
    H = torch.tensor(H, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')
    n_edge = H.shape[1]
    W = torch.ones(n_edge, dtype=torch.float).to(H.device)
    DV = torch.sum(H * W, dim=1)
    DE = torch.sum(H, dim=0)
    invDE = torch.diag(torch.pow(DE, -1))
    DV2 = torch.diag(torch.pow(DV, -0.5))
    W = torch.diag(W)
    HT = H.T
    G = torch.mm(torch.mm(torch.mm(torch.mm(DV2, H), W), invDE), torch.mm(HT, DV2))
    return G.cpu().numpy()
