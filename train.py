import csv
import math
import time
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
from tqdm import tqdm
import numpy as np
from models import DHGCN
from config import get_config
from datasets.visual_data import load_feature_construct_H
import torch.nn as nn


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
       refer to: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
    """

    def __init__(self, smoothing=0.1):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def save_results_to_csv(all_results, csv_file='results_DHGCN_DTU_1s.csv'):
    with open(csv_file, mode='w', newline='') as file:
        writer_csv = csv.DictWriter(file, fieldnames=all_results[0].keys())
        writer_csv.writeheader()
        for result in all_results:
            writer_csv.writerow(result)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class EEGDataset(Dataset):
    def __init__(self, data, labels, G, G_sp):
        self.data = data
        self.labels = labels
        self.G = G
        self.G_sp = G_sp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        G = self.G[idx]
        sp = self.G_sp[idx]
        return sample, G, sp, label


def train_model(train_loader, valid_loader, model, criterion, optimizer, scheduler, num_epochs, print_freq,
                patience=50):
    since = time.time()
    state_dict_updates = 0
    best_acc = 0
    loss_min = float('inf')
    acc_epo = 0
    loss_epo = 0
    model_wts_best_val_acc = copy.deepcopy(model.state_dict())
    model_wts_lowest_val_loss = copy.deepcopy(model.state_dict())

    epochs_no_improve = 0

    for epoch in tqdm(range(num_epochs)):
        # train
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for i_batch, batch_data in enumerate(train_loader):
            data, G, G_sp, train_label = batch_data
            train_label = train_label.squeeze(-1)
            data, G, G_sp, train_label = data.to(device), G.to(device), G_sp.to(device), train_label.to(
                device)
            batch_size = train_label.size(0)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(data, G, G_sp)
                outputs = outputs.to(device)
                loss = criterion(outputs, train_label)
                predicted = outputs.data.max(1)[1]

                loss.backward()
                optimizer.step()

                running_loss += loss * batch_size
                running_corrects += predicted.eq(train_label).cpu().sum()

        epoch_loss_train = running_loss / len(train_loader.dataset)
        epoch_acc_train = running_corrects.double() / len(train_loader.dataset)
        scheduler.step()

        if epoch % print_freq == 0:
            print(f'Epoch {epoch}: Train Loss: {epoch_loss_train:.4f} Acc: {epoch_acc_train:.4f}')

        # evaluate
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for i_batch, batch_data in enumerate(valid_loader):
            data, G, G_sp, valid_label = batch_data
            valid_label = valid_label.squeeze(-1)
            data, G, G_sp, valid_label = data.to(device), G.to(device), G_sp.to(device), valid_label.to(
                device)
            batch_size = valid_label.size(0)

            with torch.no_grad():
                outputs = model(data, G, G_sp)
                outputs = outputs.to(device)
                loss = criterion(outputs, valid_label)
                predicted = outputs.data.max(1)[1]

            running_loss += loss * batch_size
            running_corrects += predicted.eq(valid_label).cpu().sum()

        epoch_loss_val = running_loss / len(valid_loader.dataset)
        epoch_acc_val = running_corrects.double() / len(valid_loader.dataset)

        if epoch % print_freq == 0:
            print(f'Epoch {epoch}: Val Loss: {epoch_loss_val:.4f} Acc: {epoch_acc_val:.4f}')
            print(f'Best val Acc: {best_acc:.4f}, Min val loss: {loss_min:.4f}')
            print('-' * 20)

        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            acc_epo = epoch
            model_wts_best_val_acc = copy.deepcopy(model.state_dict())
            state_dict_updates += 1

        if epoch_loss_val < loss_min:
            loss_min = epoch_loss_val
            model_wts_lowest_val_loss = copy.deepcopy(model.state_dict())
            loss_epo = epoch
            state_dict_updates += 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} after {patience} epochs with no improvement.")
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'\nState dict updates {state_dict_updates}')
    print(f'Best val Acc: {best_acc:.4f}')

    return (model_wts_best_val_acc, acc_epo), (model_wts_lowest_val_loss, loss_epo)


def test(model, best_model_wts, test_loader, test_time=1):
    best_model_wts, epo = best_model_wts
    model = model.to(device)
    model.load_state_dict(best_model_wts)
    model.eval()

    total_corrects = 0.0
    total_samples = 0
    num_batches = 0

    for i_batch, batch_data in enumerate(test_loader):
        data, G, G_sp, test_label = batch_data
        G = G.to(device)
        G_sp = G_sp.to(device)
        test_label = test_label.squeeze(-1)
        data, test_label = data.to(device), test_label.to(device)
        batch_size = test_label.size(0)
        total_samples += batch_size

        # Voting mechanism if test_time > 1
        outputs = torch.zeros(batch_size, 2).to(device)
        for _ in range(test_time):
            with torch.no_grad():
                output = model(data, G, G_sp)
                outputs += output

        num_batches += 1
        outputs /= test_time
        predicted = outputs.data.max(1)[1]
        total_corrects += predicted.eq(test_label).cpu().sum()

    test_acc = total_corrects / total_samples
    print('*' * 20)
    print(f'Test Accuracy: {test_acc:.4f} @Epoch-{epo}')
    print('*' * 20)

    return test_acc, epo


def train_test_model_for_subject(cfg, subject_idx):

    (train_data, valid_data, test_data, train_label, valid_label, test_label,
     G_train, G_valid, G_test, G_train_sp, G_valid_sp, G_test_sp) = (
        load_feature_construct_H(K_neigs=cfg['K_neigs'],
                                 K_neigs_sp=cfg['K_neigs_sp'],
                                 test_idx=subject_idx + 1,
                                 is_probH=True, m_prob=1))

    time_point = math.ceil(cfg['fs'] * cfg['length'])
    model_ft = DHGCN(in_ch=cfg['eeg_channel'],
                     n_class=2,
                     time_point=time_point,
                     dropout=cfg['drop_out'])
    model_ft = model_ft.to(device)
    print(model_ft)
    print(f"The model has {count_parameters(model_ft):,} trainable parameters.")
    optimizer = optim.AdamW(model_ft.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=cfg['lr'] / 10)
    criterion = LabelSmoothing(smoothing=0.1)

    train_loader = DataLoader(dataset=EEGDataset(train_data, train_label, G_train, G_train_sp), batch_size=cfg['batch'],
                              drop_last=True)
    valid_loader = DataLoader(dataset=EEGDataset(valid_data, valid_label, G_valid, G_valid_sp), batch_size=cfg['batch'],
                              drop_last=True)
    test_loader = DataLoader(dataset=EEGDataset(test_data, test_label, G_test, G_test_sp), batch_size=cfg['batch'],
                             drop_last=True)

    model_wts_best_val_acc, model_wts_lowest_val_loss = train_model(
        train_loader, valid_loader, model_ft, criterion, optimizer,
        scheduler, cfg['max_epoch'], print_freq=cfg['print_freq'])

    # Testing with the lowest validation loss
    print('**** Model of lowest val loss ****')
    test_acc_lvl, epo_lvl = test(model_ft, model_wts_lowest_val_loss, test_loader, cfg['test_time'])
    print('**** Model of best val acc ****')
    test_acc_bva, epo_bva = test(model_ft, model_wts_best_val_acc, test_loader, cfg['test_time'])

    return {
        'model_index': subject_idx + 1,
        'test_acc_lowest_val_loss': test_acc_lvl,
        'epoch_lowest_val_loss': epo_lvl,
        'test_acc_best_val_acc': test_acc_bva,
        'epoch_best_val_acc': epo_bva
    }, test_acc_lvl.item()


if __name__ == '__main__':
    set_seed(3407)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = get_config('config/config.yaml')
    all_results = []
    all_acc = []
    for subject_idx in range(cfg['subject_number']):
        result, acc = train_test_model_for_subject(cfg, subject_idx)

        all_results.append(result)
        all_acc.append(acc)

    save_results_to_csv(all_results)

    avg_acc = np.mean(all_acc)
    print(f'Average accuracy over all subjects: {avg_acc:.4f}')
