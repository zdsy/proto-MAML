import copy

from data_loader import ESC50Dataset
from utils import episodic_sampling
from proto_net import load_protonet_conv
import torch
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import scipy


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


# Load the metadata file
# meta_data = pd.read_csv('../ESC-50/meta/esc50.csv')
#
# np.random.seed(66)
# classes = np.arange(50)
# np.random.shuffle(classes)
# train_classes = classes[:25]
# val_classes = classes[25:40]
# test_classes = classes[40:]
#
# train_meta = meta_data[meta_data.target.isin(train_classes)]
# val_meta = meta_data[meta_data.target.isin(val_classes)]
# test_meta = meta_data[meta_data.target.isin(test_classes)]
#
# train_dataset = ESC50Dataset('../ESC-50/audio', train_meta)
# val_dataset = ESC50Dataset('../ESC-50/audio', val_meta)
# test_dataset = ESC50Dataset('../ESC-50/audio', test_meta)
#
# train_dataset.meta_data = train_dataset.meta_data.reset_index(drop=True)
# val_dataset.meta_data = val_dataset.meta_data.reset_index(drop=True)
# test_dataset.meta_data = test_dataset.meta_data.reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training loop
proto = load_protonet_conv([1, 128, 431], 64, 64).to(device)

# checkpoint_path = '../Proto_checkpoints/ESC-50/5-way-5-shot/model_epoch_38.pth'  # Update this path
# checkpoint_path = '../Proto_checkpoints/ESC-50/Robust/model_epoch_146.pth'
checkpoint_path = '../Proto_checkpoints/ESC-50/10-way-1-shot/model_epoch_750.pth'
checkpoint = torch.load(checkpoint_path)
proto.load_state_dict(checkpoint['model_state_dict'])

with open('../test_set_ESC50_3000.pkl', 'rb') as f:
    test_tasks = pickle.load(f)

episodes = len(test_tasks)

ACC = []
POST = []

for i in tqdm(range(episodes), leave=True):
    # s, q = episodic_sampling(5, 5, test_classes, test_dataset)
    s, q = test_tasks[i]
    s = s.to(device)
    q = q.to(device)
    loss, pre, acc = proto.loss(s, q)
    ACC.append(acc.item())

    task_model = copy.deepcopy(proto)
    opt = torch.optim.Adam(task_model.parameters(), lr=1e-2)

    print(mean_confidence_interval(ACC))

