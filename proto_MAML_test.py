from Proto.data_loader import ESC50Dataset
from proto_net import load_protonet_conv, Protonet
from utils import episodic_sampling, manage_checkpoints
import torch
import pandas as pd
import numpy as np
import os
import learn2learn as l2l
from tqdm import tqdm
import scipy.stats
import pickle

with open('../test_set_ESC50.pkl', 'rb') as f:
    test_tasks = pickle.load(f)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


# Load the metadata file
# meta_data = pd.read_csv('../ESC-50/meta/esc50.csv')

# np.random.seed(66)
# classes = np.arange(50)
# np.random.shuffle(classes)
# train_classes = classes[:25]
# val_classes = classes[25:40]
# train_classes = classes[:40]
# test_classes = classes[40:]

# train_meta = meta_data[meta_data.target.isin(train_classes)]
# val_meta = meta_data[meta_data.target.isin(val_classes)]
# test_meta = meta_data[meta_data.target.isin(test_classes)]

# train_dataset = ESC50Dataset('../ESC-50/audio', train_meta)
# val_dataset = ESC50Dataset('../ESC-50/audio', val_meta)
# test_dataset = ESC50Dataset('../ESC-50/audio', test_meta)

# train_dataset.meta_data = train_dataset.meta_data.reset_index(drop=True)
# val_dataset.meta_data = val_dataset.meta_data.reset_index(drop=True)
# test_dataset.meta_data = test_dataset.meta_data.reset_index(drop=True)

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_episodes = 500
steps = 8

proto = load_protonet_conv([1, 128, 431], 64, 64)

# maml = l2l.algorithms.MAML(proto, lr=0.2, first_order=True, allow_nograd=True).to(device)

maml = l2l.algorithms.GBML(
    module=proto,
    transform=l2l.optim.MetaCurvatureTransform,
    lr=0.2,
    adapt_transform=False,
    first_order=True  # has both 1st and 2nd order versions
).to(device)

checkpoint_path = '../MC_checkpoints/5-way-5-shot/EDFT/model_epoch_6000.pth'  # Update this path
checkpoint = torch.load(checkpoint_path)
maml.load_state_dict(checkpoint['model_state_dict'])

PRE_VAL = []
TRAIN_ACC = []
POST_VAL = []

for i in tqdm(range(test_episodes), leave=True):
    task_model = maml.clone()

    with torch.no_grad():
        # s_v, q_v = episodic_sampling(5, 5, test_classes, test_dataset)
        s_v, q_v = test_tasks[i]
        s_v = s_v.to(device)
        q_v = q_v.to(device)
        pre_loss, pre_pred, pre_acc = task_model.module.loss(s_v, q_v)
        PRE_VAL.append(pre_acc.item())
        # print(pre_pred)

        # for tr_batch in range(batches):
    for step in range(steps):
        for j in range(1, s_v.size(1)):
            # s_f = torch.stack([torch.cat((s_v[i, :j], s_v[i, j + 1:])) for i in range(s_v.size(0))])
            # q_f = torch.stack([s_v[i, j] for i in range(s_v.size(0))])
            s_f = torch.stack([s_v[i, :j] for i in range(s_v.size(0))])
            q_f = torch.stack([s_v[i, j] for i in range(s_v.size(0))])
            s_f = s_f.to(device)
            q_f = q_f.to(device)
            loss, pre, acc = task_model.module.loss(s_f, q_f)
            task_model.adapt(loss)
            TRAIN_ACC.append(acc.item())

            # peak_memory = torch.cuda.max_memory_allocated() / 1e9
            # total_params = sum(p.numel() for p in task_model.parameters()) / 1e6
            # print(s_v.shape)
            # print(f'Model size: {total_params} Mb')
            # print(f'Peak Memory Usage: {peak_memory:.2f} Gb')
            # breakpoint()

    with torch.no_grad():
        post_loss, post_pred, post_acc = task_model.module.loss(s_v, q_v)
        # print(post_pred)
        POST_VAL.append(post_acc.item())
        # del s_v, q_v, s_f, q_f

        # print(f"Epoch [{epoch + 1}/{epochs}], Episode [{episode + 1}/{episodes}], train_acc: {np.mean(train_acc):.4f}, "
        #       f"val_Loss: {np.mean(post_val_acc):.4f}")

    # print(f"Epoch [{epoch + 1}/{epochs}], Meta-Loss: {meta_loss.item():.4f}")
    print(
        f"Episode [{i + 1}/{test_episodes}], pre_val_acc: {np.mean(PRE_VAL):.4f}, train_acc: {np.mean(TRAIN_ACC):.4f}, "
        f"post_val_acc: {np.mean(POST_VAL):.4f}")

pre = mean_confidence_interval(PRE_VAL)
post = mean_confidence_interval(POST_VAL)
print(pre, post)
