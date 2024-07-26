import copy
from proto_net import load_protonet_conv
import torch
import numpy as np
import pickle
from tqdm import tqdm
import scipy

# Load pre-sampled test task set with [sample_test_set.py]
with open('../test_set_ESC50_3000.pkl', 'rb') as f:
    test_tasks = pickle.load(f)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training loop
proto = load_protonet_conv([1, 128, 431], 64, 64).to(device)

checkpoint_path = '' # Specify your checkpoint path here
checkpoint = torch.load(checkpoint_path)
proto.load_state_dict(checkpoint['model_state_dict'])

episodes = len(test_tasks)

ACC = []
POST = []

for i in tqdm(range(episodes), leave=True):
    s, q = test_tasks[i]
    s = s.to(device)
    q = q.to(device)
    loss, pre, acc = proto.loss(s, q)
    ACC.append(acc.item())

    print(mean_confidence_interval(ACC))

