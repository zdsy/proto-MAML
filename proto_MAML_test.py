from proto_net import load_protonet_conv
import torch
import numpy as np
import learn2learn as l2l
from tqdm import tqdm
import scipy.stats
import pickle


# Load pre-sampled test task set with [sample_test_set.py]
with open('../test_set_ESC50.pkl', 'rb') as f:
    test_tasks = pickle.load(f)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_episodes = 500
steps = 8

proto = load_protonet_conv([1, 128, 431], 64, 64)

# maml = l2l.algorithms.MAML(proto, lr=0.2, first_order=True, allow_nograd=True).to(device)

# Meta-Curvature
maml = l2l.algorithms.GBML(
    module=proto,
    transform=l2l.optim.MetaCurvatureTransform,
    lr=0.2,
    adapt_transform=False,
    first_order=True  # has both 1st and 2nd order versions
).to(device)

checkpoint_path = ''  # Specify your checkpoint path here
checkpoint = torch.load(checkpoint_path)
maml.load_state_dict(checkpoint['model_state_dict'])

PRE_VAL = []
TRAIN_ACC = []
POST_VAL = []

for i in tqdm(range(test_episodes), leave=True):
    task_model = maml.clone()

    with torch.no_grad():
        s_v, q_v = test_tasks[i]
        s_v = s_v.to(device)
        q_v = q_v.to(device)
        pre_loss, pre_pred, pre_acc = task_model.module.loss(s_v, q_v)
        PRE_VAL.append(pre_acc.item())
        # print(pre_pred)

    for step in range(steps):
        for j in range(1, s_v.size(1)):
            s_f = torch.stack([torch.cat((s_v[i, :j], s_v[i, j + 1:])) for i in range(s_v.size(0))])
            q_f = torch.stack([s_v[i, j] for i in range(s_v.size(0))])
            s_f = s_f.to(device)
            q_f = q_f.to(device)
            loss, pre, acc = task_model.module.loss(s_f, q_f)
            task_model.adapt(loss)
            TRAIN_ACC.append(acc.item())

    with torch.no_grad():
        post_loss, post_pred, post_acc = task_model.module.loss(s_v, q_v)
        # print(post_pred)
        POST_VAL.append(post_acc.item())

    print(
        f"Episode [{i + 1}/{test_episodes}], pre_val_acc: {np.mean(PRE_VAL):.4f}, train_acc: {np.mean(TRAIN_ACC):.4f}, "
        f"post_val_acc: {np.mean(POST_VAL):.4f}")

pre = mean_confidence_interval(PRE_VAL)
post = mean_confidence_interval(POST_VAL)
print(pre, post)
