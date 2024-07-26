import random
import torch
import os


def episodic_sampling(c_way, k_shot, base_classes, data):
    ways = random.sample(base_classes.tolist(), c_way)
    support_set = []
    query_set = []
    for way in ways:
        indicies = data.meta_data[data.meta_data['target'] == way].index.tolist()
        indicies = random.sample(indicies, k_shot + 1)
        if k_shot == 1:
            support = data[indicies[0]][0]
        else:
            support = torch.stack([data[i][0] for i in indicies[:-1]])
        query = data[indicies[-1]][0]
        support_set.append(support)
        query_set.append(query)
    return torch.stack(support_set), torch.stack(query_set)


def manage_checkpoints(checkpoint_dir, max_checkpoints):
    """
    Keep only the latest 'max_checkpoints' files in the directory.
    """
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
                         key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))

    if len(checkpoints) > max_checkpoints:
        for chkpt in checkpoints[:-max_checkpoints]:
            os.remove(os.path.join(checkpoint_dir, chkpt))


def Acc(logits, ground_truth):
    _, predicted_labels = torch.max(logits, 1)
    accuracy = (predicted_labels == ground_truth).float().mean()
    return accuracy
