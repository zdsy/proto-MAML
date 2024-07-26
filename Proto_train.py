from data_loader import ESC50Dataset
import proto_net
from utils import episodic_sampling
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Load the metadata file
meta_data = pd.read_csv('../ESC-50/meta/esc50.csv') # Specify your dataset path here

np.random.seed(66)
classes = np.arange(50)
np.random.shuffle(classes)
train_classes = classes[:25]
val_classes = classes[25:40]
test_classes = classes[40:]

train_meta = meta_data[meta_data.target.isin(train_classes)]
val_meta = meta_data[meta_data.target.isin(val_classes)]
test_meta = meta_data[meta_data.target.isin(test_classes)]

train_dataset = ESC50Dataset('../ESC-50/audio', train_meta) # Also here
val_dataset = ESC50Dataset('../ESC-50/audio', val_meta)
test_dataset = ESC50Dataset('../ESC-50/audio', test_meta)

train_dataset.meta_data = train_dataset.meta_data.reset_index(drop=True)
val_dataset.meta_data = val_dataset.meta_data.reset_index(drop=True)
test_dataset.meta_data = test_dataset.meta_data.reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

proto = proto_net.load_protonet_conv([1, 128, 431], 64, 64).to(device)

optimizer = torch.optim.Adam(proto.parameters(), lr=1e-3)

epochs = 5000
episodes = 8
val_eps = 500
min_improvement = 1e-3
best_val_acc = 0
best_epoch = 0
patience = 20
patience_counter = 0
checkpoint_dir = '' # Specify your checkpoint path
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(epochs):

    ACC = []
    VAL_ACC = []

    for episode in tqdm(range(episodes), desc=f'Epoch {epoch+1}/{epochs}', leave=True):
        s, q = episodic_sampling(5, 5, train_classes, train_dataset)
        s = s.to(device)
        q = q.to(device)
        optimizer.zero_grad()
        loss, pre, acc = proto.loss(s, q)
        loss.backward()
        optimizer.step()
        ACC.append(acc.item())

    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': proto.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Add any other information you need

    }, checkpoint_path)

    # print(f'Model checkpoint saved to {checkpoint_path}')

    if (epoch + 1) % 50 == 0:
        for val_ep in tqdm(range(val_eps), desc=f'Test_epoch {epoch + 1}/{epochs}', leave=True):
            s_v, q_v = episodic_sampling(5, 5, val_classes, val_dataset)
            s_v = s_v.to(device)
            q_v = q_v.to(device)
            with torch.no_grad():
                val_loss, val_pre, val_acc = proto.loss(s_v, q_v)
                VAL_ACC.append(val_acc.item())

        if np.mean(VAL_ACC) - best_val_acc > min_improvement:
                best_val_acc = np.mean(VAL_ACC)
                best_epoch = epoch + 1
                patience_counter = 0  # Reset counter
        else:
                patience_counter += 1

        if patience_counter > patience:
                break

    print(f"Epoch [{epoch + 1}/{epochs}], train_acc: {np.mean(ACC):.4f}, "
          f"val_acc: {np.mean(VAL_ACC):.4f}, best_test_acc: {best_val_acc:.4f}, best_epoch: {best_epoch}")
