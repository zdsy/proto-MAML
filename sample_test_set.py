from data_loader import ESC50Dataset
from utils import episodic_sampling
import pandas as pd
import numpy as np
import pickle
import random

# Load the metadata file
meta_data = pd.read_csv('ESC-50/meta/esc50.csv') # Specify your dataset path here

np.random.seed(66)
classes = np.arange(50)
np.random.shuffle(classes)
train_classes = classes[:25]
val_classes = classes[25:40]
test_classes = classes[40:]

train_meta = meta_data[meta_data.target.isin(train_classes)]
val_meta = meta_data[meta_data.target.isin(val_classes)]
test_meta = meta_data[meta_data.target.isin(test_classes)]

train_dataset = ESC50Dataset('ESC-50/audio', train_meta) # Also here
val_dataset = ESC50Dataset('ESC-50/audio', val_meta)
test_dataset = ESC50Dataset('ESC-50/audio', test_meta)

train_dataset.meta_data = train_dataset.meta_data.reset_index(drop=True)
val_dataset.meta_data = val_dataset.meta_data.reset_index(drop=True)
test_dataset.meta_data = test_dataset.meta_data.reset_index(drop=True)

volume = 3000
task_set = []

for i in range(volume):
    s, q = episodic_sampling(5, 5, test_classes, test_dataset)
    task_set.append([s, q])

with open('test_set_ESC50.pkl', 'wb') as f:
    pickle.dump(task_set, f)
