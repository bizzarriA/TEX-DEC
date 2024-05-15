import torch
from torch.utils.data import DataLoader, random_split
from .benchmark_data import ADCIFAR10, ADMNIST, CICIDS2017

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

### Benchmark

DS_CHOICES = {  # list of implemented datasets (most can also be used as OE)
    'cifar10': {
        'class': ADCIFAR10, 'default_size': 32, 'no_classes': 10,
        'str_labels': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    },
    'mnist': {
        'class': ADMNIST, 'default_size': 28, 'no_classes': 10,
        'str_labels': [
          "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
        ]
    },
    'cicids2017': {
        'class': CICIDS2017, 'default_size': 1500, 'no_classes': 15,
    },

}


def load_dataset(name, batch_size, root='../dataset/data', num_worker=4, target_classes=None, transform=None):
    assert name in DS_CHOICES, f'{name} is not in {DS_CHOICES}'
    ds_train = DS_CHOICES[name]['class'](root, train=True, transform=transform,
                                         target_classes=target_classes)
    ds_test = DS_CHOICES[name]['class'](root, train=False, transform=transform,
                                        target_classes=target_classes)
    n_train = int(len(ds_train) * 0.9)
    n_val = len(ds_train) - n_train
    # ds_test = ConcatDataset([ds_test, ds_train])
    ds_train, ds_val = random_split(ds_train, [n_train, n_val])
    ds_train = DataLoader(ds_train,
                          batch_size,
                          num_workers=num_worker,
                          # drop_last=True
                          )
    ds_val = DataLoader(ds_val,
                        batch_size,
                        num_workers=num_worker,
                        # drop_last=True
                        )
    ds_test = DataLoader(ds_test,
                         batch_size,
                         num_workers=num_worker,
                         # drop_last=True
                         )
    return ds_train, ds_val, ds_test


def preprocess_data(df, remove, feature_name, target_name):
    # Function to preprocess the data, including label conversion, splitting, and combining datasets
    df_c = df.loc[df['real_label'].isin(remove)]
    df_train = df.loc[~df['real_label'].isin(remove)]
    print(np.unique(df_c.real_label, return_counts=True), np.unique(df_train.real_label, return_counts=True))
    print(np.unique(df_c.OOD, return_counts=True), np.unique(df_train.OOD, return_counts=True))
    # Convert labels to binary in df_train and df_c
    # Convert labels to binary in df_train and df_c using loc

    # Split data into train and test and add zero days to the test set
    df_train, df_test = train_test_split(df_train, test_size=0.8, random_state=42)
    X_train = df_train[feature_name]
    y_train = df_train[target_name]

    df_balance = df_test.sample(n=len(df_c), replace=True)
    X_test = df_test[feature_name]
    y_test = df_test[target_name]

    X_c = pd.concat([df_c[feature_name], df_balance[feature_name]])
    y_c = pd.concat([df_c[target_name], df_balance[target_name]])

    return X_train, y_train, X_c, y_c, X_test, y_test

