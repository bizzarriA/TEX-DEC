from torch.utils.data import Dataset
import pandas as pd
import torch

class CICIDS2017(Dataset):
    def __init__(self, root, train=True, transform=None, target_classes=None):
        if train:
            csv_path = f"{root}/payload_train_resampled.csv"
        else:
            csv_path = f"{root}/payload_test_resampled.csv"
        self.csv = self.filter_columns(pd.read_csv(csv_path))
        self.transform = transform
        self.name = csv_path.split('/')[-1]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        try:
            features = torch.tensor(row.drop(['label', 'unique_id']), dtype=torch.float32) / 255.
            unique_id = torch.tensor(row['unique_id'])
        except Exception:
            features = torch.tensor(row.drop('label'), dtype=torch.float32) / 255.
            unique_id = 0
        try:
            labels = torch.tensor(row['label'], dtype=torch.int)
        except TypeError:
            labels = torch.tensor(0 if row['label'] == 'normal' else 1)

        if self.transform:
            features = self.transform(features)

        return features, labels, unique_id

    def __len__(self):
        return len(self.csv)


    def filter_columns(self, df):
        selected_columns = [col for col in df.columns if
                            col.startswith("payload_byte_") or col == "label" or col == 'unique_id']
        df = df[selected_columns]

        return df

