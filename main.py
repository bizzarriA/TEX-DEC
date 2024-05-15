import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")
import argparse

import os
import torch
import pandas as pd
from torchvision import transforms

import wandb

from DEC import DEC
from utils.data import load_dataset, preprocess_data
from utils.for_train import grid_search_xgboost
from utils.for_eval import evalXG



def get_arg():
    # Function to parse command-line arguments using argparse
    arg = argparse.ArgumentParser()
    arg.add_argument('--name', required=True, type=str, help='Name of experiments')
    arg.add_argument('--version', type=str,default='v0', help="version of experiments") ## same experiments, different save_dir
    arg.add_argument('--bs', default=32, type=int, help='batch size')
    arg.add_argument('--pre_epoch', default=300, type=int, help='epochs for train Autoencoder')
    arg.add_argument('--epoch', type=int, default=200, help='epochs for train DEC')
    arg.add_argument('--input_dim', type=int, default=1500, help='dim input')
    arg.add_argument('-k', type=int, default=2, help='num of clusters')
    arg.add_argument('--features', type=int, default=12, help='num features in latent space')
    arg.add_argument('--worker', default=4, type=int, help='num of workers')
    arg.add_argument('--dir', default='TEX-DEC/', type=str, help="project dir")
    arg.add_argument('--wandb', type=bool, default=False, help="Use or not WANDB")
    arg.add_argument('--DEC', action='store_true')
    arg.add_argument('--XG', action='store_true')
    arg = arg.parse_args()
    return arg


def main():
    arg = get_arg()
    output_dir = f"{arg.dir}/results/{arg.name}_{arg.version}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    UNSW = False

    if not os.path.exists(arg.dir + 'results/'):
        os.makedirs(f"{arg.dir}/results/", exist_ok=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"{arg.name} test for pretrain: {arg.pre_epoch} epochs and train: {arg.epoch} epochs")
    if arg.name == 'mnist':
        normal_classes = [0, 1, 2, 3, 4, 6, 5]
        remove = [9, 8, 7]
    elif arg.name == 'cicids2017':
        normal_classes = [0, 1, 2, 3, 4, 6, 5]
        remove = [9, 8, 7]
    else:
        print("--name not valid")
        exit()
    if arg.DEC:
        transform = transforms.Compose([
            transforms.ToTensor(),
            torch.nn.Flatten(0)
        ])
        alpha, beta, omega = 1, 1, 1
        ## TODO: implement merge test and val dataset CICIDS2017
        tr_ds, val_ds, ood_ds = load_dataset(arg.name, arg.bs, transform=transform, num_worker=arg.worker, target_classes=normal_classes)
        print("N sample of training dataset:", len(tr_ds.dataset))
        print("N sample of validation dataset:", len(val_ds.dataset))
        if arg.wandb:
            try:
                wandb.init(
                    project="DEC-DT",  # Set the project where this run will be logged
                    name=arg.name,
                )
            except Exception:
                print("WANDB not available!", flush=True)
                arg.wandb = False
        dec = DEC(device, arg.input_dim, output_dir, arg.k, arg.wandb)
        dec.pretrain(tr_ds, arg.pre_epoch)
        dec.train(tr_ds, val_ds, arg.epoch, alpha, beta, omega)


        if arg.wandb:
            wandb.finish()

        del tr_ds, val_ds  # Free up memory

        # evaluate test set and zero days. it produces the input file for next step (ML step)
        ## TODO: Code to evaluate DEC also with new dataset

        # evaluate test set and zero days. it produces the input file for next step (ML step)

        dec.eval(arg.name, ood_ds)

        del ood_ds  # Free up memory

        # if UNSW:
        #     print("UNSW - only for test:")
        #     # evaluate UNSW sample. it is input file for next step (ML step)
        #     csv_path = 'dataset/Payload_data_UNSW.csv'
        #     UNSW_ds = load_cyber(csv_path, arg.bs, num_worker=arg.worker)
        #     eval_DEC('UNSW', UNSW_ds, output_dir)
        #     del UNSW_ds

    if arg.XG:
        # Read data and define feature and target names
        feature_name = [f"f_{i}" for i in range(0, arg.features)]
        target_name = ['real_label', 'OOD']
        df = pd.read_csv(os.path.join(output_dir, f'{arg.name}_result_test.csv'))
        df['OOD'] = [0 if l in normal_classes else 1 for l in df['real_label']]
        auroc = []
        for r in remove:
            # Loop over attacks to evaluate DT, XGBoost and LoOP
            print('*' * 50)
            print(f'Prediction attack {r}:')

            X_train, y_train, X_c, y_c, X_test, y_test = preprocess_data(df, [r], feature_name, target_name)
            xg_best_model = grid_search_xgboost(X_train, y_train['OOD'])
            auroc.append(evalXG(xg_best_model, X_c, y_c, y_c['real_label'], name=f"results/mnist/{r}_"))

        print(auroc)

if __name__ == '__main__':
    main()
