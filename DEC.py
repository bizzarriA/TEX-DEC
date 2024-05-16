from torch import nn
import torch
from torch.optim import AdamW, SGD

import wandb
import pandas as pd
import numpy as np

from utils.nn import DEC_model, AutoEncoder, get_p
from utils.for_eval import History, plot, set_data_plot, accuracy
from utils.for_train import get_initial_center


class DEC:
    def __init__(self, device, input_dim, save_dir, k, is_wandb=True):
        self.device = device
        self.input_dim = input_dim
        self.wandb = is_wandb
        self.save_dir = save_dir
        self.k = k
        self.center = None
        self.label_name = None

    def pretrain(self, ds, epochs):
        # train autoencoder
        self.ae = AutoEncoder(self.input_dim).to(self.device)
        opt = AdamW(self.ae.parameters())
        print('begin train AutoEncoder ...', flush=True)
        loss_fn = nn.MSELoss()
        n_batch = len(ds)
        self.ae.train()
        loss_h = History('min')
        if epochs > 0:
            # train AE
            for epoch in range(1, epochs + 1):
                print(f'\nEpoch {epoch}:', flush=True)
                print('-' * 10)
                loss = 0.
                for i, (x, y, _) in enumerate(ds, 1):
                    opt.zero_grad()
                    x = x.to(self.device)
                    _, gen = self.ae(x)
                    batch_loss = loss_fn(x, gen)
                    batch_loss.backward()
                    opt.step()
                    loss += batch_loss
                    print(f'{i}/{n_batch}', end='\r')

                loss /= n_batch
                loss_h.add(loss)
                if loss_h.better:
                    torch.save(self.ae, f'{self.save_dir}/fine_tune_AE.pt')
                if self.wandb:
                    wandb.log({"loss_AE": loss.item()})
                print(f'loss : {loss.item():.4f}  min loss : {loss_h.best.item():.4f}')
                print(f'lr: {opt.param_groups[0]["lr"]}')
        else:
            try:
                self.ae = torch.load(f'{self.save_dir}/fine_tune_AE.pt', self.device)
            except FileNotFoundError:
                print(
                    f'{self.save_dir}/fine_tune_AE.pt not found!\nI\'m creating new AE from scratch, without pretrain...')
        self.center = get_initial_center(self.ae.encoder, ds, self.device, self.k)

    def train(self, ds, val_ds, epochs, alpha=1, beta=1, omega=1):
        # for visualize
        set_data_plot(ds, val_ds, self.device)
        # train dec
        print('\nload the best encoder and build DEC ...', flush=True)
        model = DEC_model(self.ae.encoder, self.center, alpha=1).to(self.device)
        print(f'DEC param: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M', flush=True)
        opt = SGD(model.parameters(), 0.01, 0.9, nesterov=True)
        print('begin train DEC ...')

        loss_fn_kld = nn.KLDivLoss(reduction='batchmean')
        n_sample, n_batch = len(ds.dataset), len(ds)
        loss_h, acc_cluster_h = History('min'), History('max')

        for epoch in range(1, epochs + 1):
            print(f'\nEpoch {epoch}:', flush=True)
            print('-' * 10)
            model.train()
            closs = 0.
            kloss = 0.
            clusterloss = 0.
            for i, (x, labels, _) in enumerate(ds, 1):
                opt.zero_grad()
                x = x.to(self.device)
                labels = labels.to(self.device)
                y = torch.tensor([0 if y == self.label_name else 1 for y in labels])
                q = model(x)

                # kl divergence
                loss_kld = alpha * loss_fn_kld(q.log(), get_p(q))

                # contrastive loss
                centroids = model.cluster.center
                num_clusters = len(centroids)
                centroid_distances = torch.sum(torch.cdist(centroids, centroids, p=num_clusters)) / (
                        num_clusters * (num_clusters - 1))
                loss_cluster = beta * 1 / centroid_distances

                # classification loss
                y = nn.functional.one_hot(y, num_classes=num_clusters).float().to(self.device)
                loss_cls = omega * torch.nn.functional.cross_entropy(y, q)

                # total loss
                batch_loss = loss_kld + loss_cluster + loss_cls
                batch_loss.backward()
                opt.step()
                closs += loss_cls
                kloss += loss_kld
                clusterloss += loss_cluster
                print(f'{i}/{n_batch}', end='\r')

            loss = kloss + clusterloss + closs
            loss /= n_batch
            loss_h.add(loss)
            print(f'loss : {loss:.4f}  min loss : {loss_h.best:.4f} = ')
            print(
                f'class loss : {closs / n_batch:.4f} + \n'
                f'KL loss : {kloss / n_batch:.4f} + \n'
                f'cluster Loss: {clusterloss / n_batch:.4f}',
                flush=True
            )
            acc_cluster = accuracy(model, ds, self.device)
            acc_cluster_h.add(acc_cluster)
            print(f'acc cluster : {acc_cluster:.4f}  max acc : {acc_cluster_h.best:.4f}')
            print(f'lr: {opt.param_groups[0]["lr"]}', flush=True)
            if loss_h.better:
                torch.save(model, f'{self.save_dir}/DEC_best.pt')
            if self.wandb:
                wandb.log({"loss": loss.item(),
                           "loss_kl": kloss / n_batch,
                           "loss_cont": clusterloss / n_batch,
                           "loss_class": closs / n_batch,
                           # "accuracy": acc_cluster
                           })
            if epoch % 5 == 0:
                plot(model, self.save_dir, 'train', epoch)
        torch.save(model, f'{self.save_dir}/DEC.pt')

        df = pd.DataFrame(zip(range(1, epoch + 1), loss_h.history, acc_cluster_h.history),
                          columns=['epoch', 'loss', 'acc_cluster'])
        df.to_excel(f'{self.save_dir}/train.xlsx', index=False)
        print()
        print('*' * 50)
        print('load the best DEC ...', flush=True)
        dec = torch.load(f'{self.save_dir}/DEC.pt', self.device)
        print('Evaluate ...', flush=True)
        acc_cluster = accuracy(dec, val_ds, self.device)
        print(f'test acc cluster: {acc_cluster:.4f}', flush=True)
        print('*' * 50)
        plot(dec, self.save_dir, 'test')

    def eval(self, name, OOD_ds, test_ds=None):
        # evaluate DEC and print {name}_test_result.csv - it is input file for next step (ML step)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"{name} test OOD ad ID")
        print()
        print('*' * 50)
        print('load the best DEC ...')
        dec = torch.load(f'{self.save_dir}/DEC.pt', device)
        print('Evaluate ...')
        print('*' * 50)
        dec.eval()
        feature, y, ood_id, pred, ids = [], [], [], [], []
        with torch.no_grad():
            for batch in OOD_ds:
                data, labels, id = batch
                data = data.to(device)
                x = dec.encoder(data)
                feature.append(x)
                labels = labels.to(device)
                ids.append(id)
                ood_id.append(torch.full(labels.shape, 1))
                y.append(labels)
            if test_ds is not None:
                for batch in test_ds:
                    data, labels, id = batch
                    data = data.to(device)
                    x = dec.encoder(data)
                    labels = labels.to(device)
                    feature.append(x)
                    ood_id.append(torch.full(labels.shape, 0))
                    y.append(labels)
                    ids.append(id)
            feature = torch.cat(feature)
            y = torch.cat(y).cpu()
            ids = torch.cat(ids).cpu()
            ood_id = torch.cat(ood_id).cpu()
            pred = dec.cluster(feature)[:, 1].cpu().numpy()
            feature = feature.cpu()

            print('save csv ...')
            feature_names = [f"f_{i}" for i in range(feature.shape[1])]
            df = pd.DataFrame(feature, columns=feature_names)
            df['real_label'] = y
            df['pred'] = pred
            df['OOD'] = ood_id
            df['unique_id'] = ids
            print(np.unique(df.real_label, return_counts=True))
            df.to_csv(self.save_dir + f'/{name}_result_test.csv', index=False)
            centroids = dec.cluster.center
            df_center = pd.DataFrame(centroids, columns=feature_names)
            df_center.to_csv(self.save_dir + f'/{name}_center.csv', index=False)
