import torch
from scipy.optimize import linear_sum_assignment

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score
from matplotlib import pyplot as plt
import numpy as np

def accuracy(model, ds, device, OOD_ds=None):
    truth, pred_cluster = [], []
    model.eval()
    with torch.no_grad():
        for x, y, _ in ds:
            x = x.to(device)
            truth.append(y)
            _, pred = model(x)
            pred_cluster.append(pred.max(1)[1].cpu())
        if OOD_ds:
            for x, y, _ in OOD_ds:
                x = x.to(device)
                truth.append(torch.full(y.shape, 2))
                _, pred = model(x)
                pred_cluster.append(pred.max(1)[1].cpu())
    y_pred = [0 if p < 0.5 else 1 for p in torch.cat(pred_cluster)]
    confusion_m = confusion_matrix(torch.cat(truth).numpy(), torch.cat(pred_cluster).numpy())
    _, col_idx = linear_sum_assignment(confusion_m, maximize=True)
    acc = np.trace(confusion_m[:, col_idx]) / confusion_m.sum()
    # acc = accuracy_score(torch.cat(truth).numpy(), torch.cat(pred_cluster).numpy())
    return acc

def normalize(X):
    return torch.exp(1 - X)


def set_data_plot(tr_ds, test_ds, device):
    global DATA_PLOT

    # select 100 sample per class
    tr_x, tr_y = [], []
    count = torch.zeros(15, dtype=torch.int)
    for batch in tr_ds:
        for data, label, id in zip(*batch):
            if count[label] < 100:
                tr_x.append(data[None])
                tr_y.append(label[None])
            count[label] += 1
    tr_x, tr_y = torch.cat(tr_x).to(device), torch.cat(tr_y).to(device)

    # select 100 sample per class
    test_x, test_y = [], []
    count = torch.zeros(15, dtype=torch.int)
    if test_ds:
        for batch in test_ds:
            for data, label, _ in zip(*batch):
                if count[label] < 100:
                    test_x.append(data[None])
                    test_y.append(label[None])
                count[label] += 1
        test_x, test_y = torch.cat(test_x).to(device), torch.cat(test_y).to(device)

    DATA_PLOT = {'train': (tr_x, tr_y),
                 'test': (test_x, test_y)}


class History:
    def __init__(self, target='min'):
        self.value = None
        self.best = float('inf') if target == 'min' else 0.
        self.n_no_better = 0
        self.better = False
        self.target = target
        self.history = []
        self._check(target)

    def add(self, value):
        if self.target == 'min' and value < self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        elif self.target == 'max' and value > self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        else:
            self.n_no_better += 1
            self.better = False

        self.value = value
        self.history.append(value)

    def _check(self, target):
        if target not in {'min', 'max'}:
            raise ValueError('target only allow "max" or "min" !')


def plot(model, save_dir, target='train', epoch=None):
    # plot latent space cluster in 2D
    assert target in {'train', 'test'}
    assert len(DATA_PLOT[target]) > 0
    print('plotting ...')

    model.eval()
    with torch.no_grad():
        feature = model.encoder(DATA_PLOT[target][0])
        pred = model.cluster(feature).max(1)[1].cpu().numpy()

    feature_2D = TSNE(2).fit_transform(feature.cpu().numpy())
    plt.scatter(feature_2D[:, 0], feature_2D[:, 1], 4, pred, cmap='Paired')
    if epoch is None:
        plt.title(f'Test data')
        plt.savefig(f'{save_dir}/test.png')
    else:
        plt.title(f'Epoch: {epoch}')
        plt.savefig(f'{save_dir}/epoch_{epoch}.png')
    plt.close()
    if epoch is None:
        plt.scatter(feature_2D[:, 0], feature_2D[:, 1], 16, DATA_PLOT[target][1].cpu().numpy(), cmap='Paired')
        plt.title(f'Test data real label')
        plt.savefig(f'{save_dir}/test_real_label.png')
        plt.close()

def evalXG(bst, X_test, y_test, y_real, index=None, name="attack_estimators_depth"):
    y_probabilities = bst.predict_proba(X_test)[:, 1]
    if index is not None:
        y_probabilities = y_probabilities[index]
        y_test = y_test.iloc[index]
    if name == 'UNSW':
        y_probabilities = np.concatenate([y_probabilities, np.random.rand(100)])
    feature_names = [f"f_{i}" for i in range(X_test.shape[1])]
    threshold = find_optimal_threshold(y_test, y_probabilities)
    preds = [0 if prob < threshold else 1 for prob in y_probabilities]
    df = pd.DataFrame(X_test, columns=feature_names)
    df['prob_XGB'] = y_probabilities
    df['OOD'] = np.array(y_test)
    df['real_label'] = np.array(y_real)
    df['pred_XGB'] = preds
    df.to_csv(f"{name}_XGBoost_output.csv")
    print(f"\tthreshold:{threshold}")
    acc = accuracy_score(y_test, preds)
    print(f"\tacc: {acc}")
    f1 = f1_score(y_test, preds)
    print(f"\tf1-score: {f1}")
    n_class = len(np.unique(y_test))
    if n_class > 1:
        auroc = roc_auc_score(y_test, y_probabilities)
        print(f"\tAUROC: {roc_auc_score(y_test, y_probabilities)}")
    else:
        auroc = None
    cm = confusion_matrix(y_test, preds)
    print(f"\tconfusion matrix: \n{cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f"{name}_OOD.png")
    #
    # TNR, TPR = true_negative_rate(y_probabilities, y_test)
    #
    # print(f"True Negative Rate (TNR) @ 95% TPR: {TNR} (@{TPR})")

    return auroc