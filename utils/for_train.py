from sklearn.cluster import KMeans

import torch
from torch.nn import Parameter

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV



def get_initial_center(model, ds, device, n_cluster):
    # fit
    print('\nbegin fit kmeans++ to get initial cluster centroids ...')

    model.eval()
    with torch.no_grad():
        feature = []
        for x, _, _ in ds:
            x = x.to(device)
            feature.append(model(x).cpu())

    kmeans = KMeans(n_cluster).fit(torch.cat(feature).numpy())
    center = Parameter(torch.tensor(kmeans.cluster_centers_,
                                    device=device,
                                    dtype=torch.float))

    return center


def grid_search_xgboost(X_train, y_train):
    # Definisci la griglia dei parametri da cercare
    param_grid = {
        'n_estimators': range(10, 91, 1),#[2, 3, 4, 5, 6, 7, 8, 9],
        'max_depth': range(2, 6, 1)#[2, 3, 4, 5, 6, 7, 8]
    }

    # Inizializza il classificatore XGBoost
    xgb = XGBClassifier(objective='binary:logistic')

    # Inizializza GridSearchCV
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, scoring='roc_auc', verbose=1)

    # Esegui la grid search sui dati di addestramento
    grid_search.fit(X_train, y_train)

    # Stampa i migliori parametri trovati
    print("Migliori parametri trovati dalla grid search:")
    print(grid_search.best_params_)

    # Utilizza il classificatore con i migliori parametri per fare previsioni
    best_xgb = grid_search.best_estimator_

    print('*' * 50)
    print('OOD zero days test set utilizzando i migliori parametri:')
    return best_xgb