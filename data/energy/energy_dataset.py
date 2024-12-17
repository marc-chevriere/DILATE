import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def create_windows(data, input_length, output_length):
    X, y = [], []
    for i in range(len(data) - input_length - output_length + 1):
        X.append(data[i:i + input_length])
        y.append(data[i + input_length:i + input_length + output_length])
    return np.array(X), np.array(y)

def get_energy_data(path_data, batch_size=64, input_length=12, output_length=1, 
                   train_split=0.6, valid_split=0.2, normalization=True):
    """
    Charge et prépare les données de vents pour l'entraînement, la validation et le test.

    :param path_data: Chemin vers le fichier MS_winds.dat
    :param batch_size: Taille des batches pour les DataLoaders
    :param input_length: Nombre de pas de temps en entrée
    :param output_length: Nombre de pas de temps en sortie
    :param train_split: Fraction des données pour l'entraînement
    :param valid_split: Fraction des données pour la validation
    :param normalization: Booléen pour appliquer ou non la normalisation
    :return: Tuple (train_loader, valid_loader, test_loader)
    """
    # Charger les données avec pandas
    df = pd.read_csv(path_data, header=None, delimiter=',')
    
    # Sélectionner les colonnes pertinentes (57 stations)
    data = df.iloc[:, :57].values  # Assurez-vous que les colonnes 0 à 56 sont les stations
    
    # Normalisation par station (si nécessaire)
    if normalization:
        means = data.mean(axis=0)
        stds = data.std(axis=0)
        data = (data - means) / stds
    else:
        means, stds = None, None
    
    # Créer les fenêtres d'entrée et de sortie
    X, y = create_windows(data, input_length, output_length)
    
    # Diviser les données en ensembles d'entraînement, de validation et de test
    n_windows = X.shape[0]
    train_end = int(train_split * n_windows)
    valid_end = int((train_split + valid_split) * n_windows)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_valid, y_valid = X[train_end:valid_end], y[train_end:valid_end]
    X_test, y_test = X[valid_end:], y[valid_end:]
    
    # Convertir en tenseurs PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Créer les DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), 
                              batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(TensorDataset(X_valid_tensor, y_valid_tensor), 
                              batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), 
                             batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, valid_loader, test_loader
