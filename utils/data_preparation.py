import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data_path = "data/raw/mubawab_listings_eda.csv"

def prepare_data(data_path, test_size=0.2, random_state=42):
    # Chargement des données
    df = pd.read_csv(data_path)
    
    # Sélection des features pertinentes
    features = ['Area', 'Rooms', 'Bathrooms', 'Floor']  # Features numériques
    target = 'Price'  # Variable cible
    
    # Conversion des colonnes au format numérique
    for col in features + [target]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Suppression des lignes avec valeurs manquantes
    df = df.dropna(subset=features + [target])
    
    # Séparation features/target
    X = df[features].values
    y = df[target].values.reshape(-1, 1)
    
    # Normalisation (Z-score)
    def normalize(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std, mean, std
    
    X_normalized, X_mean, X_std = normalize(X)
    
    # Ajout du biais
    X_normalized = np.hstack([np.ones((X_normalized.shape[0], 1)), X_normalized])
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test, X_mean, X_std, features