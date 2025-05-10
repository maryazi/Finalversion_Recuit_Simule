import numpy as np
import matplotlib.pyplot as plt
from utils.data_preparation import prepare_data
from models.linear_regression import LinearRegression
# Chargement et préparation des données
data_path = "data/raw/mubawab_listings_eda.csv"
X_train, X_test, y_train, y_test, X_mean, X_std, features = prepare_data(data_path)

# Entraînement du modèle
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Prédictions sur le test set
y_pred = model.predict(X_test)

# 🔹 Calcul des métriques d’évaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 🔸 Affichage des résultats
print("\nÉvaluation du modèle sur le test set :")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R²   : {r2:.4f}")
