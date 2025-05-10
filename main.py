import numpy as np
import matplotlib.pyplot as plt
from utils.data_preparation import prepare_data
from models.linear_regression import LinearRegression

data_path = "data/raw/mubawab_listings_eda.csv"
X_train, X_test, y_train, y_test, X_mean, X_std, features = prepare_data(data_path)
print("Features utilisées:", features)

# Initialisation et entraînement du modèle
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Sauvegarde du modèle
model.save('models/trained_model.pkl')

# Évaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calcul des métriques
def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    return mse, rmse, mae, r2


mse_train, rmse_train, mae_train, r2_train = compute_metrics(y_train, y_pred_train)
mse_test, rmse_test, mae_test, r2_test = compute_metrics(y_test, y_pred_test)

print(f"Train - MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, MAE: {mae_train:.2f}, R2: {r2_train:.2f}")
print(f"Test - MSE: {mse_test:.2f}, RMSE: {rmse_test:.2f}, MAE: {mae_test:.2f}, R2: {r2_test:.2f}")

# Visualisations
plt.figure(figsize=(15, 5))

# Courbe d'apprentissage
plt.subplot(1, 2, 1)
plt.plot(model.loss_history)
plt.title('Courbe de convergence')
plt.xlabel('Itérations')
plt.ylabel('MSE')
plt.grid()

# Prédictions vs Réelles
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Prédictions vs Valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.grid()

plt.tight_layout()
plt.savefig('results/performance.png')
plt.show()