import numpy as np
import pickle

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.mean = None
        self.std = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Calcul et stockage de la moyenne et l'écart-type (hors biais)
        self.mean = np.mean(X[:, 1:], axis=0)
        self.std = np.std(X[:, 1:], axis=0)

        # Initialisation des paramètres
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'bias': self.bias,
                'loss_history': self.loss_history,
                'mean': self.mean,
                'std': self.std
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.weights = params['weights']
        self.bias = params['bias']
        self.loss_history = params.get('loss_history', [])
        self.mean = params.get('mean', None)
        self.std = params.get('std', None)
