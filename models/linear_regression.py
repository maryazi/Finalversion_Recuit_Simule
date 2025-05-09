import numpy as np
import pickle

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialisation des paramètres
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        # Descente de gradient
        for i in range(self.n_iterations):
            # Prédiction
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calcul des gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Mise à jour des paramètres
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calcul et stockage du coût (MSE)
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'weights': self.weights, 'bias': self.bias}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.weights = params['weights']
        self.bias = params['bias']