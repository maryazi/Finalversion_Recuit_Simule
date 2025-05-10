from flask import Flask, render_template, request, flash
import numpy as np
import os
from werkzeug.utils import secure_filename
from models.linear_regression import LinearRegression
from utils.data_preparation import prepare_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration de l'application
app = Flask(__name__)
app.secret_key = 'votre_cle_secrete_ici'
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Initialisation du modèle
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
data_stats = {
    'X_mean': None,
    'X_std': None,
    'features': None,
    'expected_features': None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def create_combined_plot(y_true, y_pred, loss_history):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Courbe de Convergence')
    plt.xlabel('Itérations')
    plt.ylabel('MSE')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('Prédictions vs Valeurs réelles')
    plt.xlabel('Valeurs réelles (DH)')
    plt.ylabel('Prédictions (DH)')
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join('static', 'images', 'performance.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

@app.route('/', methods=['GET', 'POST'])
def index():
    metrics = None
    prediction = None
    plot_path = None
    features_list = []

    if request.method == 'POST':
        try:
            # Entraînement du modèle
            if 'train' in request.form:
                file = request.files.get('dataset')
                
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)

                    X_train, X_test, y_train, y_test, X_mean, X_std, features = prepare_data(filepath)
                    data_stats.update({
                        'X_mean': X_mean,
                        'X_std': X_std,
                        'features': features,
                        'expected_features': X_train.shape[1]
                    })
                    features_list = features.tolist() if hasattr(features, 'tolist') else features

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mse = np.mean((y_test - y_pred)**2)
                    metrics = {
                        "mse": round(mse, 2),
                        "rmse": round(np.sqrt(mse), 2),
                        "mae": round(np.mean(np.abs(y_test - y_pred)), 2),
                        "r2": round(1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2), 3)
                    }

                    plot_path = create_combined_plot(y_test, y_pred, model.loss_history)
                    flash("Modèle entraîné avec succès!", 'success')
                else:
                    flash("Fichier non valide. Veuillez uploader un fichier CSV.", 'error')

            # Prédiction
            elif 'predict' in request.form:
                try:
                    # Vérifie que le modèle a été entraîné
                    if data_stats['features'] is None:
                        raise ValueError("Veuillez d'abord entraîner le modèle")
                    
                    # Récupère les valeurs du formulaire
                    input_values = []
                    for feat in ['area', 'rooms', 'bathrooms', 'floor']:
                        value = request.form.get(feat, '0').strip()
                        if not value:
                            raise ValueError(f"Veuillez remplir le champ {feat}")
                        try:
                            input_values.append(float(value))
                        except ValueError:
                            raise ValueError(f"Valeur invalide pour {feat}")

                    # Crée le tableau d'entrée avec le terme de biais
                    inputs = np.array([[1.0] + input_values])  # 1.0 pour le biais
                    
                    # Normalisation
                    if data_stats['X_mean'] is not None and data_stats['X_std'] is not None:
                        inputs[0, 1:] = (inputs[0, 1:] - data_stats['X_mean']) / data_stats['X_std']
                    
                    # Prédiction
                    prediction = round(float(model.predict(inputs)[0]), 2)
                    flash("Prédiction effectuée avec succès!", 'success')
                
                except ValueError as e:
                    flash(f"Erreur de saisie: {str(e)}", 'error')
                except Exception as e:
                    flash(f"Erreur de prédiction: {str(e)}", 'error')

            # Sauvegarde
            elif 'save' in request.form:
                os.makedirs('models', exist_ok=True)
                model.save('models/trained_model.pkl')
                flash("Modèle sauvegardé avec succès!", 'success')

            # Chargement
            elif 'load' in request.form:
                model.load('models/trained_model.pkl')
                flash("Modèle chargé avec succès!", 'success')

        except Exception as e:
            flash(f"Une erreur est survenue : {str(e)}", 'error')

    return render_template(
        'index.html',
        metrics=metrics,
        prediction=prediction,
        plot_path=plot_path,
        features=features_list,
        stats=data_stats
    )

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)