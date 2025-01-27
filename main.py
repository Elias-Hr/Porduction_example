from memory_profiler import profile
from line_profiler import profile as cprofile
from sklearn.model_selection import train_test_split

from utils.DataManager import load_data, preprocess_data
from utils.Pipeline import evaluate_model, save_model, train_model


# 6. Main pour l'exécution
#  ajouter ce paramètre pour voir la consommation de RAM du code
# @profile

#ajouter ce paramètre pour voir la consommation de cpu du code
@cprofile 
def main():
    # Remplacer par le chemin réel du dataset (sur votre machine ou depuis Kaggle)
    dataset_path = "./data/creditcard.csv"  # Assurez-vous que le fichier est bien téléchargé et dans le bon répertoire

    # Charger les données
    df = load_data(dataset_path)

    # Prétraiter les données
    X, y, scaler = preprocess_data(df)

    # Diviser les données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entraîner le modèle
    model = train_model(X_train, y_train)

    # Évaluer le modèle
    print("Évaluation sur l'ensemble de test :")
    evaluate_model(model, X_test, y_test)

    # Sauvegarder le modèle et le scaler
    save_model(model, scaler, model_filename="./models/fd.model.pkl", scaler_filename="./models/fd_scaler.pkl")


if __name__ == "__main__":
    main()
