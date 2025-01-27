import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


# 3. Entraîner un modèle
def train_model(X, y):
    print("start traning")
    # model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model


# 4. Évaluer le modèle
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


# 5. Sauvegarder le modèle et le scaler
def save_model(
    model,
    scaler,
    model_filename="fraud_detection_model.pkl",
    scaler_filename="scaler.pkl",
):
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(
        f"Modèle et scaler sauvegardés dans '{model_filename}' et '{scaler_filename}'."
    )
