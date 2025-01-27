import pandas as pd
from sklearn.preprocessing import StandardScaler

# from models.trainer import laod_model


# 1. Charger les données
def load_data(file_path: str):
    df = pd.read_csv(file_path)
    # print(df.head())
    return df


# 2. Prétraitement des données
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame :
    """ma fonction permet de blabala

    Args:
        df (pd.DataFrame): data raw non procesess en format pandas

    Returns:
        pd.DataFrame: _description_
    """

    # Afficher les informations de base
    # print(df.info())

    # Vérifier les valeurs manquantes
    # print(df.isnull().sum())

    # Diviser les features et la target
    X = df.drop(
        columns=["Class"]
    )  # 'Class' est la colonne cible (fraude=1, non fraude=0)
    y = df["Class"]

    # Standardiser les données (normalisation)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
if __name__ == "__main__":
    print("j'excute le module")
    load_data("./inputs/creditcard.csv")