import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(path):
    data = pd.read_csv(path)
    data = data.dropna(subset=['Ventilation support status'])
    data = pd.get_dummies(data, columns=['Gender'], drop_first=True)
    data['Vital status'] = data['Vital status'].apply(lambda x: 1 if x == 'Deceased' else 0)
    X = data.drop(columns=['Case ID', 'Vital status', 'Ventilation support status'])
    y = data['Vital status']
    return X, y

def prepare_data(X, y):
    # Podział danych na zbiory treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    # Stosowanie SMOTE tylko na zbiorze treningowym
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Skalowanie danych
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    return X_train_res, X_test, y_train_res, y_test
