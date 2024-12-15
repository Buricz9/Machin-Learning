import pandas as pd

def preprocess_data(path):
    data = pd.read_csv(path)
    data = data.dropna(subset=['Ventilation support status'])
    data = pd.get_dummies(data, columns=['Gender'], drop_first=True)
    data['Vital status'] = data['Vital status'].apply(lambda x: 1 if x == 'Deceased' else 0)
    X = data.drop(columns=['Case ID', 'Vital status', 'Ventilation support status'])
    y = data['Vital status']
    return X, y
