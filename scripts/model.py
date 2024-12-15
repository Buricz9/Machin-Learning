from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np

def train_svm_model_with_randomized_search(X_train, y_train):
    # Definiowanie przestrzeni przeszukiwania hiperparametrów
    param_dist = {
        'C': np.logspace(-4, 2, 20),  # Zakres wartości hiperparametru C
        'gamma': np.logspace(-4, 1, 20),  # Zakres wartości hiperparametru gamma
        'kernel': ['rbf', 'linear']  # Rodzaje jąder do przetestowania
    }

    # Inicjalizacja modelu SVM
    model = SVC(probability=True, random_state=42)

    # RandomizedSearchCV do optymalizacji hiperparametrów
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,  # Liczba prób losowego przeszukiwania
        cv=5,  # Walidacja krzyżowa
        scoring='roc_auc',  # Metryka optymalizacji
        n_jobs=-1,
        random_state=42
    )

    # Trening modelu z RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Zwracanie najlepszego modelu i parametrów
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_model, best_params
