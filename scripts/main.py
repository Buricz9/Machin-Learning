import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from preprocessing import preprocess_data, prepare_data
from model import train_random_forest_model_with_optuna
from evaluation import evaluate_model, report_metrics, plot_confusion_matrix, plot_roc_curve

# Ścieżka do pliku z danymi
data_path = 'data/KAMC_COVID-19.csv'

# 1. Przetwarzanie danych
X, y = preprocess_data(data_path)
X_train, X_test, y_train, y_test = prepare_data(X, y)

# 2. Trening i optymalizacja modelu Random Forest z Optuna
best_model, best_params = train_random_forest_model_with_optuna(X_train, y_train)
print(f"Best parameters found: {best_params}")

# 3. Ewaluacja modelu
accuracy, auc, f1, precision, recall = evaluate_model(best_model, X_test, y_test)
report_metrics(accuracy, auc, f1, precision, recall)

# 4. Generowanie i wyświetlanie wykresów
y_pred = best_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(best_model, X_test, y_test)

# 5. Obliczenie wartości SHAP i generowanie wykresów SHAP
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from preprocessing import preprocess_data, prepare_data
from model import train_random_forest_model_with_optuna
from evaluation import evaluate_model, report_metrics, plot_confusion_matrix, plot_roc_curve

# Ścieżka do pliku z danymi
data_path = 'data/KAMC_COVID-19.csv'

# 1. Przetwarzanie danych
X, y = preprocess_data(data_path)
X_train, X_test, y_train, y_test = prepare_data(X, y)

# 2. Trening i optymalizacja modelu Random Forest z Optuna i walidacją krzyżową
best_model, best_params = train_random_forest_model_with_optuna(X_train, y_train)
print(f"Best parameters found: {best_params}")

# 3. Ewaluacja modelu na zbiorze testowym
accuracy, auc, f1, precision, recall = evaluate_model(best_model, X_test, y_test)
report_metrics(accuracy, auc, f1, precision, recall)

# 4. Generowanie i wyświetlanie wykresów metryk
y_pred = best_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(best_model, X_test, y_test)

# 5. Obliczanie i wizualizacja wartości SHAP
import shap

# Tworzymy obiekt TreeExplainer dla modelu
explainer = shap.TreeExplainer(best_model)

# Obliczamy wartości SHAP dla danych walidacyjnych X_test
# Dla klasyfikacji binarnej zwracane są dwie macierze - bierzemy różnicę między klasami
shap_values = explainer.shap_values(X_test)

# Jeśli shap_values jest listą, wybieramy SHAP dla klasy 1 (lub sumujemy dla większej czytelności)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Dla klasy 1 (pozytywna klasa)

# Sprawdzamy zgodność wymiarów
print("shap_values:", shap_values.shape)
print("X_test:", X_test.shape)

# Generowanie wykresu podsumowującego - bar plot (globalna ważność cech)
shap.summary_plot(shap_values, X_test, plot_type='bar')

# Klasyczny wykres podsumowujący (dot plot)
shap.summary_plot(shap_values, X_test)
