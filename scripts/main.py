import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Ustawienie backendu dla Matplotlib
import matplotlib.pyplot as plt
import shap

from preprocessing import preprocess_data, prepare_data
from model import train_random_forest_model_with_optuna
from evaluation import evaluate_model, report_metrics, plot_confusion_matrix, plot_roc_curve

# 1. Ścieżka do pliku z danymi
data_path = 'data/KAMC_COVID-19.csv'

# 2. Przetwarzanie danych
X, y = preprocess_data(data_path)
X_train, X_test, y_train, y_test = prepare_data(X, y)

# 3. Trening i optymalizacja modelu Random Forest z Optuna
best_model, best_params = train_random_forest_model_with_optuna(X_train, y_train)
print(f"Best parameters found: {best_params}")

# 4. Ewaluacja modelu na zbiorze testowym
accuracy, auc, f1, precision, recall = evaluate_model(best_model, X_test, y_test)
report_metrics(accuracy, auc, f1, precision, recall)

# 5. Generowanie i wyświetlanie wykresów metryk
y_pred = best_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(best_model, X_test, y_test)

# 6. Obliczanie i wizualizacja wartości SHAP
print(f"Liczba cech w X_test: {X_test.shape[1]}")
feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
print("Wygenerowane nazwy cech:")
print(feature_names)

# Tworzenie obiektu explainer dla Random Forest
# Tworzenie TreeExplainer dla modelu
explainer = shap.TreeExplainer(best_model)

# Obliczenie SHAP values
shap_values = explainer.shap_values(X_test)

# Sprawdzenie kształtu shap_values
print(f"shap_values shape (before selection): {shap_values.shape}")

# Wybór SHAP values dla klasy 1 i redukcja wymiaru
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Wybieramy dla klasy 1 (binarna klasyfikacja)
elif len(shap_values.shape) == 3:  # Jeśli trójwymiarowe (interakcje), redukujemy
    shap_values = shap_values[:, :, 1]  # Wybór SHAP wartości dla klasy 1

print(f"shap_values shape (after selection): {shap_values.shape}")

# Konwersja X_test do DataFrame (jeśli nie jest)
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test, columns=feature_names)

# Weryfikacja kształtów
print(f"X_test shape: {X_test.shape}")
print(f"shap_values shape: {shap_values.shape}")

# Generowanie klasycznego SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar")

plt.figure()
shap.summary_plot(shap_values, X_test)