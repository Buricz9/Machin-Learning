import matplotlib
matplotlib.use('TkAgg')

from preprocessing import preprocess_data, prepare_data
from model import optimize_lightgbm_with_randomsearch
from evaluation import evaluate_model, report_metrics, plot_confusion_matrix, plot_roc_curve, plot_shap_summary_polish, plot_shap_bar_polish

# Ścieżka do pliku z danymi
data_path = 'data/KAMC_COVID-19.csv'

# Preprocessing i przygotowanie danych
X, y = preprocess_data(data_path)
X_train, X_test, y_train, y_test = prepare_data(X, y)

# Optymalizacja i trening modelu LightGBM z RandomizedSearch
best_model, best_params = optimize_lightgbm_with_randomsearch(X_train, y_train)
print(f"Best parameters found: {best_params}")

# Ewaluacja modelu
accuracy, auc, f1, precision, recall = evaluate_model(best_model, X_test, y_test)
report_metrics(accuracy, auc, f1, precision, recall)

# Wykresy ewaluacji
y_pred = best_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(best_model, X_test, y_test)

# Nazwy cech
nazwy_cech = X.columns.tolist()

# Generowanie wykresów SHAP
print("Generowanie wykresów SHAP...")
plot_shap_summary_polish(best_model, X_train, nazwy_cech)
print("Wykres sumaryczny SHAP wygenerowany.")
plot_shap_bar_polish(best_model, X_train, nazwy_cech)
print("Wykres paskowy SHAP wygenerowany.")
