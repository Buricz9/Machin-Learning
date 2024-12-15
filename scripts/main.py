import matplotlib
matplotlib.use('TkAgg')

from preprocessing import preprocess_data, prepare_data
from model import optimize_lightgbm_with_randomsearch
from evaluation import evaluate_model, report_metrics, plot_confusion_matrix, plot_roc_curve, plot_shap_summary_polish, plot_shap_bar_polish

# Ścieżka do pliku z danymi
data_path = 'data/KAMC_COVID-19.csv'

# Preprocessing and preparing data
X, y = preprocess_data(data_path)
X_train, X_test, y_train, y_test = prepare_data(X, y)

# Optimizing and training the model
best_model, best_params = optimize_lightgbm_with_randomsearch(X_train, y_train)
print(f"Best parameters found: {best_params}")

# Evaluating the model
accuracy, auc, f1, precision, recall = evaluate_model(best_model, X_test, y_test)
report_metrics(accuracy, auc, f1, precision, recall)

# Plotting evaluation metrics
y_pred = best_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(best_model, X_test, y_test)

# Pobranie nazw cech z danych
nazwy_cech = X.columns.tolist()

# Generowanie wykresów SHAP z polskimi nazwami cech
print("Generowanie wykresów SHAP...")
plot_shap_summary_polish(best_model, X_train, nazwy_cech)
print("Wykres sumaryczny SHAP wygenerowany.")
plot_shap_bar_polish(best_model, X_train, nazwy_cech)
print("Wykres paskowy SHAP wygenerowany.")