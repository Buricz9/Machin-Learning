import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from preprocessing import preprocess_data, prepare_data
from model import optimize_logistic_regression_with_optuna
from evaluation import evaluate_model, report_metrics, plot_confusion_matrix, plot_roc_curve

# Ścieżka do pliku z danymi
data_path = 'data/KAMC_COVID-19.csv'

# 1. Przetwarzanie danych
X, y = preprocess_data(data_path)
X_train, X_test, y_train, y_test = prepare_data(X, y)

# 2. Trening i optymalizacja modelu Logistic Regression z Optuna
best_model, best_params = optimize_logistic_regression_with_optuna(X_train, y_train)
print(f"Best parameters found: {best_params}")

# 3. Ewaluacja modelu
accuracy, auc, f1, precision, recall = evaluate_model(best_model, X_test, y_test)
report_metrics(accuracy, auc, f1, precision, recall)

# 4. Generowanie i wyświetlanie wykresów
y_pred = best_model.predict(X_test)
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(best_model, X_test, y_test)
