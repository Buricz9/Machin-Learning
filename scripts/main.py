import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from preprocessing import preprocess_data
from model import train_random_forest_model, optimize_random_forest_model
from evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, adjust_threshold

# Ścieżka do pliku z danymi
data_path = 'data/KAMC_COVID-19.csv'

# 1. Przetwarzanie danych
X, y = preprocess_data(data_path)

# 2. Trening modelu Random Forest
rf_model, X_train, X_test, y_train, y_test = train_random_forest_model(X, y)

# 3. Optymalizacja modelu
best_model, best_params = optimize_random_forest_model(rf_model, X_train, y_train)
print(f"Best parameters found: {best_params}")

# 4. Ewaluacja przy domyślnym progu (0.5)
accuracy, auc = evaluate_model(best_model, X_test, y_test)
print(f'Optimized Accuracy: {accuracy:.2f}')
print(f'Optimized AUC: {auc:.2f}')

# 5. Testowanie różnych progów decyzyjnych
best_threshold = 0.5
best_f1 = 0

for threshold in [i * 0.01 for i in range(0, 101)]:
    y_pred_adj = adjust_threshold(best_model, X_test, threshold)
    
    accuracy_adj = accuracy_score(y_test, y_pred_adj)
    auc_adj = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    f1_adj = f1_score(y_test, y_pred_adj)
    precision_adj = precision_score(y_test, y_pred_adj)
    recall_adj = recall_score(y_test, y_pred_adj)
    
    print(f'Threshold: {threshold:.2f} | Adjusted Accuracy: {accuracy_adj:.2f} | Adjusted AUC: {auc_adj:.2f} | Adjusted F1-Score: {f1_adj:.2f} | Precision: {precision_adj:.2f} | Recall: {recall_adj:.2f}')
    
    if f1_adj > best_f1:
        best_f1 = f1_adj
        best_threshold = threshold

print(f"\nBest threshold found: {best_threshold:.2f} with F1-Score: {best_f1:.2f}")

# 6. Ewaluacja modelu przy najlepszym progu
y_pred_best = adjust_threshold(best_model, X_test, best_threshold)
accuracy_best = accuracy_score(y_test, y_pred_best)
auc_best = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f'Best Threshold Accuracy: {accuracy_best:.2f}')
print(f'Best Threshold AUC: {auc_best:.2f}')

# 7. Generowanie i wyświetlanie wykresów
plt.close('all')
plot_confusion_matrix(y_test, y_pred_best)
plot_roc_curve(best_model, X_test, y_test)
