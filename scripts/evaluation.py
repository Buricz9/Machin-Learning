# evaluation.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import numpy as np
# Preprocessing and preparing data
def preprocess_data(path):
    data = pd.read_csv(path)
    data = data.dropna(subset=['Ventilation support status'])
    data = pd.get_dummies(data, columns=['Gender'], drop_first=True)
    data['Vital status'] = data['Vital status'].apply(lambda x: 1 if x == 'Deceased' else 0)
    X = data.drop(columns=['Case ID', 'Vital status', 'Ventilation support status'])
    y = data['Vital status']
    return X, y

def prepare_data(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Optimizing LightGBM with RandomizedSearchCV
def optimize_lightgbm_with_randomsearch(X_train, y_train):
    param_dist = {
        'n_estimators': np.arange(100, 501, 100),
        'max_depth': np.arange(5, 30, 5),
        'learning_rate': np.linspace(0.01, 0.2, 5),
        'num_leaves': np.arange(20, 150, 20),
        'subsample': np.linspace(0.6, 1.0, 5),
        'colsample_bytree': np.linspace(0.6, 1.0, 5)
    }

    model = LGBMClassifier(random_state=42)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, scoring='roc_auc', verbose=1, random_state=42)
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    return best_model, best_params

# Evaluation metrics and plots
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return accuracy, auc, f1, precision, recall

def report_metrics(accuracy, auc, f1, precision, recall):
    print(f'Accuracy: {accuracy:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def plot_shap_summary_polish(model, X_train, nazwy_cech):
    # Tworzenie objaśniacza SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Tworzenie wykresu SHAP z polskimi etykietami
    shap.summary_plot(
        shap_values, 
        X_train, 
        plot_type="dot", 
        feature_names=nazwy_cech  # Prawdziwe nazwy cech
    )
    plt.xlabel("Wartość SHAP (wpływ na wynik modelu)")
    plt.title("Wykres wartości SHAP (sumaryczny)")
    plt.show()

def plot_shap_bar_polish(model, X_train, nazwy_cech):
    # Tworzenie objaśniacza SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Tworzenie wykresu paskowego SHAP z polskimi etykietami
    shap.summary_plot(
        shap_values, 
        X_train, 
        plot_type="bar", 
        feature_names=nazwy_cech  # Prawdziwe nazwy cech
    )
    plt.xlabel("Średnia wartość SHAP (wpływ na wynik modelu)")
    plt.title("Wykres wartości SHAP (paskowy)")
    plt.show()
