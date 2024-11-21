import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Завантаження тренувальних даних
data = pd.read_csv('C:/Users/Lenovo/PycharmProjects/Lab5/cleaned_data/cleaned_data1.csv')
X = data.drop(columns=['y_yes'])
y = data['y_yes']

# Розділення на train/test для тренування
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Моделі
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Тренування моделей
model_performance = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    performance = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_val, y_pred),
        'F1-Score': f1_score(y_val, y_pred),
        'ROC-AUC': roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]),
    }
    model_performance.append(performance)

# Збереження результатів
results = pd.DataFrame(model_performance)
results.to_csv('C:/Users/Lenovo/PycharmProjects/Lab5/results/model_performance.csv', index=False)

print("Тренування завершено! Результати збережено у results/model_performance.csv.")
