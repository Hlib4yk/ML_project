import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Завантаження даних
data = pd.read_csv('C:/Users/Lenovo/PycharmProjects/Lab5/cleaned_data/cleaned_data1.csv')
X = data.drop(columns=['y_yes'])
y = data['y_yes']

# Розподіл на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Масштабування ознак
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Застосування SMOTE для збільшення меншості
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Тренування моделей
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    print(f'{name} Accuracy:', accuracy_score(y_test, y_pred))
    print(f'{name} F1-score:', f1_score(y_test, y_pred))
    print(f'{name} ROC-AUC:', roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Пошук оптимальних гіперпараметрів для Random Forest за допомогою RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

scorer = {'f1': make_scorer(f1_score), 'roc_auc': 'roc_auc'}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    scoring=scorer,
    refit='f1',  # Вибір моделі на основі F1-score
    n_iter=50,
    cv=5,
    random_state=42,
    verbose=2,
    n_jobs=-1
)

random_search.fit(X_train_sm, y_train_sm)

# Виведення найкращих гіперпараметрів та оцінки моделі
print("Best F1-score:", random_search.best_score_)
print("Best hyperparameters:", random_search.best_params_)

# Оцінка на тестовому наборі
best_rf = random_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

f1_best = f1_score(y_test, y_pred_best_rf)
roc_auc_best = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])

print("Best Random Forest F1-score on test data:", f1_best)
print("Best Random Forest ROC-AUC on test data:", roc_auc_best)
