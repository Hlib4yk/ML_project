import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Завантаження нових даних для передбачень
new_data = pd.read_csv('C:/Users/Lenovo/PycharmProjects/Lab5/cleaned_data/new_input.csv')
X_new = new_data.drop(columns=['y_yes'])

# Завантаження попередньо тренованої моделі (Random Forest як приклад)
# У реальному проекті модель слід серіалізувати, наприклад, через joblib
trained_model = RandomForestClassifier(n_estimators=100, random_state=42)
trained_model.fit(X_new, new_data['y_yes'])  # Тут припускається, що модель вже тренована

# Передбачення
predictions = trained_model.predict(X_new)
new_data['Predicted'] = predictions

# Збереження передбачень
new_data.to_csv('C:/Users/Lenovo/PycharmProjects/Lab5/results/new_input_predictions.csv', index=False)

print("Передбачення завершено! Результати збережено у results/new_input_predictions.csv.")
