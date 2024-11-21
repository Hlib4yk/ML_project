import pandas as pd
from sklearn.preprocessing import StandardScaler

# Завантаження даних
data = pd.read_csv('C:/Users/Lenovo/PycharmProjects/Lab5/cleaned_data/variant_6.csv')

# Перевірка на наявність пропущених значень
print(data['y'].isnull().sum())

# Визначення аномальних рядків
anomalies_condition = (data['age'] > 90) | (data['balance'] < 0) | (data['duration'] > 500)

# Видалення аномальних рядків
data_without_anomalies = data[~anomalies_condition]

# Збереження очищених даних
data_without_anomalies.to_csv('C:/Users/Lenovo/PycharmProjects/Lab5/cleaned_data/cleaned_variant_6.csv', index=False)

print(anomalies_condition.sum())

# Видалення рядків з пропущеними значеннями у колонці 'y'
data_labeled = data_without_anomalies.dropna(subset=['y'], axis=0)

# Видалення непотрібних колонок
data_cleaned = data_labeled.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])

# Заповнення пропущених значень
data_cleaned['age'] = data_cleaned['age'].fillna(data_cleaned['age'].median())
data_cleaned['duration'] = data_cleaned['duration'].fillna(data_cleaned['duration'].median())
data_cleaned['campaign'] = data_cleaned['campaign'].fillna(data_cleaned['campaign'].median())
data_cleaned['previous'] = data_cleaned['previous'].fillna(data_cleaned['previous'].median())

# Заповнення пропущених значень у категоріальних колонках
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'month', 'y']
for column in categorical_columns:
    data_cleaned[column] = data_cleaned[column].fillna('unknown')

# Збереження очищених даних
data_cleaned.to_csv('C:/Users/Lenovo/PycharmProjects/Lab5/cleaned_data/cleaned_data1.csv', index=False)

# Перетворення категоріальних ознак у числові
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)
data_encoded.info()

# Масштабування числових ознак
scaler = StandardScaler()
numerical_columns = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
data_encoded[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])

# Збереження кінцевого очищеного набору даних
data_encoded.to_csv('C:/Users/Lenovo/PycharmProjects/Lab5/cleaned_data/cleaned_data1.csv', index=False)
