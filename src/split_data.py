import pandas as pd
from sklearn.model_selection import train_test_split

# Завантаження очищених даних
data = pd.read_csv('C:/Users/Lenovo/PycharmProjects/Lab5/cleaned_data/cleaned_data1.csv')

# Розділення даних у співвідношенні 90:10
train_data, new_input_data = train_test_split(data, test_size=0.1, random_state=42)

# Збереження поділених даних у окремі файли
train_data.to_csv('C:/Users/Lenovo/PycharmProjects/Lab5/cleaned_data/train.csv', index=False)
new_input_data.to_csv('C:/Users/Lenovo/PycharmProjects/Lab5/cleaned_data/new_input.csv', index=False)

print("Дані успішно розділені на train та new_input!")
