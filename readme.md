# Machine Learning Project: Customer Response Prediction

This project focuses on building a machine learning pipeline to predict customer responses to marketing campaigns using the `cleaned_data1.csv` dataset. The pipeline includes data preprocessing, model training, and predictions, with a well-organized project structure for reproducibility.

---

## **Project Structure**

The project is organized as follows:
1. **`src/`**  
   Contains the Python scripts for data splitting, model training, and prediction:
   - `data_split.py`: Splits the dataset into training and prediction datasets.
   - `model_training.py`: Trains machine learning models on the training data.
   - `model_prediction.py`: Makes predictions using the trained models.

2. **`cleaned_data/`**  
   Contains the processed dataset and its splits:
   - `cleaned_data1.csv`: Preprocessed dataset.
   - `train.csv`: 90% of the data used for training.
   - `new_input.csv`: 10% of the data used for predictions.

3. **`models/`**  
   Contains the trained machine learning models:
   - `random_forest_model.pkl`: Random Forest model.
   - `logistic_regression_model.pkl`: Logistic Regression model.
   - `xgboost_model.pkl`: XGBoost model.

4. **`results/`**  
   Contains the predictions:
   - `predictions.csv`: Predictions made on `new_input.csv`.

5. **`README.md`**  
   Documentation for the project.

6. **`requirements.txt`**  
   List of dependencies required for the project.

---

## **Getting Started**

### **Prerequisites**

Ensure you have Python 3.8+ installed and the following libraries available. Install them using `pip`:




# Machine Learning Project: Customer Response Prediction

## **Dependencies**

The following libraries are used in this project:
- **`pandas`**: for data manipulation.
- **`scikit-learn`**: for machine learning models and evaluation.
- **`xgboost`**: for gradient boosting.
- **`imbalanced-learn`**: for handling imbalanced datasets.
- **`joblib`**: for saving and loading models.

---

## **Steps to Reproduce**

### **1. Data Splitting**
Run `data_split.py` to split the dataset into `train.csv` (90%) and `new_input.csv` (10%).

```bash
python src/data_split.py
```

---

### **2. Model Training**
Run `model_training.py` to train machine learning models on the `train.csv` dataset.

```bash
python src/model_training.py
```

The trained models are saved in the `models/` directory.

---

### **3. Predictions**
Run `model_prediction.py` to make predictions on `new_input.csv` using the trained models.

```bash
python src/model_prediction.py
```

The predictions are saved in the `results/` directory as `predictions.csv`.

---

## **Evaluation Metrics**

The project evaluates models using:
- **Accuracy**
- **F1-score**
- **ROC-AUC**

Performance is printed during execution and stored in the output files.

---

## **Conclusion**

This project demonstrates the complete pipeline for building a machine learning solution, including:
1. **Data preprocessing**, anomaly handling, and encoding.
2. **Training and evaluating multiple models** (Logistic Regression, Random Forest, XGBoost).
3. **Handling imbalanced datasets** using SMOTE.
4. **Hyperparameter tuning** using RandomizedSearchCV.
5. **Generating predictions** for unseen data.

---

### **Key Insights**
- **Random Forest** and **XGBoost** models performed better than Logistic Regression in terms of F1-score and ROC-AUC.
- Imbalanced datasets required the use of **SMOTE** to improve model performance.
- **Feature importance analysis** showed that `duration` and `balance` were the most critical features.

---

### **Challenges**
- Handling missing data and anomalies required domain understanding.
- Hyperparameter tuning was computationally intensive but significantly improved model performance.

---

### **Learnings**
- Enhanced understanding of **data preprocessing techniques**.
- Learned to use **SMOTE** for imbalanced datasets.
- Gained experience with **model evaluation** and **hyperparameter tuning**.
- Developed skills in structuring machine learning projects for **scalability and reproducibility**.
