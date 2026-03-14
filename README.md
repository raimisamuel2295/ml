# 🏠 Housing Dataset Machine Learning Project

## 📌 Project Overview

This project demonstrates a **machine learning workflow for analyzing housing data and building a predictive model** using Python and Scikit-learn.

The goal is to train a model that learns patterns from housing features and predicts a target variable from the dataset. The notebook walks through data loading, preprocessing, training a Support Vector Machine model, and evaluating the model’s performance.

This project is useful for practicing:

* Data preprocessing
* Feature engineering
* Model training
* Hyperparameter tuning
* Model evaluation

---

# 📊 Dataset

The project uses a dataset called **`housing.csv`**.

The dataset contains housing-related attributes such as:

* Geographic features
* Housing characteristics
* Location proximity to the ocean
* Economic indicators

One categorical feature called **`ocean_proximity`** is converted into numeric values so that machine learning algorithms can process it.

Example mapping used in the project:

```
'<1H OCEAN' → 0  
'INLAND' → 1  
'NEAR OCEAN' → 2  
'NEAR BAY' → 3  
'ISLAND' → 4
```

---

# ⚙️ Technologies Used

The project is implemented using the following tools:

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **SciPy**
* **Jupyter Notebook**

---

# 🧠 Machine Learning Workflow

## 1️⃣ Data Loading

The dataset is loaded using **Pandas**:

```
df = pd.read_csv("housing.csv")
```

This allows exploration and manipulation of the dataset.

---

## 2️⃣ Data Preprocessing

The categorical column **`ocean_proximity`** is converted to numeric values using a dictionary mapping.

This step ensures that machine learning models can work with the data.

---

## 3️⃣ Feature Selection

The dataset is split into:

* **Features (X)** – input variables
* **Target (y)** – variable to predict

```
X = df.drop('median_income', axis=1)
y = df['median_income']
```

---

## 4️⃣ Train-Test Split

The data is divided into training and testing sets to evaluate model performance.

```
train_test_split(test_size=0.2, random_state=42)
```

* **80%** training data
* **20%** testing data

---

## 5️⃣ Model Training

A **Support Vector Regression (SVR)** model is used for prediction.

```
svm_reg = SVR()
svm_reg.fit(X_train, y_train)
```

SVR is useful for regression tasks where the relationship between variables may be complex.

---

## 6️⃣ Hyperparameter Tuning

### Grid Search

GridSearchCV is used to find the best combination of model parameters.

Parameters tested include:

* `kernel`
* `C`
* `gamma`

Example:

```
param_grid = [
 {'kernel': ['linear'], 'C': [10, 30, 100]},
 {'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [0.01, 0.1, 1]}
]
```

---

### Randomized Search

RandomizedSearchCV is also used to explore a wider range of parameter values using probability distributions.

This approach can be faster than grid search.

---

## 7️⃣ Machine Learning Pipeline

A **pipeline** is created to automate preprocessing and model training.

```
Pipeline([
 ("scaler", StandardScaler()),
 ("svm", SVR())
])
```

Steps in the pipeline:

1. **StandardScaler** – normalizes the features
2. **SVR model** – performs regression

---

## 8️⃣ Model Evaluation

The model is evaluated using **Mean Squared Error (MSE)**.

```
mean_squared_error(y_test, predictions)
```

MSE measures how close predicted values are to the actual values.

Lower MSE indicates better performance.

---

# 📈 Project Workflow

```
Load Dataset
      ↓
Data Preprocessing
      ↓
Feature Selection
      ↓
Train/Test Split
      ↓
Model Training (SVR)
      ↓
Hyperparameter Tuning
      ↓
Pipeline Creation
      ↓
Model Evaluation
```

---

# 🚀 How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/raimisamuel2295/housing-ml-project.git
```

### 2. Navigate to the project directory

```
cd housing-ml-project
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the notebook

```
jupyter notebook
```

Open the notebook file and run the cells step-by-step.

---

# 📂 Project Structure

```
housing-ml-project
│
├── housing.csv
├── housing_model.ipynb
├── README.md
└── requirements.txt
```

---

# 🔮 Future Improvements

Possible improvements for the project include:

* Adding **more feature engineering**
* Trying **different regression models**
* Implementing **cross-validation**
* Visualizing **feature importance**
* Deploying the model with **FastAPI or Flask**

---

# 👨‍💻 Author

This project was created as part of a **machine learning practice project** to learn regression modeling and model optimization using Scikit-learn.
