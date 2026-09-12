# Housing
# California Housing Price Prediction App

##  Project Overview

This project is an interactive **Streamlit web application** that predicts California housing prices using a machine learning model trained on the California Housing dataset from *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron.

The application allows users to:

* Explore the California Housing dataset
* Visualize housing data on an interactive map
* Analyze feature distributions
* Input housing characteristics
* Generate house value predictions using a trained machine learning model

The project demonstrates a complete end-to-end machine learning workflow, from data preprocessing and feature engineering to model deployment with Streamlit.

---

##  Live Features

The application includes:

### Dataset Exploration

* Dataset preview
* Summary statistics
* Feature descriptions
* Missing value inspection

### Interactive Visualizations

* California housing map
* Median house value distribution
* Median income analysis
* Population trends
* Geographic housing patterns

### House Price Prediction

Users can enter:

* Longitude
* Latitude
* Housing Median Age
* Total Rooms
* Total Bedrooms
* Population
* Households
* Median Income
* Ocean Proximity

The model then predicts the estimated median house value.

---

## Dataset

The California Housing dataset contains demographic and housing information collected from California census districts.

### Features

| Feature            | Description                 |
| ------------------ | --------------------------- |
| longitude          | Geographic longitude        |
| latitude           | Geographic latitude         |
| housing_median_age | Median age of houses        |
| total_rooms        | Total number of rooms       |
| total_bedrooms     | Total bedrooms              |
| population         | Population in block group   |
| households         | Number of households        |
| median_income      | Median income of households |
| ocean_proximity    | Distance from the ocean     |

### Target Variable

```text
median_house_value
```

---

## 🛠 Machine Learning Pipeline

The project follows an end-to-end machine learning workflow.

### Data Preparation

The preprocessing pipeline includes:

* Missing value imputation
* Feature engineering
* Numerical scaling
* One-hot encoding of categorical variables
* Pipeline automation using Scikit-Learn

### Engineered Features

Additional features include:

* Rooms per household
* Bedrooms per room
* Population per household

These engineered features improve the model's predictive performance.

---

##  Model Selection

Several models were evaluated during experimentation.

The final deployed model is:

```python
RandomForestRegressor()
```

### Hyperparameter Tuning

Grid Search Cross Validation was used to identify the best-performing configuration.

### Best Parameters

| Parameter    | Value |
| ------------ | ----: |
| max_features |     6 |
| n_estimators |    30 |

Best model:

```python
RandomForestRegressor(
    max_features=6,
    n_estimators=30,
    random_state=42
)
```

---

##  Model Performance

The final model achieved:

| Metric     |      Value |
| ---------- | ---------: |
| Final RMSE | 48,035.545 |

### 95% Confidence Interval

The model's expected prediction error falls within:

```text
46,531.531 – 50,586
```

This means that, with 95% confidence, the true RMSE of the model is expected to lie within this range.

### Interpretation

An RMSE of approximately:

```text
$48,036
```

indicates that predictions are typically within about **$48,000** of the actual house value on average.

---

## Feature Importance

The Random Forest model identified the following features as the most influential:

| Feature                  | Importance |
| ------------------------ | ---------: |
| median_income            |    Highest |
| INLAND                   |       High |
| population_per_household |       High |
| longitude                |   Moderate |
| latitude                 |   Moderate |
| rooms_per_household      |   Moderate |
| bedrooms_per_room        |   Moderate |
| housing_median_age       |      Lower |

The most important predictor was:

```text
median_income
```

which aligns with expectations since household income strongly influences housing prices.

---

## Streamlit Deployment

The application was built using:

```text
Streamlit
```

Key components include:

* Sidebar navigation
* Interactive widgets
* Plotly visualizations
* Dynamic prediction forms
* Model loading with Joblib

---

##  Project Structure

```text
Housing_app/
│
├── firstapp.py
├── house_prices.ipynb
├── housing_model.pkl
├── full_pipeline.pkl
├── model_rmse.pkl
├── requirements.txt
├── README.md
└── housing.csv
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Plotly
* Matplotlib
* SciPy
* Joblib

---

##  Skills Demonstrated

This project demonstrates:

* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis
* Pipeline Construction
* Hyperparameter Tuning
* Model Evaluation
* Confidence Interval Estimation
* Random Forest Regression
* Interactive Dashboard Development
* Model Deployment with Streamlit

---

## Future Improvements

Potential enhancements include:

* Model comparison dashboard
* Additional regression algorithms
* SHAP explainability visualizations
* User-uploaded datasets
* Cloud deployment
* Real-time prediction analytics
* Improved geographic visualizations

---

## 👤 Author

**Brian Njuguna**

Business Computing Graduate | Data Science & Machine Learning Enthusiast

GitHub: https://github.com/Brian-Njuguna

site: https://housinggit-arzdntqqvuahaiolqywuaj.streamlit.app/
