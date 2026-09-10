import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="California Housing Analysis",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "housing_model.pkl"
PIPELINE_PATH = BASE_DIR / "full_pipeline.pkl"
RMSE_PATH = BASE_DIR / "model_rmse.pkl"


# ============================================================
# LOAD DATA
# ============================================================

URL = (
    "https://raw.githubusercontent.com/ageron/"
    "handson-ml2/master/datasets/housing/housing.csv"
)


@st.cache_data
def fetch_data(url):
    """Load the California Housing dataset."""
    return pd.read_csv(url)


housing = fetch_data(URL)


# ============================================================
# DATA PREPARATION
# ============================================================

housing_num = housing.drop("ocean_proximity", axis=1)


# ============================================================
# CUSTOM FEATURE ENGINEERING
# ============================================================

rooms_ix = 3
bedrooms_ix = 4
population_ix = 5
households_ix = 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        rooms_per_household = (
            X[:, rooms_ix] / X[:, households_ix]
        )

        population_per_household = (
            X[:, population_ix] / X[:, households_ix]
        )

        if self.add_bedrooms_per_room:

            bedrooms_per_room = (
                X[:, bedrooms_ix] / X[:, rooms_ix]
            )

            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room
            ]

        return np.c_[
            X,
            rooms_per_household,
            population_per_household
        ]


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]


num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attradder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler())
])


full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])


# Fit pipeline for dataset exploration
housing_prepared = full_pipeline.fit_transform(housing)


# ============================================================
# STRATIFIED TRAIN/TEST SPLIT
# ============================================================

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)


split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)


for train_index, test_index in split.split(
    housing,
    housing["income_cat"]
):

    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# Remove income category after stratification
strat_train_set = strat_train_set.drop(
    "income_cat",
    axis=1
)

strat_test_set = strat_test_set.drop(
    "income_cat",
    axis=1
)


# Use training set for exploration
housing = strat_train_set.copy()


st.title("California Housing Dataset")

st.write(
    "An interactive exploration and machine learning "
    "application based on the California Housing dataset."
)



st.header("Dataset Overview")


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Rows",
        housing_num.shape[0]
    )


with col2:
    st.metric(
        "Columns",
        housing_num.shape[1]
    )


with col3:
    st.metric(
        "Missing Values",
        housing_num.isna().sum().sum()
    )



col1, col2 = st.columns([2, 1])


with col1:

    st.subheader("Data Preview")

    st.dataframe(
        housing.head(20),
        use_container_width=True
    )


with col2:

    st.subheader("Statistical Summary")

    st.dataframe(
        housing.describe(),
        use_container_width=True
    )



st.header(
    "California Housing Prices by Geographic Location"
)


fig_map = px.scatter(
    housing,
    x="longitude",
    y="latitude",
    size="population",
    color="median_house_value",
    color_continuous_scale="jet",
    hover_data=[
        "median_income",
        "housing_median_age",
        "total_rooms",
        "population",
        "households",
        "median_house_value"
    ],
    opacity=0.4,
    title="California Housing Prices"
)


col1, col2 = st.columns([2, 1])


with col1:

    st.subheader("California Housing Map")

    st.plotly_chart(
        fig_map,
        use_container_width=True,
        key="california_housing_map"
    )


with col2:

    st.subheader("Dataset Information")

    st.write("This visualization shows:")

    st.write(
        "-  Location of housing districts"
    )

    st.write(
        "-  Population represented by point size"
    )

    st.write(
        "-  House value represented by color"
    )




st.header(
    "Does Income Influence House Values in California?"
)


min_income = st.sidebar.slider(
    "Minimum Income",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)


filtered_housing = housing[
    housing["median_income"] >= min_income
]


col1, col2 = st.columns([1, 2])


with col1:

    st.subheader("Filtered Data")

    st.dataframe(
        filtered_housing.head(20),
        use_container_width=True
    )


with col2:

    fig_income = px.scatter(
        filtered_housing,
        x="median_income",
        y="median_house_value",
        labels={
            "median_income": "Median Income ($10,000s)",
            "median_house_value": "Median House Value ($)"
        },
        title="Median Income vs. House Value"
    )

    st.plotly_chart(
        fig_income,
        use_container_width=True,
        key="income_house_value_chart"
    )




st.header("Top 8 Feature Importances")


importance_df = pd.DataFrame({

    "Feature": [
        "median_income",
        "INLAND",
        "pop_per_hhold",
        "longitude",
        "latitude",
        "rooms_per_hhold",
        "bedrooms_per_room",
        "housing_median_age"
    ],

    "Importance": [
        0.3778,
        0.1631,
        0.1128,
        0.0633,
        0.0622,
        0.0547,
        0.0524,
        0.0435
    ]
})


fig_importance = px.bar(
    importance_df.sort_values("Importance"),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 8 Features Used by the Model",
    labels={
        "Importance": "Feature Importance",
        "Feature": "Feature"
    }
)


st.plotly_chart(
    fig_importance,
    use_container_width=True,
    key="feature_importance_chart"
)



model = joblib.load(MODEL_PATH)

prediction_pipeline = joblib.load(
    PIPELINE_PATH
)

rmse_interval = joblib.load(
    RMSE_PATH
)




st.header(" Predict House Value")


st.write(
    "Enter the characteristics of a house to estimate "
    "its median house value."
)


col1, col2 = st.columns(2)


with col1:

    longitude = st.number_input(
        "Longitude",
        value=-122.23
    )

    latitude = st.number_input(
        "Latitude",
        value=37.88
    )

    housing_median_age = st.number_input(
        "Housing Median Age",
        value=30
    )

    total_rooms = st.number_input(
        "Total Rooms",
        value=2000
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        value=400
    )


with col2:

    population = st.number_input(
        "Population",
        value=1000
    )

    households = st.number_input(
        "Households",
        value=400
    )

    median_income = st.number_input(
        "Median Income",
        value=3.5
    )

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        [
            "<1H OCEAN",
            "INLAND",
            "NEAR OCEAN",
            "NEAR BAY",
            "ISLAND"
        ]
    )




input_data = pd.DataFrame({

    "longitude": [longitude],

    "latitude": [latitude],

    "housing_median_age": [
        housing_median_age
    ],

    "total_rooms": [
        total_rooms
    ],

    "total_bedrooms": [
        total_bedrooms
    ],

    "population": [
        population
    ],

    "households": [
        households
    ],

    "median_income": [
        median_income
    ],

    "ocean_proximity": [
        ocean_proximity
    ]
})




prepared_input_arr = prediction_pipeline.transform(
    input_data
)


# These are the same transformed feature positions
# used when training the saved model.
selected_indices = [
    0, 1, 2, 3, 4, 5, 6, 7
]


prepared_input = prepared_input_arr[
    :, selected_indices
]


if st.button(
    "Predict House Value",
    type="primary"
):

    prediction = model.predict(
        prepared_input
    )

    predicted_value = prediction[0]

    st.subheader(
        "Predicted House Value"
    )

    st.success(
        f"${predicted_value:,.0f}"
    )


st.header("Model Performance")


lower = rmse_interval[0]
upper = rmse_interval[1]


col1, col2 = st.columns(2)


with col1:

    st.metric(
        label="Lower RMSE Bound",
        value=f"${lower:,.0f}"
    )


with col2:

    st.metric(
        label="Upper RMSE Bound",
        value=f"${upper:,.0f}"
    )


st.info(
    "The interval above represents the 95% confidence "
    "interval estimated for the model's RMSE."
)

