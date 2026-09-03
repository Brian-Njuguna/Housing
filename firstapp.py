import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
def fetch_data(url):
    data = pd.read_csv(url)
    return data
housing = fetch_data(url)
housing_num = housing.drop('ocean_proximity', axis =1)
housing_cat = housing["ocean_proximity"].copy
rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__ (self, add_bedrooms_per_room =True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household = X[:,rooms_ix] / X[:,households_ix]
        population_per_household = X[:,population_ix] / X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/ X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attradder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline,num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)
st.title("California Housing Dataset")
st.write("An interactive exploration of the California Housing dataset.")
st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", housing_num.shape[0])
with col2:
    st.metric("Columns", housing_num.shape[1])
with col3:
    st.metric("Missing Values", housing_num.isna().sum().sum())


col1, col2 = st.columns([2,1])

with col1:
    st.header("Data Preview")
    st.dataframe(housing)
with col2:
    st.subheader("Statistical Summary")
    housing_prepared_df = pd.DataFrame(housing_prepared)
    st.dataframe(housing.describe())

housing["income_cat"] = pd.cut(housing["median_income"],
bins = [0.,1.5,3.0,4.5,6.,np.inf],
labels = [1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

st.header("A plot showing prices and population in various districts of California")
fig = px.scatter(
    housing,
    x="longitude",
    y="latitude",
    size = "population",
    color="median_house_value",
    color_continuous_scale="jet",
    hover_data = [
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

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("California Housing Map")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("Dataset Information")
    st.write("This plot Shows:")
    st.write("- Location of Districts")
    st.write("- population represented by point size")
    st.write("- House value represented by color")
st.header("Does income influence House values in California ?")
min_income = st.sidebar.slider(
    "Minimum Income",
    0.0,
    15.0,
    5.0
)
filtered_housing = housing[
    housing["median_income"] >= min_income
]
col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(filtered_housing)
with col2:
    fig = px.scatter(
    filtered_housing,
    x="median_income",
    y="median_house_value"
)
st.plotly_chart(fig)