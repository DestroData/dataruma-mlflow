import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from imblearn.pipeline import Pipeline
import os
import joblib


def training(X, y, model, params):

    model = model(**params)
    model.fit(X, y)

    return model

