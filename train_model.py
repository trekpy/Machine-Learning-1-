# train_model.py
import pandas as pd
import os
import dill
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.utils import save_object

# Sample data
df = pd.read_csv("notebooks/data/stud.csv")  # make sure the dataset is there

# Features and target
X = df.drop("math_score", axis=1)
y = df["math_score"]

# Categorical and numerical features
cat_features = ['gender', 'race_ethinicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
num_features = ['reading_score', 'writing_score']

# Preprocessor pipeline
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Final model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save artifacts
os.makedirs("artifacts", exist_ok=True)
save_object("artifacts/preprocessor.pkl", preprocessor)
save_object("artifacts/model.pkl", model)

print("✅ Model and preprocessor saved to artifacts/")
