import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_model():
    # Load dataset
    df = pd.read_csv("Housing.csv")

    # Convert categorical columns
    df = pd.get_dummies(df, drop_first=True)

    # Split data
    X = df.drop("price", axis=1)
    y = df["price"]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, X.columns