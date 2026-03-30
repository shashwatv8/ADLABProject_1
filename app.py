from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from model import train_model

app = Flask(__name__)

model, columns = train_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area"])

        if area < 1000:
            return render_template(
                "index.html",
                prediction_text=" Error: Area must be at least 1000 sq ft"
            )

        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        stories = int(request.form["stories"])
        parking = int(request.form["parking"])

        mainroad = request.form["mainroad"]
        guestroom = request.form["guestroom"]
        basement = request.form["basement"]
        hotwaterheating = request.form["hotwaterheating"]
        airconditioning = request.form["airconditioning"]
        prefarea = request.form["prefarea"]
        furnishingstatus = request.form["furnishingstatus"]

        input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

        input_df["area"] = area
        input_df["bedrooms"] = bedrooms
        input_df["bathrooms"] = bathrooms
        input_df["stories"] = stories
        input_df["parking"] = parking

        for col in input_df.columns:
            if col == f"mainroad_{mainroad}":
                input_df[col] = 1
            if col == f"guestroom_{guestroom}":
                input_df[col] = 1
            if col == f"basement_{basement}":
                input_df[col] = 1
            if col == f"hotwaterheating_{hotwaterheating}":
                input_df[col] = 1
            if col == f"airconditioning_{airconditioning}":
                input_df[col] = 1
            if col == f"prefarea_{prefarea}":
                input_df[col] = 1
            if col == f"furnishingstatus_{furnishingstatus}":
                input_df[col] = 1

        prediction = model.predict(input_df)[0]

        return render_template(
            "index.html",
            prediction_text=f"Estimated Price: ₹ {round(prediction, 2)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"❌ Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)