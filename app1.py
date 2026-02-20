from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load("model.joblib")
preprocessor = joblib.load("preprocessor.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        input_data = pd.DataFrame({
            "Age": [int(request.form["Age"])],
            "TypeofContact": [request.form["TypeofContact"]],
            "CityTier": [int(request.form["CityTier"])],
            "DurationOfPitch": [int(request.form["DurationOfPitch"])],
            "Occupation": [request.form["Occupation"]],
            "Gender": [request.form["Gender"]],
            "NumberOfFollowups": [int(request.form["NumberOfFollowups"])],
            "ProductPitched": [request.form["ProductPitched"]],
            "PreferredPropertyStar": [int(request.form["PreferredPropertyStar"])],
            "MaritalStatus": [request.form["MaritalStatus"]],
            "NumberOfTrips": [int(request.form["NumberOfTrips"])],
            "Passport": [int(request.form["Passport"])],
            "PitchSatisfactionScore": [int(request.form["PitchSatisfactionScore"])],
            "OwnCar": [int(request.form["OwnCar"])],
            "Designation": [request.form["Designation"]],
            "MonthlyIncome": [float(request.form["MonthlyIncome"])],
            "TotalVisiting": [int(request.form["TotalVisiting"])]
        })

        X = preprocessor.transform(input_data)
        prediction = int(model.predict(X)[0])

        if hasattr(model, "predict_proba"):
            probability = round(model.predict_proba(X)[0][1], 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)
