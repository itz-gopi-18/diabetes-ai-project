from flask import Flask, render_template, request, session
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.secret_key = "diabetes_ai_secret"

# Load dataset
data = pd.read_csv("data/diet_data.csv")

X = data[["sugar_value"]]
y_type = data["type"]
y_risk = data["risk"]
y_diet = data["diet"]
y_exercise = data["exercise"]

type_model = RandomForestClassifier(n_estimators=100)
risk_model = RandomForestClassifier(n_estimators=100)
diet_model = RandomForestClassifier(n_estimators=100)
exercise_model = RandomForestClassifier(n_estimators=100)

type_model.fit(X, y_type)
risk_model.fit(X, y_risk)
diet_model.fit(X, y_diet)
exercise_model.fit(X, y_exercise)


@app.route("/", methods=["GET", "POST"])
def chat():

    if "history" not in session:
        session["history"] = []

    response = None

    if request.method == "POST":

        # Get sugar
        sugar = int(request.form["sugar"])

        # Save history
        session["history"].append(sugar)
        if len(session["history"]) > 7:
            session["history"].pop(0)

        # Get BMI inputs safely
        weight = request.form.get("weight", "").strip()
        height = request.form.get("height", "").strip()

        bmi = None
        bmi_status = None
        advice = None

        # Only calculate if BOTH provided
        if weight != "" and height != "":
            weight = float(weight)
            height = float(height)

            bmi = round(weight / (height * height), 2)

            if bmi < 18.5:
                bmi_status = "Underweight"
                advice = "Increase nutritious food intake"
            elif bmi < 25:
                bmi_status = "Normal"
                advice = "Maintain healthy lifestyle"
            elif bmi < 30:
                bmi_status = "Overweight"
                advice = "Exercise regularly and control diet"
            else:
                bmi_status = "Obese"
                advice = "Strict diet control and daily physical activity"

        response = {
            "type": type_model.predict([[sugar]])[0],
            "risk": risk_model.predict([[sugar]])[0],
            "diet": diet_model.predict([[sugar]])[0],
            "exercise": exercise_model.predict([[sugar]])[0],
            "history": session["history"],
            "bmi": bmi,
            "bmi_status": bmi_status,
            "advice": advice
        }

    return render_template("chat.html", response=response)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


