from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")


def get_hdi_category(hdi):
    if hdi < 0.550:
        return "Low Human Development"
    elif hdi < 0.700:
        return "Medium Human Development"
    elif hdi < 0.800:
        return "High Human Development"
    else:
        return "Very High Human Development"


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    life = float(request.form["life"])
    expected = float(request.form["expected"])
    mean = float(request.form["mean"])
    income = float(request.form["income"])

    result = model.predict([
        [life, expected, mean, income]
    ])

    prediction = round(float(result[0]), 3)

    # Keep display range between 0 and 1
    prediction = max(0, min(prediction, 1))

    category = get_hdi_category(prediction)

    percentage = round(prediction * 100, 1)

    return render_template(
        "result.html",
        prediction=prediction,
        percentage=percentage,
        category=category,
        life=life,
        expected=expected,
        mean=mean,
        income=income
    )


if __name__ == "__main__":
    app.run(debug=True)