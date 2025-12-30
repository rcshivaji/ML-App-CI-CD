from flask import Blueprint, render_template, request, current_app
from app.inference import predict_survival
import pandas as pd
from pathlib import Path

main = Blueprint("main", __name__)

@main.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_data = {
            "Name": request.form["Name"],
            "Pclass": int(request.form["Pclass"]),
            "Sex": request.form["Sex"],
            "Age": float(request.form["Age"]),
            "Fare": float(request.form["Fare"]),
        }

        model = current_app.config["MODEL"]
        prediction, probability = predict_survival(
            input_data,
            model
        )

        return render_template(
            "result.html",
            name= input_data["Name"],
            prediction=prediction,
            probability=round(probability * 100, 2)
        )

    return render_template("index.html")

@main.route("/models", methods=["GET"])
def model_performance():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    metrics_path = PROJECT_ROOT / "reports" / "model_metrics.csv"

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)

        # Convert date for proper sorting
        df["trained_on"] = pd.to_datetime(df["trained_on"], format="%d/%m/%Y")

        # Sort by most recent
        df = df.sort_values(by="trained_on", ascending=False)

        # Convert back to string for display
        df["trained_on"] = df["trained_on"].dt.strftime("%d/%m/%Y")

        records = df.to_dict(orient="records")
    else:
        records = []

    return render_template(
        "model_performance.html",
        records=records
    )