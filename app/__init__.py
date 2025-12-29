from flask import Flask
from app.inference import load_artifacts

def create_app():
    app = Flask(__name__)

    # Load ML artifacts ONCE
    model, scaler = load_artifacts()
    app.config["MODEL"] = model
    app.config["SCALER"] = scaler

    from app.routes import main
    app.register_blueprint(main)

    return app
