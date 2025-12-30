from flask import Flask
from app.inference import load_model

def create_app():
    app = Flask(__name__)

    app.config["MODEL"] = load_model()

    from app.routes import main
    app.register_blueprint(main)

    return app
