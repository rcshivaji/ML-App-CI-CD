import argparse
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
ACTIVE_DIR = MODELS_DIR / "active"
CANDIDATES_DIR = MODELS_DIR / "candidates"
ARCHIVE_DIR = MODELS_DIR / "archive"

ACTIVE_MODEL_PATH = ACTIVE_DIR / "model.pkl"


def promote_model(candidate_name: str):
    candidate_path = CANDIDATES_DIR / candidate_name

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate model not found: {candidate_path}"
        )

    ACTIVE_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Archive existing active model (if any)
    if ACTIVE_MODEL_PATH.exists():
        archived_name = f"model_archived_{timestamp}.pkl"
        archived_path = ARCHIVE_DIR / archived_name
        shutil.move(ACTIVE_MODEL_PATH, archived_path)
        print(f"📦 Archived old active model → {archived_path.name}")
    else:
        print("ℹ️ No existing active model found (first promotion)")

    # 2. Promote candidate → active/model.pkl
    shutil.copy(candidate_path, ACTIVE_MODEL_PATH)
    print(f"🚀 Promoted {candidate_name} → active/model.pkl")

    print("✅ Model promotion completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Promote a candidate model to active"
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Filename of model in models/candidates/"
    )

    args = parser.parse_args()
    promote_model(args.candidate)


if __name__ == "__main__":
    main()
