from pathlib import Path
import uuid

from app.pipeline.preprocessing import preprocess_image_to_video
from app.pipeline.dlc_inference import run_dlc_on_video
from app.pipeline.features import gen_features
from app.pipeline.label import create_labeled_video
from app.pipeline.scoring_04 import score_from_features

MODELS_DIR = Path("/models")
DATA_DIR = Path("/data")
SCORER = "DLC_Resnet50_pose_analysisJun27shuffle1_snapshot_680"


def run_pipeline_one(
    input_image: Path,
    job_dir: Path,
    dlc_config: Path,
    coeffs_json: Path | None = None,
):
    job_dir.mkdir(parents=True, exist_ok=True)

    dlc_dir = job_dir / "dlc"
    artifacts_dir = job_dir / "artifacts"
    dlc_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)

    video_path = preprocess_image_to_video(
        input_image=input_image,
        output_video=dlc_dir / "input.avi",
        target_width=960,
    )

    csv_path = run_dlc_on_video(
        video_path=video_path,
        config_path=dlc_config,
        out_dlc_dir=dlc_dir,
        shuffle=1,
    )

    features_path = gen_features(
        csv_path=csv_path,
        scorer=SCORER,
        out_path=artifacts_dir / "features.json",
    )

    score = score_from_features(
    features_json=features_path,
    coeffs_json=MODELS_DIR / "model_coeffs.json",
    xb_range_json=MODELS_DIR / "xb_range.json",
    )["score_0_100"]  

    labeled_video = create_labeled_video(
        config_path=dlc_config,
        video_path=video_path,
        shuffle=1,
    )

    return {
        "job_id": job_dir.name,
        "features": str(features_path),
        "labeled_video": str(labeled_video),
        "score": float(score),
    }