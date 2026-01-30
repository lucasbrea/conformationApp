from app.pipeline.preprocessing import preprocess_image_to_video
from app.pipeline.dlc_inference import run_dlc_on_video
from app.pipeline.features import gen_features
from app.pipeline.label import create_labeled_video
from pathlib import Path
from datetime import datetime
import uuid

MODELS_DIR = Path("/models")
DATA_DIR = Path("/data")
scorer = "DLC_Resnet50_pose_analysisJun27shuffle1_snapshot_680"
#job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
job_id = uuid.uuid4().hex[:12]

config_path = MODELS_DIR / "dlc_project" / "config_inference.yaml"
images_dir = DATA_DIR / "inputs"
base_outdir = DATA_DIR / "jobs"
outdir = base_outdir / job_id

artifacts_dir = outdir / "artifacts"

video_path = preprocess_image_to_video(
    input_image=images_dir/"test_02.jpg",
    output_video=outdir/"dlc" / "input.avi",
    target_width=960,
)

csv_path = run_dlc_on_video(
    video_path=outdir/"dlc" / "input.avi",
    config_path=config_path,
    out_dlc_dir=outdir / "dlc",
    shuffle=1,
)

features_path = gen_features(
    csv_path=csv_path,
    scorer=scorer,   
    out_path=artifacts_dir / "features.json"
)

create_labeled_video(
    config_path=config_path,
    video_path=outdir/"dlc" / "input.avi",
    shuffle=1,

)