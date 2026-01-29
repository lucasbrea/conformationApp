from preprocessing_00 import preprocess_image_to_video
from inference_01 import run_dlc_on_video
from features_02 import gen_features
from labeling_03 import create_labeled_video
from pathlib import Path
from datetime import datetime
import uuid

config_path=Path("W:\ComputerVision\Web App\DLC\pose_analysis-tr-2025-06-27\config_inference.yaml")
scorer = "DLC_Resnet50_pose_analysisJun27shuffle1_snapshot_680"
#job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
job_id= "05"

images_dir = Path("W:\ComputerVision\Web App\Data\Inputs")
base_outdir = Path("W:\ComputerVision\Web App\Data\Outputs")
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