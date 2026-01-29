from pathlib import Path
import shutil
import uuid
import deeplabcut



def run_dlc_on_video(
    video_path: Path,
    config_path: Path,
    out_dlc_dir: Path,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    save_as_csv: bool = True,
) -> Path:


    video_path = Path(video_path).resolve()
    config_path = Path(config_path).resolve()
    out_dlc_dir = Path(out_dlc_dir).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"DLC config not found: {config_path}")

    out_dlc_dir.mkdir(parents=True, exist_ok=True)

    # --- Create isolated work directory (pipeline-safe) ---
    work_dir = out_dlc_dir / f"work_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir()

    try:
        # DLC prefers files inside its working directory
        work_video = work_dir / video_path.name
        shutil.copy2(video_path, work_video)

        # --- Run DLC ---
        deeplabcut.analyze_videos(
            str(config_path),
            [str(work_video)],
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            save_as_csv=save_as_csv,
        )

        # --- Collect outputs ---
        csv_files = list(work_dir.glob("*.csv"))
        h5_files = list(work_dir.glob("*.h5"))
        meta_files = list(work_dir.glob("*.pickle"))

        if not csv_files:
            raise RuntimeError(
                "DLC did not produce a CSV file. "
                "Check shuffle/trainingsetindex and model paths."
            )

        # Move artifacts to final output directory
        for p in csv_files + h5_files + meta_files:
            shutil.move(str(p), out_dlc_dir / p.name)

        return out_dlc_dir / csv_files[0].name

    finally:
        # Always clean up temp dir
        shutil.rmtree(work_dir, ignore_errors=True)