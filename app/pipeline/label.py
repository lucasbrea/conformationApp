from pathlib import Path


def create_labeled_video(
    video_path: Path,
    config_path: Path,
    shuffle,
    draw_skeleton: bool = True,
) -> None:
    """
    Create a labeled video (mp4/avi depending on DLC + codec support) next to `video_path`.

    DLC writes the labeled video in the same folder as the input video.
    This function returns None; the output is the labeled video file on disk.
    """
    import deeplabcut  # lazy import

    video_path = Path(video_path).resolve()
    config_path = Path(config_path).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    deeplabcut.create_labeled_video(
        str(config_path),
        [str(video_path)],
        shuffle=shuffle,
        draw_skeleton=draw_skeleton,
    )