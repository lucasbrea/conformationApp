from pathlib import Path
import cv2
import math

FPS = 1
CODEC = CODEC = cv2.VideoWriter_fourcc(*"MJPG")

def _resize_keep_ratio_even_h(img, width: int):
    h, w = img.shape[:2]
    scale = width / max(w, 1)
    new_h = math.ceil(h * scale)
    if new_h % 2:
        new_h += 1
    return cv2.resize(img, (width, new_h), cv2.INTER_AREA)


def _write_mp4_single_frame(img, out_path, fps, codec):
    h, w = img.shape[:2]
    vw = cv2.VideoWriter(str(out_path), codec, fps, (w, h))
    if not vw.isOpened():
        return False, "video writer failed to open"
    vw.write(img); vw.release()
    return True, ""


def preprocess_image_to_video(
    input_image: Path,
    output_video: Path,
    target_width: int = 960,
    fps: int = 25,
):
    """
    Converts a single image into a single-frame MP4 for DLC.

    Parameters
    ----------
    input_image : Path
        Path to the input image (jpg/png/etc.)
    output_video : Path
        Path where the MP4 should be written.
    target_width : int
        Resize width (keeps aspect ratio, forces even height).
    fps : int
        FPS metadata for the MP4 (irrelevant for DLC, but required).

    Returns
    -------
    Path
        Path to the written MP4 file.
    """

    img = cv2.imread(str(input_image), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {input_image}")

    img_resized = _resize_keep_ratio_even_h(img, target_width)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    _write_mp4_single_frame(img_resized, output_video, FPS,CODEC)

    return output_video

