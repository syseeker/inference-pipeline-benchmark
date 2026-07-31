import os
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "source")
CV_DIR = os.path.join(BASE_DIR, "cv")
VLM_DIR = os.path.join(BASE_DIR, "vlm")


def resize_images():
    scene_path = os.path.join(SOURCE_DIR, "scene.jpg")
    doc_path = os.path.join(SOURCE_DIR, "document.png")

    scene = Image.open(scene_path)
    doc = Image.open(doc_path)

    cv_sizes = {
        "sample_224x224.jpg": (224, 224),
        "sample_320x320.jpg": (320, 320),
        "sample_640x640.jpg": (640, 640),
        "sample_1280x1280.jpg": (1280, 1280),
    }

    for filename, size in cv_sizes.items():
        out_path = os.path.join(CV_DIR, filename)
        resized = scene.resize(size, Image.LANCZOS)
        resized.save(out_path, quality=95)
        print(f"  Created {filename} ({size[0]}x{size[1]})")

    doc_resized = doc.resize((1920, 1080), Image.LANCZOS)
    doc_resized.save(os.path.join(CV_DIR, "sample_document.png"))
    print(f"  Created sample_document.png (1920x1080)")


def transcode_video(source_path, output_path, width, height, duration_s, fps):
    """Transcode source video to target resolution, duration, and fps."""
    cap = cv2.VideoCapture(source_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_target_frames = int(duration_s * fps)
    written = 0

    for i in range(total_target_frames):
        t = i / fps
        src_frame_idx = int(t * src_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        writer.write(resized)
        written += 1

    writer.release()
    cap.release()
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Created {os.path.basename(output_path)} ({width}x{height}, {duration_s}s, {fps}fps, {written} frames, {size_kb:.0f} KB)")


def main():
    os.makedirs(CV_DIR, exist_ok=True)
    os.makedirs(VLM_DIR, exist_ok=True)

    print("Resizing images for CV models...")
    resize_images()

    print("\nTranscoding video for VLM...")
    video_source = os.path.join(SOURCE_DIR, "test_clip.mp4")
    if not os.path.exists(video_source):
        print(f"  ERROR: Source video not found at {video_source}")
        print("  Download a video to this path first (e.g., from Pexels)")
        return

    transcode_video(
        video_source, os.path.join(VLM_DIR, "clip_3s_224.mp4"),
        width=224, height=224, duration_s=3, fps=1
    )
    transcode_video(
        video_source, os.path.join(VLM_DIR, "clip_10s_720p.mp4"),
        width=1280, height=720, duration_s=10, fps=4
    )

    print("\nDone. Test data ready:")
    for root, dirs, files in os.walk(BASE_DIR):
        for f in sorted(files):
            if f == "prepare_data.py":
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, BASE_DIR)
            size_kb = os.path.getsize(path) / 1024
            print(f"  {rel} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()