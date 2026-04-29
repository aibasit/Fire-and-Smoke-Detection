import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import pygame
from ultralytics import YOLO


DEFAULT_MODEL = "optimized150.pt"
DEFAULT_ALERT_SOUND = "alert_sound.mp3"
WINDOW_NAME = "Fire and Smoke Detection"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fire and smoke YOLOv8 model on an image, video, folder, or webcam."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Input source. Use a file/folder path, URL, or webcam index like 0 or 1.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to the YOLO model weights file.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.5, help="Base confidence threshold.")
    parser.add_argument("--save", action="store_true", help="Save annotated outputs.")
    parser.add_argument("--show", action="store_true", help="Display the annotated result window.")
    parser.add_argument(
        "--fire-threshold",
        type=float,
        default=0.55,
        help="Confidence threshold used to count a fire detection.",
    )
    parser.add_argument(
        "--smoke-threshold",
        type=float,
        default=0.75,
        help="Confidence threshold used to count a smoke detection.",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=3.0,
        help="Minimum seconds between alert sounds or saved detection snapshots.",
    )
    return parser.parse_args()


def resolve_source(raw_source):
    return int(raw_source) if raw_source.isdigit() else raw_source


def validate_paths(model_path, source):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if isinstance(source, str):
        source_path = Path(source)
        if not source.startswith(("http://", "https://", "rtsp://")) and not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source}")


def init_audio(sound_path):
    sound_file = Path(sound_path)
    if not sound_file.exists():
        print(f"Alert sound file not found: {sound_path}")
        return False

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(str(sound_file))
        return True
    except Exception as exc:
        print(f"Audio disabled: could not initialize alert sound ({exc})")
        return False


def play_alert_sound(audio_enabled):
    if not audio_enabled:
        return

    try:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
    except Exception as exc:
        print(f"Audio playback failed: {exc}")


def get_session_dir():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    session_dir = Path("runs") / "live_test" / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def summarize_detection(results, fire_threshold, smoke_threshold):
    filtered_boxes = []
    labels = []

    for box in results[0].boxes:
        class_index = int(box.cls.item())
        confidence = float(box.conf.item())

        if class_index == 0 and confidence >= fire_threshold:
            filtered_boxes.append(box)
            labels.append("fire")
        elif class_index == 1 and confidence >= smoke_threshold:
            filtered_boxes.append(box)
            labels.append("smoke")

    return filtered_boxes, labels


def write_session_summary(summary_path, source, frames_processed, detection_counts, snapshots):
    lines = [
        "Fire and Smoke Detection Session Summary",
        f"Source: {source}",
        f"Frames processed: {frames_processed}",
        f"Fire detections: {detection_counts['fire']}",
        f"Smoke detections: {detection_counts['smoke']}",
        f"Saved detection snapshots: {len(snapshots)}",
        "",
        "Snapshots:",
    ]

    if snapshots:
        lines.extend(str(path) for path in snapshots)
    else:
        lines.append("None")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run_live_detection(model, source, args):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    session_dir = get_session_dir()
    audio_enabled = init_audio(DEFAULT_ALERT_SOUND)
    last_alert_time = 0.0
    frames_processed = 0
    detection_counts = Counter({"fire": 0, "smoke": 0})
    saved_snapshots = []

    print(f"Live detection started on source {source}")
    print(f"Results will be saved to: {session_dir}")
    print("Press q, d, or Esc to stop the test.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Video stream ended or a frame could not be read.")
                break

            frames_processed += 1
            results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            filtered_boxes, labels = summarize_detection(results, args.fire_threshold, args.smoke_threshold)
            results[0].boxes = filtered_boxes
            annotated_frame = results[0].plot()

            current_time = time.time()
            has_fire = "fire" in labels
            has_smoke = "smoke" in labels

            if has_fire:
                detection_counts["fire"] += 1
            if has_smoke:
                detection_counts["smoke"] += 1

            status_text = "No detection"
            if has_fire and has_smoke:
                status_text = "Fire and smoke detected"
            elif has_fire:
                status_text = "Fire detected"
            elif has_smoke:
                status_text = "Smoke detected"

            cv2.putText(
                annotated_frame,
                "Press q, d, or Esc to exit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                status_text,
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if status_text != "No detection" else (200, 200, 200),
                2,
            )

            should_alert = (has_fire or has_smoke) and (current_time - last_alert_time >= args.alert_cooldown)
            if should_alert:
                play_alert_sound(audio_enabled)
                last_alert_time = current_time

                if args.save:
                    label_text = "_".join(sorted(set(labels)))
                    snapshot_path = session_dir / f"{label_text}_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                    cv2.imwrite(str(snapshot_path), annotated_frame)
                    saved_snapshots.append(snapshot_path)
                    print(f"Saved detection snapshot: {snapshot_path}")

            if args.show:
                cv2.imshow(WINDOW_NAME, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("d"), 27):
                    print("Exit requested by user.")
                    break

    finally:
        capture.release()
        cv2.destroyAllWindows()
        if pygame.mixer.get_init():
            pygame.mixer.quit()

    summary_path = session_dir / "summary.txt"
    write_session_summary(summary_path, source, frames_processed, detection_counts, saved_snapshots)
    print(f"Session summary saved to: {summary_path}")


def run_batch_prediction(model, source, args):
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        show=args.show,
        save=args.save,
    )

    if results:
        print("Inference completed successfully.")
        if args.save:
            print(f"Saved annotated output to: {results[0].save_dir}")


def main():
    args = parse_args()
    source = resolve_source(args.source)
    validate_paths(args.model, source)

    model = YOLO(args.model)

    if isinstance(source, int):
        run_live_detection(model, source, args)
    else:
        run_batch_prediction(model, source, args)


if __name__ == "__main__":
    main()
