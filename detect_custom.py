#!/usr/bin/env python3
# detect_custom.py

import argparse
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a custom-trained YOLOv11 model"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best.pt",
        help="Path to your custom .pt weights file"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="File/dir/URL/glob, or '0' for webcam"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference image size (pixels)"
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="NMS IoU threshold"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g. 'cpu', 'cuda:0'. Auto‑detect if unset."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="inference_output",
        help="Directory to save inference images/videos"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Force real-time display of inference (overrides batch mode)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto‑select device if not provided
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Prepare output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(args.weights)

    # If webcam or --stream flag, run streaming inference and save to video
    if args.stream or args.source == "0":
        print(f"🔴 Streaming inference on {args.source} (press ESC or 'q' to quit)")
        # Open video source
        capture_source = int(args.source) if args.source == "0" else args.source
        cap = cv2.VideoCapture(capture_source)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = save_dir / "stream_output.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        # Run inference frame by frame
        for result in model(
            args.source,
            stream=True,
            imgsz=args.img_size,
            conf=args.conf_thres,
            iou=args.iou_thres,
            device=args.device
        ):
            annotated = result.plot()          # BGR ndarray with boxes & labels
            writer.write(annotated)           # save frame to output video
            cv2.imshow("Real-time Stream", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                print("Stopping stream…")
                break

        # Release resources
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Stream saved to {out_path}")

    else:
        # Batch inference (save outputs)
        print(f"▶️ Running batch inference on {args.source}")
        results = model(
            args.source,
            imgsz=args.img_size,
            conf=args.conf_thres,
            iou=args.iou_thres,
            device=args.device,
            save=True,
            save_dir=str(save_dir)
        )
        print(f"✅ Done. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
