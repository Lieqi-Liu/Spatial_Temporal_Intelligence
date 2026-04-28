#!/usr/bin/env python3
"""
Generate 2D bounding-box annotations with SAM 3 (facebook/sam3).

This script runs text-prompt segmentation with SAM 3 and converts the
predicted instances into 2D boxes for each image.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from huggingface_hub import login
from PIL import Image, ImageDraw
from transformers import Sam3Model, Sam3Processor

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_INPUT_ROOT = Path("/home/lieqiliu/AutoDriving/waymo_test2/images")
DEFAULT_IMAGE_GLOB = "**/CAM_FRONT/*.jpg"
DEFAULT_PROMPTS = (
    "parked car,moving car,pedestrian,truck,bus,bicycle,motorcycle,"
    "traffic cone,traffic barrier,traffic sign."
)
DEFAULT_OUTPUT_JSON = Path(
    "/home/lieqiliu/AutoDriving/waymo_test2/annotations/sam3_bbox_annotations.json"
)
DEFAULT_VIZ_DIR = Path("/home/lieqiliu/AutoDriving/waymo_test2/annotations/sam3_viz")
DEFAULT_SCORE_THRESHOLD = 0.35
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_MIN_BOX_AREA = 500.0
DEFAULT_TOP_K = 10


@dataclass
class BoxAnnotation:
    label: str
    score: float
    bbox_xyxy: list[float]
    bbox_xywh: list[float]
    importance_score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate Waymo test2 images with SAM3 and export 2D boxes."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root folder containing images.",
    )
    parser.add_argument(
        "--image-glob",
        type=str,
        default=DEFAULT_IMAGE_GLOB,
        help="Glob pattern (relative to input root) to collect images.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path for predicted boxes.",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=DEFAULT_VIZ_DIR,
        help="Directory to save rendered bbox visualization images.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=DEFAULT_PROMPTS,
        help="Comma-separated text prompts.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If > 0, only process N matched images.",
    )
    parser.add_argument(
        "--sample-random",
        action="store_true",
        help="If set with --max-images > 0, randomly sample images instead of taking first N.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --sample-random is enabled.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Inference device, e.g. "cuda" or "cpu".',
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help=(
            "Hugging Face token for gated model access. "
            "If omitted, uses HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Keep top-K most important boxes per image; <= 0 keeps all.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Score threshold for SAM3 instance post-processing.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=DEFAULT_MASK_THRESHOLD,
        help="Mask threshold for SAM3 instance post-processing.",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=DEFAULT_MIN_BOX_AREA,
        help="Minimum 2D bbox area to keep.",
    )
    return parser.parse_args()


def collect_images(input_root: Path, image_glob: str) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    image_paths = [
        p for p in input_root.glob(image_glob) if p.suffix.lower() in IMAGE_EXTS
    ]
    image_paths.sort()
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {input_root} with pattern '{image_glob}'"
        )
    return image_paths


def to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1, y1, w, h]


def class_priority(label: str) -> float:
    key = label.lower()
    if "pedestrian" in key:
        return 1.0
    if "bicycle" in key or "motorcycle" in key:
        return 0.95
    if "bus" in key or "truck" in key:
        return 0.65
    if "car" in key:
        return 0.8
    if "traffic cone" in key or "traffic barrier" in key:
        return 0.85
    if "traffic sign" in key:
        return 0.5
    return 0.6


def compute_importance_score(
    ann: BoxAnnotation, image_width: int, image_height: int
) -> float:
    x1, _, x2, y2 = ann.bbox_xyxy
    center_x = 0.5 * (x1 + x2)

    proximity = max(0.0, min(1.0, y2 / max(1.0, float(image_height))))
    center_dist = abs(center_x - (0.5 * float(image_width))) / max(
        1.0, 0.5 * float(image_width)
    )
    center_path = max(0.0, 1.0 - center_dist)
    conf = max(0.0, min(1.0, ann.score))
    cls = class_priority(ann.label)

    return 0.35 * conf + 0.30 * proximity + 0.20 * center_path + 0.15 * cls


def select_top_k_annotations(
    annotations: list[BoxAnnotation],
    image_width: int,
    image_height: int,
    top_k: int,
) -> list[BoxAnnotation]:
    for ann in annotations:
        ann.importance_score = compute_importance_score(ann, image_width, image_height)

    ranked = sorted(
        annotations,
        key=lambda a: (a.importance_score, a.score),
        reverse=True,
    )
    if top_k <= 0:
        return ranked
    return ranked[:top_k]


def build_annotations_for_prompt(
    model: Sam3Model,
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
    device: str,
    score_threshold: float,
    mask_threshold: float,
    min_box_area: float,
) -> list[BoxAnnotation]:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    boxes = result.get("boxes")
    scores = result.get("scores")
    if boxes is None or scores is None:
        return []

    boxes = boxes.detach().cpu().tolist()
    scores = scores.detach().cpu().tolist()

    annotations: list[BoxAnnotation] = []
    for box_xyxy, score in zip(boxes, scores):
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        xywh = to_xywh(x1, y1, x2, y2)
        area = xywh[2] * xywh[3]
        if area < min_box_area:
            continue
        annotations.append(
            BoxAnnotation(
                label=prompt,
                score=float(score),
                bbox_xyxy=[x1, y1, x2, y2],
                bbox_xywh=xywh,
            )
        )
    return annotations


def draw_boxes(image: Image.Image, boxes: Iterable[BoxAnnotation]) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for ann in boxes:
        x1, y1, x2, y2 = ann.bbox_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(255, 30, 30), width=3)
        text = f"{ann.label}:{ann.score:.2f}"
        text_y = max(0.0, y1 - 14.0)
        draw.text((x1, text_y), text, fill=(255, 30, 30))
    return canvas


def resolve_hf_token(cli_token: str | None) -> str:
    token = cli_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN), "
            "or pass --hf-token."
        )
    return token


def progress_iter(items, *, total: int | None = None, desc: str = ""):
    if tqdm is not None:
        yield from tqdm(items, total=total, desc=desc)
        return

    count = 0
    for item in items:
        count += 1
        if total is not None and (count == 1 or count == total or count % 10 == 0):
            label = f"{desc}: " if desc else ""
            print(f"{label}{count}/{total}")
        yield item


def main() -> None:
    args = parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        raise ValueError("No valid prompt provided in --prompts.")

    image_paths = collect_images(args.input_root, args.image_glob)
    if args.max_images > 0:
        limit = min(args.max_images, len(image_paths))
        if args.sample_random:
            rng = random.Random(args.seed)
            image_paths = rng.sample(image_paths, k=limit)
            image_paths.sort()
        else:
            image_paths = image_paths[:limit]

    hf_token = resolve_hf_token(args.hf_token)
    login(token=hf_token, add_to_git_credential=False)

    print(f"Loading SAM3 model on {args.device} ...")
    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
    model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token).to(args.device)
    model.eval()

    args.viz_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    for idx, image_path in enumerate(
        progress_iter(image_paths, total=len(image_paths), desc="Annotating Waymo images"),
        start=1,
    ):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        image_annotations: list[BoxAnnotation] = []
        for prompt in prompts:
            anns = build_annotations_for_prompt(
                model=model,
                processor=processor,
                image=image,
                prompt=prompt,
                device=args.device,
                score_threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
                min_box_area=args.min_box_area,
            )
            image_annotations.extend(anns)

        image_annotations = select_top_k_annotations(
            image_annotations,
            image_width=width,
            image_height=height,
            top_k=args.top_k,
        )

        rel_path = image_path.relative_to(args.input_root).as_posix()
        record = {
            "image_path": rel_path,
            "width": width,
            "height": height,
            "annotations": [asdict(ann) for ann in image_annotations],
        }
        all_records.append(record)

        viz = draw_boxes(image, image_annotations)
        viz_name = image_path.stem + "_boxes.jpg"
        viz.save(args.viz_dir / viz_name, quality=95)

        print(
            f"[{idx}/{len(image_paths)}] {rel_path}: "
            f"{len(image_annotations)} boxes"
        )

    output = {
        "model": "facebook/sam3",
        "input_root": str(args.input_root),
        "image_glob": args.image_glob,
        "prompts": prompts,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "min_box_area": args.min_box_area,
        "top_k": args.top_k,
        "images": all_records,
    }
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved annotation JSON to: {args.output_json}")
    print(f"Saved visualization images to: {args.viz_dir}")


if __name__ == "__main__":
    main()
