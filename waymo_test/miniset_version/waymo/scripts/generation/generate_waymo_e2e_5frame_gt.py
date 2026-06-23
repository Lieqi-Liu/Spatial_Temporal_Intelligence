#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from generate_answers_waymo import (
    FRAME_DT_S,
    Obj,
    answer_text,
    assign_track_ids,
    build_object_ref,
    choose_selected_object,
    coarse_class,
    dedupe_objects,
    infer_sp1_choice_waymo,
    infer_su1_choice_waymo,
    infer_su3_choice_waymo,
    infer_su4_choice_waymo,
    infer_su5_choice_waymo,
    infer_su6_choice_waymo,
    infer_te1_choice_waymo,
    infer_te2_choice_waymo,
    infer_te4_choice_waymo,
    infer_te5_choice_waymo,
    match_object,
    sp2_from_pose,
    sp3_from_distance,
    sp4_from_lateral,
    sp5_from_bbox,
    tm2_choice_from_sp2,
)
from generate_waymo_e2e_questions import (
    ATTENTION_FACTOR_TEXT,
    COUNTERFACTUAL_ISSUE_CHOICES,
    DEFAULT_DATASET_ROOT,
    SCENARIO_OPTION_TEXT,
    TRAJ_CHOICES,
    TRJ_ENDPOINT_CHOICES,
    TRJ_MOTION_CHOICES,
    TRJ_SPEED_CHOICES,
    best_and_worst,
    build_llm_prompt,
    build_task_template_map,
    describe_maneuver,
    get_template,
    has_enabled_task,
    history_frame_paths,
    infer_counterfactual_issue,
    infer_endpoint_region,
    infer_future_speed_trend,
    infer_maneuver_choice,
    label_paths,
    load_json,
    load_optional_json,
    make_base_item,
    make_dynamic_choices,
    normalize_cluster,
    parse_frame_name,
    points_relative_to_start,
    progress_iter,
    sample_future_points,
    state_points,
    state_speed_mps,
    trajectory_summary,
    valid_preference_trajectories,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_JSON = SCRIPT_DIR / "waymo_e2e_question_templates_curated.json"
DEFAULT_BUNDLE_DIR = SCRIPT_DIR / "waymo_5frame_front_bundles"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "waymo_e2e_questions_5frame_gt.json"


def counterfactual_maneuver_phrase(alternative_maneuver: str) -> str:
    mapping = {
        "move or turn left": "turns left",
        "move or turn right": "turns right",
        "slightly veer left": "slightly veers left",
        "slightly veer right": "slightly veers right",
        "continue mostly straight": "continues mostly straight",
        "slow down or nearly stop": "slows down or nearly stops",
    }
    return mapping.get(alternative_maneuver.strip(), alternative_maneuver.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Waymo E2E questions and GT from 5-frame front-view bundles. "
            "This is a separate pipeline and does not modify the original single-frame outputs."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Waymo split root (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help=f"Directory containing 5-frame bundle JSON files (default: {DEFAULT_BUNDLE_DIR}).",
    )
    parser.add_argument(
        "--template-json",
        type=Path,
        default=DEFAULT_TEMPLATE_JSON,
        help=f"Curated template JSON (default: {DEFAULT_TEMPLATE_JSON}).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON}).",
    )
    parser.add_argument(
        "--bundle-size",
        type=int,
        default=5,
        help="Number of frames per bundle (default: 5).",
    )
    parser.add_argument(
        "--bundle-stride",
        type=int,
        default=5,
        help=(
            "Stride between consecutive bundle anchors. "
            "Use 5 for non-overlapping 5-frame groups (default: 5)."
        ),
    )
    parser.add_argument(
        "--trajectory-points",
        type=int,
        default=5,
        help="Number of future points shown for each candidate trajectory.",
    )
    parser.add_argument(
        "--trajectory-sampling",
        choices=["uniform", "first"],
        default="uniform",
        help="How to sample displayed preference trajectories.",
    )
    parser.add_argument(
        "--include-frq",
        action="store_true",
        help="Also emit TRJ-9 FRQ items.",
    )
    parser.add_argument(
        "--limit-bundles",
        type=int,
        default=None,
        help="Optional maximum number of bundles to scan.",
    )
    parser.add_argument(
        "--auto-build-bundles",
        action="store_true",
        help="If bundle-dir is missing or empty, build front-only 5-frame bundles first.",
    )
    parser.add_argument(
        "--overwrite-bundles",
        action="store_true",
        help="If auto-building bundles, overwrite any existing bundle files.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write indented JSON.",
    )
    return parser.parse_args()


def ensure_bundles(args: argparse.Namespace) -> None:
    bundle_dir = args.bundle_dir
    manifest_path = bundle_dir / "manifest.json"
    has_bundle_files = bundle_dir.exists() and any(bundle_dir.glob("*/*_bundle.json"))
    if manifest_path.exists() and has_bundle_files:
        return
    if not args.auto_build_bundles:
        raise FileNotFoundError(
            f"No 5-frame bundles found under {bundle_dir}. "
            "Run build_waymo_5frame_bundles.py first or pass --auto-build-bundles."
        )

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_waymo_5frame_bundles.py"),
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(bundle_dir),
        "--bundle-size",
        str(args.bundle_size),
        "--bundle-stride",
        str(args.bundle_stride),
        "--no-stitch-image",
    ]
    if args.overwrite_bundles:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True)


def bundle_paths(bundle_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(bundle_dir.glob("*/*_bundle.json"))
    if limit is not None:
        return paths[:limit]
    return paths


def bundle_entry_to_objects(entry: dict[str, Any]) -> dict[str, Any]:
    distance_rec = entry.get("distance_estimates") or {}
    width = int(distance_rec.get("image_size", {}).get("width", 0))
    height = int(distance_rec.get("image_size", {}).get("height", 0))
    objects: list[Obj] = []
    for i, ann in enumerate(distance_rec.get("distance_estimates", [])):
        details = ann.get("details", {}) if isinstance(ann.get("details"), dict) else {}
        uv = ann.get("sample_pixel_uv", [0.0, 0.0])
        bbox = ann.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
        objects.append(
            Obj(
                obj_id=f"{entry['frame_name']}#obj{i:02d}",
                label=str(ann.get("label", "")),
                coarse=coarse_class(str(ann.get("label", ""))),
                score=float(ann.get("score", 0.0) or 0.0),
                bbox=[float(v) for v in bbox],
                u=float(uv[0] if len(uv) > 0 else 0.0),
                v=float(uv[1] if len(uv) > 1 else 0.0),
                distance_m=float(ann["distance_m"]) if ann.get("distance_m") is not None else None,
                forward_m=(
                    float(details["forward_m"])
                    if details.get("forward_m") is not None
                    else None
                ),
                lateral_m=(
                    float(details["lateral_m"])
                    if details.get("lateral_m") is not None
                    else None
                ),
            )
        )
    return {
        "frame_name": entry["frame_name"],
        "scene_id": parse_frame_name(entry["frame_name"])[0],
        "frame_idx": int(entry["frame_index"]),
        "width": width,
        "height": height,
        "objects": dedupe_objects(objects),
    }


def add_bundle_context(item: dict[str, Any], bundle: dict[str, Any]) -> None:
    frame_entries = bundle.get("frame_entries", [])
    item["bundle_id"] = bundle.get("bundle_id")
    item["bundle_size"] = bundle.get("bundle_size")
    item["bundle_frame_names"] = [entry.get("frame_name") for entry in frame_entries]
    item["history_image_paths"] = [entry.get("image_path") for entry in frame_entries]
    item["anchor_image_path"] = frame_entries[-1].get("image_path") if frame_entries else None


def add_bundle_object_question(
    *,
    questions: list[dict[str, Any]],
    template: dict[str, Any],
    split: str,
    bundle: dict[str, Any],
    anchor_entry: dict[str, Any],
    scenario_cluster: str,
    obj: Obj,
    object_ref: str,
    ground_truth: str,
    ground_truth_text: str,
    extra: dict[str, Any] | None = None,
) -> None:
    frame_name = str(anchor_entry["frame_name"])
    scene_id, frame_idx = parse_frame_name(frame_name)
    item = make_base_item(
        question_id=f"waymo_{split}_{bundle['bundle_id'].replace(':', '_')}_{template['id']}_{obj.obj_id.split('#')[-1]}",
        task_id=template["id"],
        task=template["task"],
        question_format=template["question_format"],
        question=str(template["question"]).replace("<obj>", object_ref),
        frame_name=frame_name,
        scene_id=scene_id,
        frame_idx=frame_idx,
        image_path=Path(str(anchor_entry["image_path"])),
        scenario_cluster=scenario_cluster,
    )
    item["object_id"] = obj.obj_id
    item["object_reference"] = object_ref
    item["object_label"] = obj.label
    item["bbox_xyxy"] = [round(float(v), 3) for v in obj.bbox]
    if isinstance(template.get("choices"), dict):
        item["choices"] = template["choices"]
    item["ground_truth"] = ground_truth
    item["ground_truth_text"] = ground_truth_text
    add_bundle_context(item, bundle)
    if extra:
        item.update(extra)
    questions.append(item)


def main() -> None:
    args = parse_args()
    ensure_bundles(args)

    templates = load_json(args.template_json)
    task_map = build_task_template_map(templates)
    questions: list[dict[str, Any]] = []
    counts = {
        "bundles_scanned": 0,
        "bundles_with_object_metadata": 0,
        "bundles_with_valid_preference": 0,
        "questions_generated": 0,
    }

    bundle_path_list = bundle_paths(args.bundle_dir, args.limit_bundles)
    for bundle_path in progress_iter(
        bundle_path_list,
        total=len(bundle_path_list),
        desc="Generating Waymo 5-frame GT",
    ):
        counts["bundles_scanned"] += 1
        bundle = load_json(bundle_path)
        frame_entries = bundle.get("frame_entries", [])
        if len(frame_entries) != args.bundle_size:
            continue

        anchor_entry = frame_entries[-1]
        frame_name = str(anchor_entry["frame_name"])
        scene_id, frame_idx = parse_frame_name(frame_name)
        cluster = normalize_cluster(bundle.get("scenario_cluster"))
        image_path = Path(str(anchor_entry["image_path"]))
        history_paths = [str(entry.get("image_path")) for entry in frame_entries]

        ego_status_path = anchor_entry.get("ego_status_path")
        ego_status = load_optional_json(Path(ego_status_path)) if ego_status_path else {}
        ego_status = ego_status or {}

        labels_path = anchor_entry.get("labels_path")
        label_payload = load_optional_json(Path(labels_path)) if labels_path else {}
        label_payload = label_payload or {}

        bundle_frames = [bundle_entry_to_objects(entry) for entry in frame_entries]
        assign_track_ids(bundle_frames)
        anchor_frame = bundle_frames[-1]
        objects: list[Obj] = anchor_frame["objects"]

        observed_points = points_relative_to_start(
            state_points(ego_status, "past_states", count=args.bundle_size)
        )
        future_points = state_points(ego_status, "future_states", count=5)
        valid_trajs = valid_preference_trajectories(label_payload)

        if objects:
            counts["bundles_with_object_metadata"] += 1
            selected = choose_selected_object(objects)
            object_ref = build_object_ref(selected)
            prev_frame = bundle_frames[-2] if len(bundle_frames) >= 2 else None
            prev2_frame = bundle_frames[-3] if len(bundle_frames) >= 3 else None
            prev_match = match_object(selected, prev_frame["objects"]) if prev_frame else None
            prev2_match = match_object(selected, prev2_frame["objects"]) if prev2_frame else None

            sp1, sp1_metrics = infer_sp1_choice_waymo(selected, prev_match, FRAME_DT_S)
            sp2, sp2_metrics = sp2_from_pose(selected.forward_m, selected.lateral_m)
            sp3, sp3_metrics = sp3_from_distance(selected.distance_m)
            sp4, sp4_metrics = sp4_from_lateral(selected.lateral_m)
            sp5, sp5_metrics = sp5_from_bbox(selected, anchor_frame["width"], anchor_frame["height"])

            for task_id, value, metrics in [
                ("SP-2", sp2, sp2_metrics),
                ("SP-3", sp3, sp3_metrics),
            ]:
                if not has_enabled_task(task_id, task_map):
                    continue
                add_bundle_object_question(
                    questions=questions,
                    template=task_map[task_id],
                    split="val",
                    bundle=bundle,
                    anchor_entry=anchor_entry,
                    scenario_cluster=cluster,
                    obj=selected,
                    object_ref=object_ref,
                    ground_truth=value,
                    ground_truth_text=answer_text(task_map, task_id, value),
                    extra={"hidden_metadata": metrics},
                )

            if has_enabled_task("SP-3a", task_map):
                sp3a_answer = "unknown"
                sp3a_metrics: dict[str, Any] = {"distance_m": None}
                if selected.distance_m is not None:
                    sp3a_answer = f"{float(selected.distance_m):.1f} m"
                    sp3a_metrics = {"distance_m": round(float(selected.distance_m), 4)}
                add_bundle_object_question(
                    questions=questions,
                    template=task_map["SP-3a"],
                    split="val",
                    bundle=bundle,
                    anchor_entry=anchor_entry,
                    scenario_cluster=cluster,
                    obj=selected,
                    object_ref=object_ref,
                    ground_truth=sp3a_answer,
                    ground_truth_text=sp3a_answer,
                    extra={"hidden_metadata": sp3a_metrics},
                )

            su1, _ = infer_su1_choice_waymo(
                sp1_choice=sp1,
                sp2_choice=sp2,
                sp3_choice=sp3,
                sp4_choice=sp4,
                sp1_metrics=sp1_metrics,
            )
            su3, _ = infer_su3_choice_waymo(
                sp1_choice=sp1,
                sp2_choice=sp2,
                sp3_choice=sp3,
                su1_choice=su1,
                sp1_metrics=sp1_metrics,
            )
            su4, _ = infer_su4_choice_waymo(
                sp1_choice=sp1,
                sp2_choice=sp2,
                sp3_choice=sp3,
                sp4_choice=sp4,
                su1_choice=su1,
                su3_choice=su3,
            )
            su5, _ = infer_su5_choice_waymo(
                sp1_choice=sp1,
                sp2_choice=sp2,
                sp3_choice=sp3,
                su1_choice=su1,
                su3_choice=su3,
                su4_choice=su4,
            )
            su6, _ = infer_su6_choice_waymo([o for o in objects if o.coarse == "vehicle"])

            te1, te1_metrics = infer_te1_choice_waymo(sp1_metrics)
            te2, te2_metrics = infer_te2_choice_waymo(selected.distance_m, sp1_metrics)
            te4, te4_metrics = infer_te4_choice_waymo(te1, te2, su3, su4, su5, te2_metrics)
            te5, te5_metrics = infer_te5_choice_waymo(te1, te2, "A", su3, su6)
            for task_id, value, metrics in [
                ("TE-1", te1, te1_metrics),
                ("TE-2", te2, te2_metrics),
                ("TE-4", te4, te4_metrics),
                ("TE-5", te5, te5_metrics),
            ]:
                if not has_enabled_task(task_id, task_map):
                    continue
                add_bundle_object_question(
                    questions=questions,
                    template=task_map[task_id],
                    split="val",
                    bundle=bundle,
                    anchor_entry=anchor_entry,
                    scenario_cluster=cluster,
                    obj=selected,
                    object_ref=object_ref,
                    ground_truth=value,
                    ground_truth_text=answer_text(task_map, task_id, value),
                    extra={"hidden_metadata": metrics},
                )

            if prev_match is not None:
                tm2_sp2, tm2_metrics = sp2_from_pose(prev_match.forward_m, prev_match.lateral_m)
                tm2 = tm2_choice_from_sp2(tm2_sp2)
                tm2_metrics = {
                    "used_fallback": False,
                    "x_ego": prev_match.forward_m,
                    "y_ego": prev_match.lateral_m,
                    "angle_deg": tm2_metrics.get("angle_deg"),
                }
            else:
                tm2 = "A"
                tm2_metrics = {"used_fallback": True, "x_ego": 0.0, "y_ego": 0.0, "angle_deg": 0.0}
            if has_enabled_task("TM-2", task_map):
                add_bundle_object_question(
                    questions=questions,
                    template=task_map["TM-2"],
                    split="val",
                    bundle=bundle,
                    anchor_entry=anchor_entry,
                    scenario_cluster=cluster,
                    obj=selected,
                    object_ref=object_ref,
                    ground_truth=tm2,
                    ground_truth_text=answer_text(task_map, "TM-2", tm2),
                    extra={"hidden_metadata": tm2_metrics},
                )

            prev_range_rate = 0.0
            if prev_match is not None and prev2_match is not None:
                if prev_match.distance_m is not None and prev2_match.distance_m is not None:
                    prev_range_rate = (prev_match.distance_m - prev2_match.distance_m) / FRAME_DT_S
            speed1 = float(sp1_metrics.get("speed_rel", 0.0))
            speed0 = max(0.0, speed1 - float(sp1_metrics.get("range_rate", 0.0)) * 0.1)
            vx1 = float(sp1_metrics.get("vx_rel", 0.0))
            vy1 = float(sp1_metrics.get("vy_rel", 0.0))
            if speed1 < 0.35:
                tm3, reason = "F", "stopped_recently"
            elif abs(vy1) > abs(vx1) * 1.1 and abs(vy1) > 0.6:
                tm3, reason = ("D", "lateral_turning_pattern") if vy1 > 0 else ("E", "lateral_turning_pattern")
            elif speed1 - speed0 > 0.5:
                tm3, reason = "C", "accelerating_trend"
            elif speed0 - speed1 > 0.5:
                tm3, reason = "B", "slowing_trend"
            else:
                tm3, reason = "A", "straight_motion_trend"
            if has_enabled_task("TM-3", task_map):
                add_bundle_object_question(
                    questions=questions,
                    template=task_map["TM-3"],
                    split="val",
                    bundle=bundle,
                    anchor_entry=anchor_entry,
                    scenario_cluster=cluster,
                    obj=selected,
                    object_ref=object_ref,
                    ground_truth=tm3,
                    ground_truth_text=answer_text(task_map, "TM-3", tm3),
                    extra={"hidden_metadata": {
                        "reason": reason,
                        "speed0": round(speed0, 4),
                        "speed1": round(speed1, 4),
                        "vx_ego": round(vx1, 4),
                        "vy_ego": round(vy1, 4),
                        "prev_range_rate_mps": round(prev_range_rate, 4),
                    }},
                )

            if prev_match is not None:
                tm5_sp4, tm5_metrics = sp4_from_lateral(prev_match.lateral_m)
                tm5 = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "D"}.get(tm5_sp4, "A")
            else:
                tm5 = "A"
                tm5_metrics = {"reason": "no_prev_match"}
            if has_enabled_task("TM-5", task_map):
                add_bundle_object_question(
                    questions=questions,
                    template=task_map["TM-5"],
                    split="val",
                    bundle=bundle,
                    anchor_entry=anchor_entry,
                    scenario_cluster=cluster,
                    obj=selected,
                    object_ref=object_ref,
                    ground_truth=tm5,
                    ground_truth_text=answer_text(task_map, "TM-5", tm5),
                    extra={"hidden_metadata": tm5_metrics},
                )

        sc1 = get_template(templates, "SC-1")
        item = make_base_item(
            question_id=f"waymo_val_{bundle['bundle_id'].replace(':', '_')}_SC-1",
            task_id="SC-1",
            task=sc1["task"],
            question_format=sc1["question_format"],
            question=sc1["question"],
            frame_name=frame_name,
            scene_id=scene_id,
            frame_idx=frame_idx,
            image_path=image_path,
            scenario_cluster=cluster,
        )
        choices, ground_truth = make_dynamic_choices(
            correct_key=cluster,
            option_text_by_key=SCENARIO_OPTION_TEXT,
            seed_key=f"{bundle['bundle_id']}:SC-1",
        )
        item["choices"] = choices
        item["ground_truth"] = ground_truth
        item["ground_truth_text"] = item["choices"][item["ground_truth"]]
        add_bundle_context(item, bundle)
        questions.append(item)

        sc2 = get_template(templates, "SC-2")
        item = make_base_item(
            question_id=f"waymo_val_{bundle['bundle_id'].replace(':', '_')}_SC-2",
            task_id="SC-2",
            task=sc2["task"],
            question_format=sc2["question_format"],
            question=sc2["question"],
            frame_name=frame_name,
            scene_id=scene_id,
            frame_idx=frame_idx,
            image_path=image_path,
            scenario_cluster=cluster,
        )
        choices, ground_truth = make_dynamic_choices(
            correct_key=cluster,
            option_text_by_key=ATTENTION_FACTOR_TEXT,
            seed_key=f"{bundle['bundle_id']}:SC-2",
        )
        item["choices"] = choices
        item["ground_truth"] = ground_truth
        item["ground_truth_text"] = item["choices"][item["ground_truth"]]
        add_bundle_context(item, bundle)
        questions.append(item)

        if len(history_paths) == args.bundle_size and len(observed_points) >= args.bundle_size and len(future_points) >= 5:
            trj1 = get_template(templates, "TRJ-1")
            choice, reason = infer_maneuver_choice(observed_points)
            item = make_base_item(
                question_id=f"waymo_val_{bundle['bundle_id'].replace(':', '_')}_TRJ-1",
                task_id="TRJ-1",
                task=trj1["task"],
                question_format=trj1["question_format"],
                question=trj1["question"],
                frame_name=frame_name,
                scene_id=scene_id,
                frame_idx=frame_idx,
                image_path=image_path,
                scenario_cluster=cluster,
            )
            item["choices"] = TRJ_MOTION_CHOICES
            item["observed_trajectory"] = observed_points
            item["ground_truth"] = choice
            item["ground_truth_text"] = TRJ_MOTION_CHOICES[choice]
            item["hidden_metadata"] = {
                "rule_reason": reason,
                "current_speed_mps": round(state_speed_mps(ego_status, "past_states") or 0.0, 3),
            }
            add_bundle_context(item, bundle)
            questions.append(item)

            trj2 = get_template(templates, "TRJ-2")
            choice, reason = infer_maneuver_choice(future_points)
            item = make_base_item(
                question_id=f"waymo_val_{bundle['bundle_id'].replace(':', '_')}_TRJ-2",
                task_id="TRJ-2",
                task=trj2["task"],
                question_format=trj2["question_format"],
                question=trj2["question"],
                frame_name=frame_name,
                scene_id=scene_id,
                frame_idx=frame_idx,
                image_path=image_path,
                scenario_cluster=cluster,
            )
            item["choices"] = TRJ_MOTION_CHOICES
            item["observed_trajectory"] = observed_points
            item["future_trajectory"] = future_points
            item["ground_truth"] = choice
            item["ground_truth_text"] = TRJ_MOTION_CHOICES[choice]
            item["hidden_metadata"] = {"rule_reason": reason}
            add_bundle_context(item, bundle)
            questions.append(item)

            trj3 = get_template(templates, "TRJ-3")
            choice, reason = infer_endpoint_region(future_points)
            item = make_base_item(
                question_id=f"waymo_val_{bundle['bundle_id'].replace(':', '_')}_TRJ-3",
                task_id="TRJ-3",
                task=trj3["task"],
                question_format=trj3["question_format"],
                question=trj3["question"],
                frame_name=frame_name,
                scene_id=scene_id,
                frame_idx=frame_idx,
                image_path=image_path,
                scenario_cluster=cluster,
            )
            item["choices"] = TRJ_ENDPOINT_CHOICES
            item["future_trajectory"] = future_points
            item["ground_truth"] = choice
            item["ground_truth_text"] = TRJ_ENDPOINT_CHOICES[choice]
            item["hidden_metadata"] = {"rule_reason": reason}
            add_bundle_context(item, bundle)
            questions.append(item)

            trj4 = get_template(templates, "TRJ-4")
            choice, reason = infer_future_speed_trend(future_points)
            item = make_base_item(
                question_id=f"waymo_val_{bundle['bundle_id'].replace(':', '_')}_TRJ-4",
                task_id="TRJ-4",
                task=trj4["task"],
                question_format=trj4["question_format"],
                question=trj4["question"],
                frame_name=frame_name,
                scene_id=scene_id,
                frame_idx=frame_idx,
                image_path=image_path,
                scenario_cluster=cluster,
            )
            item["choices"] = TRJ_SPEED_CHOICES
            item["future_trajectory"] = future_points
            item["ground_truth"] = choice
            item["ground_truth_text"] = TRJ_SPEED_CHOICES[choice]
            item["hidden_metadata"] = {"rule_reason": reason}
            add_bundle_context(item, bundle)
            questions.append(item)

            if has_enabled_task("TRJ-5", task_map):
                trj5 = get_template(templates, "TRJ-5")
                item = make_base_item(
                    question_id=f"waymo_val_{bundle['bundle_id'].replace(':', '_')}_TRJ-5",
                    task_id="TRJ-5",
                    task=trj5["task"],
                    question_format=trj5["question_format"],
                    question=trj5["question"],
                    frame_name=frame_name,
                    scene_id=scene_id,
                    frame_idx=frame_idx,
                    image_path=image_path,
                    scenario_cluster=cluster,
                )
                item["observed_trajectory"] = observed_points
                item["ground_truth"] = json.dumps(future_points, ensure_ascii=True)
                item["ground_truth_text"] = item["ground_truth"]
                add_bundle_context(item, bundle)
                questions.append(item)

            if has_enabled_task("TRJ-6", task_map):
                trj6 = get_template(templates, "TRJ-6")
                item = make_base_item(
                    question_id=f"waymo_val_{bundle['bundle_id'].replace(':', '_')}_TRJ-6",
                    task_id="TRJ-6",
                    task=trj6["task"],
                    question_format=trj6["question_format"],
                    question=trj6["question"],
                    frame_name=frame_name,
                    scene_id=scene_id,
                    frame_idx=frame_idx,
                    image_path=image_path,
                    scenario_cluster=cluster,
                )
                item["observed_trajectory"] = observed_points
                item["known_future_point_local"] = future_points[0]
                item["ground_truth"] = json.dumps(future_points[1:], ensure_ascii=True)
                item["ground_truth_text"] = item["ground_truth"]
                add_bundle_context(item, bundle)
                questions.append(item)

            if has_enabled_task("TRJ-7", task_map):
                trj7 = get_template(templates, "TRJ-7")
                trj2_choice, _ = infer_maneuver_choice(future_points)
                trj3_choice, _ = infer_endpoint_region(future_points)
                trj4_choice, _ = infer_future_speed_trend(future_points)
                item = make_base_item(
                    question_id=f"waymo_val_{bundle['bundle_id'].replace(':', '_')}_TRJ-7",
                    task_id="TRJ-7",
                    task=trj7["task"],
                    question_format=trj7["question_format"],
                    question=trj7["question"],
                    frame_name=frame_name,
                    scene_id=scene_id,
                    frame_idx=frame_idx,
                    image_path=image_path,
                    scenario_cluster=cluster,
                )
                item["observed_trajectory"] = observed_points
                item["future_trajectory"] = future_points
                item["ground_truth"] = (
                    f"The ego vehicle will most likely {TRJ_MOTION_CHOICES[trj2_choice].lower()} "
                    f"and end in the {TRJ_ENDPOINT_CHOICES[trj3_choice].lower()}, while "
                    f"{TRJ_SPEED_CHOICES[trj4_choice].lower()}."
                )
                item["ground_truth_text"] = item["ground_truth"]
                add_bundle_context(item, bundle)
                questions.append(item)

        if valid_trajs:
            counts["bundles_with_valid_preference"] += 1
            best, worst = best_and_worst(valid_trajs)
            candidates = {
                traj["choice"]: sample_future_points(
                    traj["points"],
                    args.trajectory_points,
                    args.trajectory_sampling,
                )
                for traj in valid_trajs
            }
            summaries = {
                traj["choice"]: trajectory_summary(traj["points"])
                for traj in valid_trajs
            }
            hidden_scores = {traj["choice"]: traj["score"] for traj in valid_trajs}

            trj8 = get_template(templates, "TRJ-8")
            item = make_base_item(
                question_id=f"waymo_val_{bundle['bundle_id'].replace(':', '_')}_TRJ-8",
                task_id="TRJ-8",
                task=trj8["task"],
                question_format=trj8["question_format"],
                question=trj8["question"],
                frame_name=frame_name,
                scene_id=scene_id,
                frame_idx=frame_idx,
                image_path=image_path,
                scenario_cluster=cluster,
            )
            item["choices"] = TRAJ_CHOICES
            item["candidate_trajectories"] = candidates
            item["trajectory_sampling"] = args.trajectory_sampling
            item["ground_truth"] = best["choice"]
            item["ground_truth_text"] = TRAJ_CHOICES[item["ground_truth"]]
            item["hidden_metadata"] = {
                "preference_scores": hidden_scores,
                "score_rank_desc": [
                    traj["choice"]
                    for traj in sorted(valid_trajs, key=lambda x: x["score"], reverse=True)
                ],
            }
            add_bundle_context(item, bundle)
            questions.append(item)

            trj10 = get_template(templates, "TRJ-10")
            alternative_maneuver = describe_maneuver(summaries[worst["choice"]])
            alternative_phrase = counterfactual_maneuver_phrase(alternative_maneuver)
            issue_choice, issue_reason = infer_counterfactual_issue(
                cluster=cluster,
                best_summary=summaries[best["choice"]],
                alternative_summary=summaries[worst["choice"]],
                best_score=float(best["score"]),
                alternative_score=float(worst["score"]),
            )
            item = make_base_item(
                question_id=f"waymo_val_{bundle['bundle_id'].replace(':', '_')}_TRJ-10",
                task_id="TRJ-10",
                task=trj10["task"],
                question_format=trj10["question_format"],
                question=(
                    f"If the ego vehicle {alternative_phrase}, "
                    "what is the most likely consequence?"
                ),
                frame_name=frame_name,
                scene_id=scene_id,
                frame_idx=frame_idx,
                image_path=image_path,
                scenario_cluster=cluster,
            )
            item["choices"] = COUNTERFACTUAL_ISSUE_CHOICES
            item["alternative_trajectory"] = candidates[worst["choice"]]
            item["alternative_maneuver"] = alternative_maneuver
            item["candidate_trajectories"] = candidates
            item["trajectory_sampling"] = args.trajectory_sampling
            item["ground_truth"] = issue_choice
            item["ground_truth_text"] = COUNTERFACTUAL_ISSUE_CHOICES[issue_choice]
            item["hidden_metadata"] = {
                "best_candidate": best["choice"],
                "worst_candidate": worst["choice"],
                "preference_scores": hidden_scores,
                "trajectory_summaries": summaries,
                "score_gap": round(float(best["score"]) - float(worst["score"]), 3),
                "counterfactual_rule_reason": issue_reason,
            }
            add_bundle_context(item, bundle)
            questions.append(item)

            if args.include_frq:
                trj9 = get_template(templates, "TRJ-9")
                item = make_base_item(
                    question_id=f"waymo_val_{bundle['bundle_id'].replace(':', '_')}_TRJ-9",
                    task_id="TRJ-9",
                    task=trj9["task"],
                    question_format=trj9["question_format"],
                    question=trj9["question"],
                    frame_name=frame_name,
                    scene_id=scene_id,
                    frame_idx=frame_idx,
                    image_path=image_path,
                    scenario_cluster=cluster,
                )
                item["selected_candidate"] = best["choice"]
                item["candidate_trajectories"] = candidates
                item["ground_truth"] = ""
                item["ground_truth_prompt"] = build_llm_prompt(
                    task_id="TRJ-9",
                    correct_choice=best["choice"],
                    scenario_cluster=cluster,
                    candidates=candidates,
                    summaries=summaries,
                )
                item["hidden_metadata"] = {
                    "preference_scores": hidden_scores,
                    "trajectory_summaries": summaries,
                }
                add_bundle_context(item, bundle)
                questions.append(item)

    counts["questions_generated"] = len(questions)
    payload = {
        "metadata": {
            "dataset": "waymo_e2e_5frame",
            "split": "val",
            "dataset_root": str(args.dataset_root),
            "bundle_dir": str(args.bundle_dir),
            "template_json": str(args.template_json),
            "output_json": str(args.output_json),
            "bundle_size": args.bundle_size,
            "bundle_stride": args.bundle_stride,
            "trajectory_points": args.trajectory_points,
            "trajectory_sampling": args.trajectory_sampling,
            "include_frq": args.include_frq,
            "counts": counts,
        },
        "questions": questions,
    }
    write_json(args.output_json, payload, args.pretty)
    print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
