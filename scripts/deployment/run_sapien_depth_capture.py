import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import yaml

REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
TORCH_ROBOTICS_PATH = os.path.join(REPO_PATH, "mpd", "torch_robotics")
if TORCH_ROBOTICS_PATH not in sys.path:
    sys.path.insert(0, TORCH_ROBOTICS_PATH)

from mpd.deployment.scene_primitives import build_object_fields_from_boxes
from mpd.deployment.sapien_depth_adapter import (
    build_capture_request,
    build_warehouse_scene_spec,
    capture_depth_with_sapien,
    parse_camera_specs,
    save_depth_capture_bundle,
    save_depth_preview_images,
    save_yaml,
)
from mpd.paths import REPO_PATH


def _resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_PATH, path)


def main():
    parser = argparse.ArgumentParser(description="Phase-2 SAPIEN depth capture scaffold for the warehouse scene")
    parser.add_argument(
        "--cfg",
        default="scripts/deployment/cfgs/phase2_sapien_depth_capture_warehouse.yaml",
        help="Path to the phase-2 SAPIEN capture YAML config",
    )
    args = parser.parse_args()

    cfg_path = _resolve_repo_path(args.cfg)
    with open(cfg_path, "r") as stream:
        cfg = yaml.safe_load(stream)

    results_dir = _resolve_repo_path(cfg.get("results_dir", "logs/phase2_sapien_depth_capture"))
    os.makedirs(results_dir, exist_ok=True)

    extra_object_fields = build_object_fields_from_boxes(cfg.get("extra_boxes"))
    scene_spec = build_warehouse_scene_spec(
        extra_boxes=extra_object_fields,
        rotation_z_axis_deg=cfg.get("rotation_z_axis_deg", 0.0),
    )
    camera_specs = parse_camera_specs(cfg["camera_specs"])
    capture_request = build_capture_request(
        scene_spec=scene_spec,
        camera_specs=camera_specs,
        metadata={"config_path": cfg_path},
    )

    scene_spec_path = os.path.join(results_dir, cfg.get("scene_spec_filename", "scene_spec.yaml"))
    capture_request_path = os.path.join(results_dir, cfg.get("capture_request_filename", "capture_request.yaml"))
    save_yaml(scene_spec, scene_spec_path)
    save_yaml(capture_request, capture_request_path)

    print("\n----------------PHASE 2 SAPIEN DEPTH----------------")
    print(f"cfg: {cfg_path}")
    print(f"results_dir: {results_dir}")
    print(f"scene_spec_path: {scene_spec_path}")
    print(f"capture_request_path: {capture_request_path}")
    print(f"n_scene_boxes: {len(scene_spec['boxes'])}")
    print(f"n_cameras: {len(camera_specs)}")
    print(f"capture_backend: {cfg.get('capture_backend', 'render_camera')}")
    print(f"render_robot: {cfg.get('robot', {}).get('enabled', False)}")

    if not cfg.get("attempt_live_capture", False):
        print("\nSkipping live capture because attempt_live_capture=false.")
        print("The exported YAML files are the scene/camera contract for the next SAPIEN integration step.")
        return

    bundle = capture_depth_with_sapien(
        scene_spec=scene_spec,
        camera_specs=camera_specs,
        metadata={"config_path": cfg_path},
        add_ground=cfg.get("add_ground", False),
        capture_backend=cfg.get("capture_backend", "render_camera"),
        stereo_sensor_config_overrides=cfg.get("stereo_sensor_config"),
        robot_cfg=cfg.get("robot"),
    )
    bundle_path = os.path.join(results_dir, cfg.get("bundle_filename", "depth_capture_bundle.npz"))
    save_depth_capture_bundle(bundle, bundle_path)
    preview_dir = os.path.join(results_dir, cfg.get("preview_dirname", "depth_previews"))
    preview_paths = save_depth_preview_images(
        bundle,
        output_dir=preview_dir,
        cmap=cfg.get("preview_cmap", "viridis"),
    )
    print(f"\nSaved depth capture bundle to: {bundle_path}")
    print(f"Saved depth previews to: {preview_dir}")
    for frame, preview_path in zip(bundle.frames, preview_paths):
        nonzero_depth = frame.depth_meters[frame.depth_meters > 0.0]
        if nonzero_depth.size > 0:
            depth_min = float(nonzero_depth.min())
            depth_max = float(nonzero_depth.max())
        else:
            depth_min = 0.0
            depth_max = 0.0
        print(
            f"  camera={frame.camera_name} shape={frame.depth_meters.shape} "
            f"valid_pixels={(frame.depth_meters > 0.0).sum()} depth_range_m=[{depth_min:.3f}, {depth_max:.3f}] "
            f"preview={preview_path}"
        )


if __name__ == "__main__":
    main()
