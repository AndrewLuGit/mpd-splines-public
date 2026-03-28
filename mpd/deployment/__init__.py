__all__ = [
    "DepthCameraSpec",
    "DepthCaptureBundle",
    "DepthFrame",
    "OnlineMPDPlanner",
    "SceneBoxSpec",
    "box_spec_to_object_field",
    "build_capture_request",
    "build_object_fields_from_boxes",
    "build_ee_pose_goal",
    "build_ee_pose_goal_from_dict",
    "build_warehouse_scene_spec",
    "parse_camera_specs",
    "render_panda_goal_ik_debug",
    "save_depth_capture_bundle",
    "solve_panda_goal_ik",
]


def __getattr__(name):
    if name in {"build_ee_pose_goal", "build_ee_pose_goal_from_dict", "render_panda_goal_ik_debug", "solve_panda_goal_ik"}:
        from .goal_ik import (
            build_ee_pose_goal,
            build_ee_pose_goal_from_dict,
            render_panda_goal_ik_debug,
            solve_panda_goal_ik,
        )

        return {
            "build_ee_pose_goal": build_ee_pose_goal,
            "build_ee_pose_goal_from_dict": build_ee_pose_goal_from_dict,
            "render_panda_goal_ik_debug": render_panda_goal_ik_debug,
            "solve_panda_goal_ik": solve_panda_goal_ik,
        }[name]

    if name == "OnlineMPDPlanner":
        from .online_planner import OnlineMPDPlanner

        return OnlineMPDPlanner

    if name in {"box_spec_to_object_field", "build_object_fields_from_boxes"}:
        from .scene_primitives import box_spec_to_object_field, build_object_fields_from_boxes

        return {
            "box_spec_to_object_field": box_spec_to_object_field,
            "build_object_fields_from_boxes": build_object_fields_from_boxes,
        }[name]

    if name in {
        "DepthCameraSpec",
        "DepthCaptureBundle",
        "DepthFrame",
        "SceneBoxSpec",
        "build_capture_request",
        "build_warehouse_scene_spec",
        "parse_camera_specs",
        "save_depth_capture_bundle",
    }:
        from .sapien_depth_adapter import (
            DepthCameraSpec,
            DepthCaptureBundle,
            DepthFrame,
            SceneBoxSpec,
            build_capture_request,
            build_warehouse_scene_spec,
            parse_camera_specs,
            save_depth_capture_bundle,
        )

        return {
            "DepthCameraSpec": DepthCameraSpec,
            "DepthCaptureBundle": DepthCaptureBundle,
            "DepthFrame": DepthFrame,
            "SceneBoxSpec": SceneBoxSpec,
            "build_capture_request": build_capture_request,
            "build_warehouse_scene_spec": build_warehouse_scene_spec,
            "parse_camera_specs": parse_camera_specs,
            "save_depth_capture_bundle": save_depth_capture_bundle,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
