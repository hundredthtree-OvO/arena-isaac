import math

import isaac_utils.utils.paths as Paths
import numpy as np
from isaacsim.core.utils.prims import get_prim_at_path
from omni.usd import get_context
from rclpy.qos import QoSProfile

from isaacsim_msgs.srv import GetPrimAttributes

from .utils import safe

profile = QoSProfile(depth=2000)


def _split_hint(path_hint: str) -> tuple[str | None, str]:
    text = str(path_hint or "").strip()
    if not text:
        return None, ""
    if "/" not in text.strip("/"):
        return None, text.strip("/")
    stripped = text.rstrip("/")
    leaf = stripped.split("/")[-1]
    parent = stripped[: -(len(leaf) + 1)] if "/" in stripped else ""
    return parent or None, leaf


def _find_descendant_named(root_path: str | None, leaf_name: str):
    leaf = str(leaf_name or "").strip().lstrip("/")
    if not leaf:
        return None
    stage = get_context().get_stage()
    if stage is None:
        return None

    if root_path:
        direct = stage.GetPrimAtPath(f"{root_path.rstrip('/')}/{leaf}")
        if direct and direct.IsValid():
            return direct

    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        path = prim.GetPath().pathString
        if root_path and not path.startswith(root_path.rstrip("/") + "/"):
            continue
        if prim.GetName() == leaf:
            return prim
    return None


def _resolve_prim(path_hint: str):
    text = str(path_hint or "").strip()
    candidates = []
    if text.startswith("/"):
        candidates.append(text)
    elif text:
        candidates.append(Paths.scene.path(text))

    for candidate in candidates:
        prim = get_prim_at_path(candidate)
        if prim and prim.IsValid():
            return prim

    root_hint, leaf = _split_hint(text)
    search_roots = []
    if root_hint:
        search_roots.append(root_hint)
    if text.startswith("/World/") and root_hint:
        search_roots.append(root_hint.rstrip("/"))
    search_roots.append(None)

    for root in search_roots:
        prim = _find_descendant_named(root, leaf)
        if prim and prim.IsValid():
            return prim
    return None


@safe
def get_prim_attributes(request, response):
    prim = _resolve_prim(request.prim_path)
    if prim is None or not prim.IsValid():
        response.pose.position.x = math.nan
        response.pose.position.y = math.nan
        response.pose.position.z = math.nan
        response.pose.orientation.w = math.nan
        response.pose.orientation.x = math.nan
        response.pose.orientation.y = math.nan
        response.pose.orientation.z = math.nan
        raise RuntimeError(f"Could not resolve prim path for hint: {request.prim_path}")

    translate = np.array(prim.GetAttribute("xformOp:translate").Get(), dtype=np.float32)
    quat = prim.GetAttribute("xformOp:orient").Get()

    response.pose.position.x = float(translate[0])
    response.pose.position.y = float(translate[1])
    response.pose.position.z = float(translate[2])

    if quat is None:
        response.pose.orientation.w = 1.0
        response.pose.orientation.x = 0.0
        response.pose.orientation.y = 0.0
        response.pose.orientation.z = 0.0
    else:
        response.pose.orientation.w = float(quat.real)
        response.pose.orientation.x = float(quat.imaginary[0])
        response.pose.orientation.y = float(quat.imaginary[1])
        response.pose.orientation.z = float(quat.imaginary[2])

    scale = prim.GetAttribute("xformOp:scale").Get()
    if scale is not None:
        response.scale = np.array(scale, dtype=np.float32).tolist()

    return response


def get_prim_attr(controller):
    service = controller.create_service(
        srv_type=GetPrimAttributes,
        qos_profile=profile,
        srv_name='isaac/get_prim_attributes',
        callback=get_prim_attributes
    )
    return service
