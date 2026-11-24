"""Add leg collision geometry and filtered pairs to HU_D04_01.usd.

Run this script ONCE inside Isaac Sim to generate a USD with leg collision
capsules.  The original file is backed up as HU_D04_01.usd.bak.

Usage (inside Isaac Sim environment):
    python scripts/add_usd_collision.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import isaacsim  # noqa: F401 – must be first import to bootstrap Isaac Sim
from isaacsim import SimulationApp

# Start Isaac Sim in headless mode so pxr/UsdPhysics are available.
app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROBOT_DIR = Path(__file__).resolve().parents[1] / "source" / "limx_rl_lab" / "data" / "Robots" / "limx" / "HU_D04_description"
_USD_PATH = _ROBOT_DIR / "usd" / "HU_D04_01.usd"

# ---------------------------------------------------------------------------
# Collision capsule definitions (copied from MuJoCo XML)
#
# MuJoCo capsule size = [radius, half_length]
# MuJoCo euler = extrinsic XYZ (same as USD Gf.Rotation order when applied
# as X * Y * Z).
# ---------------------------------------------------------------------------
_CAPSULE_DEFS: list[dict] = [
    {
        "link_name": "left_hip_yaw_link",
        "offset_pos": (0.015, 0.0, -0.0331),
        "radius": 0.05,
        "half_length": 0.1,
        "euler_deg": (0.0, -0.1, 0.0),
    },
    {
        "link_name": "right_hip_yaw_link",
        "offset_pos": (0.015, 0.0, -0.0331),
        "radius": 0.05,
        "half_length": 0.1,
        "euler_deg": (0.0, -0.1, 0.0),
    },
    {
        "link_name": "left_knee_link",
        "offset_pos": (0.015, 0.0, -0.2),
        "radius": 0.035,
        "half_length": 0.14,
        "euler_deg": (0.0, 0.1, 0.0),
    },
    {
        "link_name": "right_knee_link",
        "offset_pos": (0.015, 0.0, -0.2),
        "radius": 0.035,
        "half_length": 0.14,
        "euler_deg": (0.0, 0.1, 0.0),
    },
]

# ---------------------------------------------------------------------------
# Filtered pairs – same-leg adjacent links that MUST NOT collide.
# Each tuple is (body1_name, body2_name).
# ---------------------------------------------------------------------------
_FILTERED_PAIRS: list[tuple[str, str]] = [
    ("left_hip_roll_link", "left_hip_yaw_link"),
    ("left_hip_yaw_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("right_hip_roll_link", "right_hip_yaw_link"),
    ("right_hip_yaw_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
]


def _euler_to_quat_wxyz(rx_deg: float, ry_deg: float, rz_deg: float) -> Gf.Quatf:
    """Convert extrinsic-XYZ Euler angles (degrees) to a wxyz quaternion."""
    import math

    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return Gf.Quatf(w, x, y, z)


def _find_link_prim(stage: Usd.Stage, link_name: str) -> Usd.Prim:
    """Find a link prim by searching for a prim whose name ends with the link name."""
    for prim in stage.Traverse():
        if prim.GetName() == link_name:
            return prim
    raise ValueError(f"Link '{link_name}' not found in USD")


def add_capsule_collision(stage: Usd.Stage, link_prim: Usd.Prim, cfg: dict) -> Usd.Prim:
    """Add a collision capsule as a child of *link_prim*."""
    capsule_name = f"{cfg['link_name']}_collision"
    capsule_path = link_prim.GetPath().AppendChild(capsule_name)

    # Create the capsule mesh prim
    capsule_prim = stage.DefinePrim(capsule_path, "Capsule")
    capsule_api = UsdGeom.Capsule.Define(stage, capsule_path)

    # Set radius and height (full height = 2 * half_length in MuJoCo)
    capsule_api.CreateRadiusAttr().Set(cfg["radius"])
    capsule_api.CreateHeightAttr().Set(cfg["half_length"] * 2)

    # Set local transform (translate + rotate)
    xformable = UsdGeom.Xformable(capsule_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*cfg["offset_pos"]))
    xformable.AddOrientOp().Set(_euler_to_quat_wxyz(*cfg["euler_deg"]))

    # Enable collision
    UsdPhysics.CollisionAPI.Apply(capsule_prim)

    # Make it invisible (collision-only)
    capsule_prim.GetAttribute("purpose").Set("guide")

    print(f"  Added capsule collision: {capsule_path}")
    return capsule_prim


def add_filtered_pairs(stage: Usd.Stage, pairs: list[tuple[str, str]]) -> None:
    """Add PhysicsFilteredPairsAPI to exclude same-leg adjacent link collisions."""
    for body1_name, body2_name in pairs:
        prim1 = _find_link_prim(stage, body1_name)
        prim2 = _find_link_prim(stage, body2_name)

        # Apply FilteredPairsAPI to body1
        filtered_api = UsdPhysics.FilteredPairsAPI.Apply(prim1)
        rel = filtered_api.CreateFilteredPairsRel()
        rel.AddTarget(prim2.GetPath())

        print(f"  Filtered pair: {body1_name} <-> {body2_name}")


def main() -> None:
    usd_path = str(_USD_PATH)
    bak_path = usd_path + ".bak"

    # Backup
    print(f"Backing up {usd_path} -> {bak_path}")
    shutil.copy2(usd_path, bak_path)

    # Open stage
    print(f"Opening {usd_path}")
    stage = Usd.Stage.Open(usd_path)

    # Add capsule collisions
    print("\nAdding capsule collision geometry:")
    for cfg in _CAPSULE_DEFS:
        link_prim = _find_link_prim(stage, cfg["link_name"])
        add_capsule_collision(stage, link_prim, cfg)

    # Add filtered pairs
    print("\nAdding collision filtered pairs:")
    add_filtered_pairs(stage, _FILTERED_PAIRS)

    # Save
    print(f"\nSaving modified USD to {usd_path}")
    stage.GetRootLayer().Save()
    print("Done!")

    # Verify
    print("\nVerification - collision prims in modified USD:")
    stage2 = Usd.Stage.Open(usd_path)
    count = 0
    for prim in stage2.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            count += 1
            print(f"  {prim.GetPath()}")
    print(f"Total collision prims: {count}")


if __name__ == "__main__":
    main()
    app.close()
