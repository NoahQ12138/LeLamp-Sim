"""Derive a scene with one spotlight, positioned from the actual diffuser mesh."""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from test_sim import load_model

ROOT = Path(__file__).resolve().parent / "simulation"


def mesh_in_body(model, name):
    mesh_id = model.mesh(name).id
    geom_id = next(i for i in range(model.ngeom)
                   if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                   and model.geom_dataid[i] == mesh_id)
    start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
    vertices = model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T + model.geom_pos[geom_id]
    return int(model.geom_bodyid[geom_id]), vertices


def main():
    model = load_model()
    body_id, surface = mesh_in_body(model, "diffuser")
    housing_body, housing = mesh_in_body(model, "lamphead")
    assert body_id == housing_body
    center = (surface.min(0) + surface.max(0)) / 2
    housing_center = (housing.min(0) + housing.max(0)) / 2
    _, axes = np.linalg.eigh(np.cov(surface.T))
    direction = axes[:, 0]  # Thin direction is the diffuser's face normal.
    if np.dot(direction, center - housing_center) < 0:
        direction = -direction
    # Place 2 mm beyond the outermost diffuser surface to avoid self-shadowing.
    position = center + direction * (np.max((surface - center) @ direction) + 0.002)
    scene = ET.parse(ROOT / "scene.xml").getroot()
    include = scene.find("include")
    assert include is not None and include.get("file") == "robot.xml"
    robot = ET.parse(ROOT / "robot.xml").getroot()
    head = robot.find(f".//body[@name='{model.body(body_id).name}']")
    assert head is not None
    ET.SubElement(head, "light", {
        "name": "lamp_light", "pos": " ".join(f"{v:.10g}" for v in position),
        "dir": " ".join(f"{v:.10g}" for v in direction),
        "directional": "false", "diffuse": "3 3 3", "ambient": "0 0 0",
        "specular": "0.3 0.3 0.3", "cutoff": "40", "exponent": "4",
        "attenuation": "1 0 0", "castshadow": "true", "active": "true",
    })
    location = list(scene).index(include)
    scene.remove(include)
    for offset, child in enumerate(robot):
        scene.insert(location + offset, child)
    ET.indent(scene)
    ET.ElementTree(scene).write(ROOT / "scene_with_light.xml", encoding="utf-8", xml_declaration=True)
    print("Added lamp_light to body", model.body(body_id).name)
    print("Body-local position:", position, "outward direction:", direction)


if __name__ == "__main__":
    main()
