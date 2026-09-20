"""Inspect existing lights and lamp-related geometry without changing the model."""
import mujoco
import sys
from test_sim import load_model


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    model = load_model()
    print(f"MuJoCo {mujoco.__version__}; model.nlight = {model.nlight}")
    print("Available light fields:", ", ".join(k for k in dir(model) if k.startswith("light_")))
    for i in range(model.nlight):
        print(f"Light ID {i}, name={model.light(i).name!r}, body={model.body(int(model.light_bodyid[i])).name!r}")
        for field in ("pos", "dir", "diffuse", "ambient", "active"):
            values = getattr(model, "light_" + field, None)
            print(f"  {field}: {values[i] if values is not None else 'unavailable'}")
    print("Camera headlight (not a robot lamp):", model.vis.headlight)
    print("Relevant geometry (unnamed geoms identified by actual body/mesh/material):")
    for i in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[i])).name
        mesh_id = int(model.geom_dataid[i])
        mesh = model.mesh(mesh_id).name if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH else ""
        material_id = int(model.geom_matid[i])
        material = model.material(material_id).name if material_id >= 0 else ""
        name = model.geom(i).name
        if any(token in f"{name} {body} {mesh} {material}".lower() for token in ("lamp", "head", "led", "diffuser")):
            print(f"  geom {i}: name={name!r}, body={body!r}, mesh={mesh!r}, material={material!r}")
            if material_id >= 0:
                print(f"    rgba={model.mat_rgba[material_id]}, emission={model.mat_emission[material_id]}")


if __name__ == "__main__":
    main()
