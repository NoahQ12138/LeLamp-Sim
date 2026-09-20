# LeLamp lighting

The original model has one unnamed world light at `[0, 0, 3.5]`, directed
downward, plus a camera headlight. It has no robot-mounted light, and the
diffuser/lamphead materials have zero emission. The relevant geoms are unnamed;
their mesh and material names identify them. Run `inspect_lighting.py` for metadata.

`simulation/scene_with_light.xml` is a derived snapshot of the original scene and
robot, with exactly one added light. The original XML, meshes, materials, joints,
and physics are unchanged. If the upstream robot changes, regenerate the snapshot:

```powershell
.\.venv\Scripts\python.exe build_light_scene.py
```

The added `lamp_light` (compiled light ID 1) is a spotlight inside body `diffuser`,
which owns the actual `diffuser` and `lamphead` meshes. Its local position is
approximately `[-0.09963243, 0.08713535, 0.018625]` meters and its direction is
`[-0.75347837, 0.65747269, 0]`. The builder derives the diffuser plane normal from
its mesh vertices, orients it away from the housing, and places the source 2 mm
beyond the face. It follows the body as the robot moves.

Run the demo:

```powershell
cd C:\Users\noahq\OneDrive\Documents\HTN\LeLamp-Sim\LeLamp
.\.venv\Scripts\python.exe light_test.py
```

It shows OFF / ON / OFF / ON for one second each and leaves the viewer open.
The demo dims studio lighting in memory to make the illuminated floor obvious.
It holds existing joint targets at neutral. No material emission/color trick is
used: the spotlight illuminates other surfaces. There is no volumetric beam in air.

Use the API in your own viewer loop:

```python
from lamp_light import load_lit_model, LampLight

model = load_lit_model()
# Create data and launch your passive viewer using this same model.
# Inside the viewer context:
lamp = LampLight(model, viewer)
lamp.lamp_on()
lamp.set_lamp_brightness(0.5)
lamp.set_lamp_color(1.0, 0.7, 0.3)  # Warm RGB tint
lamp.lamp_off()
lamp.lamp_on()  # Restores the saved brightness and color
```

Brightness and RGB values are clamped to `[0, 1]`; non-finite input is rejected.
Brightness is a fraction of the source's full strength, not lumens. Setting
brightness alone preserves the on/off state. At zero brightness, `lamp_on()`
still produces no light until brightness is raised. Only construct one controller
per light; it captures the source's initial full-strength diffuse/specular values.

MuJoCo 3.13.0 exposes writable `light_active`, `light_diffuse`, and
`light_specular` arrays. The API edits these under the viewer lock and performs a
full `viewer.sync()` afterward. Continue your normal stepping/sync loop while
waiting between changes, as demonstrated in `light_test.py`.

Validation: `verify_lighting.py` compares 188 original physics/material arrays,
checks light attachment during head rotation, tests brightness/color clamping,
and renders OFF/ON images. Over 50,000 floor pixels brightened by more than
15/255 in the final test. It writes `lighting_off.png` and `lighting_on.png`.
The interactive OFF/ON/OFF/ON test also completed with physics checks passing.

Files added for this task: `inspect_lighting.py`, `build_light_scene.py`,
`lamp_light.py`, `light_test.py`, `verify_lighting.py`,
`simulation/scene_with_light.xml`, this guide, `lighting_inspection.txt`,
and the two comparison images. No pre-existing files were changed.

Reference: [MuJoCo light documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-light).
