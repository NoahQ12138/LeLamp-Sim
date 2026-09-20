# LeLamp-Sim

HTN2026

Windows MuJoCo simulation of Human Computer Lab's LeLamp, with webcam gestures,
expression reactions, voice commands, speech, online jokes, and idle lighting.

- [All actions and states (plain text)](ACTIONS_AND_STATES.txt)
- [Code structure and simulation architecture](ARCHITECTURE.md)
- [Setup and interaction guide](LeLamp/INTERACTIONS.md)
- [Original LeLamp project and attribution](LeLamp/README.md)

From the repository root, with the project environment and models installed:

```powershell
cd LeLamp
.\.venv\Scripts\python.exe vision_behavior.py
```

Python environments and downloaded model weights are excluded from Git.
