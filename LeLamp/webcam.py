"""Shared webcam setup; prefer the verified fast Windows DirectShow backend."""
import sys
import time
import cv2

CAMERA_INDEX = 0


def open_camera():
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    started = time.perf_counter()
    print(f"Opening webcam {CAMERA_INDEX} ({'DirectShow' if sys.platform == 'win32' else 'default backend'})...", flush=True)
    camera = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Cannot open webcam. Close other camera apps and enable Windows camera access for desktop apps. "
                           "If using a different camera, change CAMERA_INDEX in webcam.py.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Camera opened in {time.perf_counter()-started:.2f}s; waiting for frames.", flush=True)
    return camera
