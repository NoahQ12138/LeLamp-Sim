"""Stage 1: show webcam 0 for ten seconds; q exits. No frames are saved."""
import time
import cv2
from webcam import open_camera


def main():
    camera = open_camera()
    frames = 0
    try:
        if not camera.isOpened():
            raise RuntimeError("Cannot open webcam 0. Check Windows camera permissions and close other camera apps.")
        deadline = time.perf_counter() + 10
        while time.perf_counter() < deadline:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Webcam opened but failed to deliver a frame.")
            frames += 1
            cv2.putText(frame, "LeLamp webcam check - q to quit", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("LeLamp Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        print(f"PASS: webcam delivered {frames} frames.", flush=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
