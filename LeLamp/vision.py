"""Local webcam detection; no camera frames are saved or uploaded."""
import argparse
from pathlib import Path
import time
import queue
import threading

import cv2
import mediapipe as mp
from gestures import is_peace_sign, pointing_direction, POINT_DIRECTIONS
from webcam import open_camera

MODEL_DIR = Path(__file__).resolve().parent / "models"
FACE_CONFIDENCE = 0.5
HAND_CONFIDENCE = 0.5
PHONE_CONFIDENCE = 0.50
YOLO_EVERY_N_FRAMES = 2
HAND_CONNECTIONS = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
                    (15,16),(13,17),(0,17),(17,18),(18,19),(19,20))


class Vision:
    def __init__(self, stage="face", expressions_enabled=False):
        self.stage = stage
        self.face = mp.tasks.vision.FaceDetector.create_from_options(
            mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_DIR / "blaze_face_short_range.tflite")),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                min_detection_confidence=FACE_CONFIDENCE,
            ))
        self.last_timestamp = -1
        self.face_boxes = []
        self.hands = None
        self.phone_model = None
        self.phone_boxes = []
        self.frame_number = 0
        self.expression = None
        try:
            if expressions_enabled:
                from expressions import ExpressionDetector
                self.expression = ExpressionDetector()
            if stage != "face":
                self.hands = mp.tasks.vision.HandLandmarker.create_from_options(
                    mp.tasks.vision.HandLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_DIR / "hand_landmarker.task")),
                        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
                        min_hand_detection_confidence=HAND_CONFIDENCE,
                        min_hand_presence_confidence=HAND_CONFIDENCE, min_tracking_confidence=HAND_CONFIDENCE))
            if stage in ("phone", "all"):
                from ultralytics import YOLO
                import torch
                torch.set_num_threads(2)
                self.phone_model = YOLO(str(MODEL_DIR / "yolo11n.pt"))
                self.phone_class = next(i for i, name in self.phone_model.names.items() if name == "cell phone")
                print(f"YOLO11n class {self.phone_class}: cell phone; confidence >= {PHONE_CONFIDENCE}", flush=True)
        except Exception:
            self.close()
            raise

    def process(self, frame):
        stamp = max(self.last_timestamp + 1, int(time.monotonic() * 1000))
        self.last_timestamp = stamp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = self.face.detect_for_video(image, stamp)
        self.face_boxes = faces.detections
        output = frame.copy()
        for detection in faces.detections:
            box = detection.bounding_box
            x, y, w, h = box.origin_x, box.origin_y, box.width, box.height
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output, "FACE DETECTED", (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        detections = {"FACE": bool(faces.detections)}
        if self.expression is not None:
            detections.update(self.expression.process(frame, faces.detections))
        if self.hands is not None:
            hands = self.hands.detect_for_video(image, stamp)
            peace = [is_peace_sign(hand) for hand in hands.hand_world_landmarks]
            height, width = frame.shape[:2]
            directions = [pointing_direction(world, image, width, height)
                          for world, image in zip(hands.hand_world_landmarks, hands.hand_landmarks)]
            unique = {direction for direction in directions if direction is not None}
            detections["PEACE"] = any(peace)
            for direction in POINT_DIRECTIONS:
                detections["POINT_"+direction] = unique == {direction} and not any(peace)
            detections["HAND"] = bool(hands.hand_landmarks) and not any(peace) and not unique
            for hand_index, landmarks in enumerate(hands.hand_landmarks):
                points = [(int(p.x*width), int(p.y*height)) for p in landmarks]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(output, points[a], points[b], (255,200,0), 2)
                for point in points:
                    cv2.circle(output, point, 3, (0,255,255), -1)
                label = ("PEACE SIGN" if peace[hand_index] else
                         "POINT " + directions[hand_index] if directions[hand_index] else "GENERIC HAND")
                cv2.putText(output, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        if self.phone_model is not None:
            # None means no new evidence; cached predictions must not count as
            # consecutive phone detections for confirmation or rearming.
            detections["PHONE"] = None
            if self.frame_number % YOLO_EVERY_N_FRAMES == 0:
                result = self.phone_model.predict(frame, classes=[self.phone_class],
                    conf=PHONE_CONFIDENCE, imgsz=416, device="cpu", verbose=False)[0]
                self.phone_boxes = [(box.xyxy[0].tolist(), float(box.conf[0])) for box in result.boxes]
                detections["PHONE"] = bool(self.phone_boxes)
            for (x1, y1, x2, y2), confidence in self.phone_boxes:
                cv2.rectangle(output, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
                cv2.putText(output, f"PHONE {confidence:.2f}", (int(x1),max(20,int(y1)-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        self.frame_number += 1
        return output, detections

    def close(self):
        if self.hands is not None:
            self.hands.close()
        self.face.close()


class VisionWorker(threading.Thread):
    """Owns camera and detectors; publishes results, never accesses MuJoCo."""
    def __init__(self, stage="all"):
        super().__init__(name="LeLamp vision", daemon=True)
        self.stage = stage
        self.stop_requested = threading.Event()
        self.results = queue.Queue(maxsize=2)
        self.error = None
        self.status = "Opening camera..."
        self.expression_label = "UNKNOWN"
        self.expression_confidence = 0.0

    def run(self):
        camera = None
        detector = None
        try:
            camera = open_camera()
            self.status = "Loading face/gesture/phone detectors..."
            detector = Vision(self.stage, expressions_enabled=True)
            self.status = "Waiting for first camera frame..."
            frames = 0
            while not self.stop_requested.is_set():
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Webcam stopped delivering frames.")
                captured = time.monotonic()
                output, detections = detector.process(frame)
                self.expression_label = detector.expression.label
                self.expression_confidence = detector.expression.confidence
                frames += 1
                if frames == 1:
                    print("Camera LIVE: first frame processed.", flush=True)
                self.status = f"Camera LIVE | frames: {frames}"
                packet = (captured, output, detections)
                try:
                    self.results.put_nowait(packet)
                except queue.Full:
                    try:
                        self.results.get_nowait()
                    except queue.Empty:
                        pass
                    self.results.put_nowait(packet)
        except Exception as error:
            self.error = error
            self.status = f"Camera error: {error}"
        finally:
            if camera is not None:
                camera.release()
            if detector is not None:
                detector.close()

    def close(self):
        self.stop_requested.set()
        self.join(timeout=10)
        if self.is_alive():
            print("Camera driver is slow to stop; vision worker will release it when its current call returns.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("face", "peace", "hand", "phone"), default="face")
    parser.add_argument("--seconds", type=float, default=15)
    args = parser.parse_args()
    detector = Vision(args.stage)
    camera = None
    counts = {}
    frames = 0
    try:
        camera = open_camera()
        if not camera.isOpened():
            raise RuntimeError("Cannot open webcam 0.")
        until = time.monotonic() + args.seconds
        while time.monotonic() < until:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera frame capture failed.")
            output, detections = detector.process(frame)
            frames += 1
            for name, present in detections.items():
                counts[name] = counts.get(name, 0) + int(bool(present))
            cv2.putText(output, f"Stage: {args.stage} | q: quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow("LeLamp Vision", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        print(f"Processed {frames} frames; positive detection counts: {counts}", flush=True)
        key = args.stage.upper()
        if not counts.get(key, 0):
            raise RuntimeError(f"No {args.stage} detected; this stage is not yet verified.")
    finally:
        if camera is not None:
            camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
