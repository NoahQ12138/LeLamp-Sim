"""Visible HAPPY/ANGRY expressions from a pretrained FER+ face-crop classifier."""
from pathlib import Path
import time
import cv2
import numpy as np

EXPRESSION_EVERY_N_FRAMES = 3
EXPRESSION_CONFIDENCE = 0.65
EXPRESSION_MARGIN = 0.15
ANGRY_CONFIDENCE = 0.25
ANGRY_NEUTRAL_MAX_GAP = 0.15
MODEL_PATH = Path(__file__).resolve().parent / "models" / "emotion-ferplus-8.onnx"
# Class order from the ONNX Model Zoo FER+ model documentation.
CLASS_NAMES = ("neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt")


def classify_scores(logits):
    logits = np.asarray(logits, dtype=float).reshape(-1)
    if logits.shape != (8,) or not np.isfinite(logits).all():
        raise ValueError("FER+ must return eight finite scores")
    probabilities = np.exp(logits-logits.max())
    probabilities /= probabilities.sum()
    order = np.argsort(probabilities)
    best = int(order[-1])
    confidence = float(probabilities[best])
    # FER+ anger is class 4; sadness remains class 3 and is not a trigger.
    anger_score = float(probabilities[4])
    anger_candidate = best == 4 or (
        best == 0 and int(order[-2]) == 4
        and probabilities[0]-anger_score <= ANGRY_NEUTRAL_MAX_GAP
    )
    if anger_candidate and anger_score >= ANGRY_CONFIDENCE:
        return "ANGRY", anger_score
    label = "HAPPY" if best == 1 else "UNKNOWN"
    if confidence < EXPRESSION_CONFIDENCE or confidence-probabilities[order[-2]] < EXPRESSION_MARGIN:
        label = "UNKNOWN"
    return label, confidence


class ExpressionDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
        self.frame_index = 0
        self.label = "UNKNOWN"
        self.confidence = 0.0
        self.last_box = None

    def process(self, frame, faces):
        self.frame_index += 1
        if not faces:
            self.label, self.confidence, self.last_box = "UNKNOWN", 0.0, None
            return {"HAPPY": False, "ANGRY": False}
        box = max(faces, key=lambda face: face.bounding_box.width*face.bounding_box.height).bounding_box
        height, width = frame.shape[:2]
        size = max(box.width, box.height)*1.15
        cx, cy = box.origin_x+box.width/2, box.origin_y+box.height/2
        x1, y1 = max(0,int(cx-size/2)), max(0,int(cy-size/2))
        x2, y2 = min(width,int(cx+size/2)), min(height,int(cy+size/2))
        if x2-x1 < 32 or y2-y1 < 32:
            self.label, self.confidence = "UNKNOWN", 0.0
            return {"HAPPY": False, "ANGRY": False}
        self.last_box = (x1,y1,x2,y2)
        if self.frame_index % EXPRESSION_EVERY_N_FRAMES:
            return {"HAPPY": None, "ANGRY": None}  # Cached label is display-only.
        crop = cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY)
        # Model preprocessing: raw grayscale intensities, 1x1x64x64 float32.
        self.net.setInput(cv2.dnn.blobFromImage(crop, scalefactor=1.0, size=(64,64)))
        self.label, self.confidence = classify_scores(self.net.forward())
        return {"HAPPY": self.label == "HAPPY", "ANGRY": self.label == "ANGRY"}


def main():
    import argparse
    from vision import Vision
    from webcam import open_camera
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    faces = Vision("face")
    camera = None
    counts = {"HAPPY":0,"ANGRY":0,"UNKNOWN":0}
    try:
        detector = ExpressionDetector()
        camera = open_camera()
        until = time.monotonic()+args.seconds
        while time.monotonic() < until:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames")
            output, _ = faces.process(frame)
            result = detector.process(frame, faces.face_boxes)
            if faces.face_boxes and result["HAPPY"] is not None:
                counts[detector.label] += 1
            cv2.rectangle(output,(0,0),(output.shape[1],70),(20,20,20),-1)
            cv2.putText(output, f"Expression: {detector.label} ({detector.confidence:.2f})", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
            cv2.putText(output, "Visible expression only | Try a smile or frown | q: quit",(10,55),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
            cv2.imshow("LeLamp Expression Test", output)
            if cv2.waitKey(1)&0xFF == ord("q"):
                break
        print(f"Expression inference results: {counts}", flush=True)
        if sum(counts.values()) == 0:
            raise RuntimeError("No visible face was available for expression inference")
    finally:
        if camera is not None:
            camera.release()
        faces.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
