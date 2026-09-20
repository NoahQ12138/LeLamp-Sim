"""Peace-sign geometry using MediaPipe's 21 metric hand landmarks."""
import math
import numpy as np

EXTENDED_ANGLE = 150.0
CURLED_ANGLE = 145.0
MIN_V_SPREAD = 0.35  # Index/middle tip separation relative to palm width.
POINT_DIRECTIONS = ("LEFT", "RIGHT", "UP", "DOWN")


def pointing_direction(world_landmarks, image_landmarks, width, height):
    """Index extended, other fingers curled; direction in the unmirrored preview."""
    if len(world_landmarks) != 21 or len(image_landmarks) != 21:
        return None
    world = np.array([[p.x, p.y, p.z] for p in world_landmarks], dtype=float)
    pixels = np.array([[p.x*width, p.y*height] for p in image_landmarks], dtype=float)
    if not np.isfinite(world).all() or not np.isfinite(pixels).all():
        return None
    mcp, pip, dip, tip = world[5:9]
    if (joint_angle(mcp, pip, dip) < EXTENDED_ANGLE or
            joint_angle(pip, dip, tip) < EXTENDED_ANGLE or
            np.linalg.norm(tip-world[0]) <= 1.15*np.linalg.norm(pip-world[0])):
        return None
    for base in (9, 13, 17):
        mcp, pip, dip, tip = world[base:base+4]
        if not (joint_angle(mcp, pip, dip) < CURLED_ANGLE or
                np.linalg.norm(tip-world[0]) < np.linalg.norm(pip-world[0])):
            return None
    palm = np.linalg.norm(pixels[5]-pixels[17])
    dx, dy = pixels[8]-pixels[5]
    if palm < 1e-6 or np.hypot(dx, dy) < 0.65*palm:
        return None
    if abs(dx) > 1.3*abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    if abs(dy) > 1.3*abs(dx):
        return "DOWN" if dy > 0 else "UP"
    return None  # Ambiguous diagonals do not select a direction.


def joint_angle(a, b, c):
    first, second = a-b, c-b
    denominator = np.linalg.norm(first)*np.linalg.norm(second)
    if denominator < 1e-10:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(first, second)/denominator, -1, 1))))


def is_peace_sign(landmarks):
    """Index/middle extended and separated, ring/pinky curled; thumb unrestricted.

    Distances and angles use world landmarks, so handedness and image rotation
    do not change the rule. This is a geometric heuristic, not a trained classifier.
    """
    if len(landmarks) != 21:
        return False
    points = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=float)
    if not np.isfinite(points).all():
        return False
    palm = np.linalg.norm(points[5]-points[17])
    if palm < 1e-6:
        return False

    def extended(base):
        mcp, pip, dip, tip = points[base:base+4]
        return (joint_angle(mcp, pip, dip) >= EXTENDED_ANGLE
                and joint_angle(pip, dip, tip) >= EXTENDED_ANGLE
                and np.linalg.norm(tip-points[0]) > 1.15*np.linalg.norm(pip-points[0]))

    def curled(base):
        mcp, pip, dip, tip = points[base:base+4]
        return (joint_angle(mcp, pip, dip) < CURLED_ANGLE
                or np.linalg.norm(tip-points[0]) < np.linalg.norm(pip-points[0]))

    return bool(extended(5) and extended(9) and curled(13) and curled(17)
                and np.linalg.norm(points[8]-points[12]) / palm >= MIN_V_SPREAD)
