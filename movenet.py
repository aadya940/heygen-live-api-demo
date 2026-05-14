"""
Video stream → MoveNet → ContextEngine pipeline.

Three layers, clearly separated:

  VideoStream       — camera / file I/O, yields BGR frames
  MoveNetInference  — keypoint extraction from frames
  ContextEngine     — packages (frame + keypoints) into LLM-ready context

The LLM receives a vision message containing:
  - the annotated frame as a base64 JPEG
  - structured keypoint data as text
  - a configurable system prompt describing the task
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Generator, Optional

import cv2
import numpy as np

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

try:
    import tflite_runtime.interpreter as tflite
    _TFLITE_AVAILABLE = True
except ImportError:
    _TFLITE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Keypoint schema (COCO-17, MoveNet order)
# ---------------------------------------------------------------------------

class KP(IntEnum):
    NOSE           = 0
    LEFT_EYE       = 1
    RIGHT_EYE      = 2
    LEFT_EAR       = 3
    RIGHT_EAR      = 4
    LEFT_SHOULDER  = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW     = 7
    RIGHT_ELBOW    = 8
    LEFT_WRIST     = 9
    RIGHT_WRIST    = 10
    LEFT_HIP       = 11
    RIGHT_HIP      = 12
    LEFT_KNEE      = 13
    RIGHT_KNEE     = 14
    LEFT_ANKLE     = 15
    RIGHT_ANKLE    = 16


KP_NAMES: dict[KP, str] = {
    KP.NOSE:           "nose",
    KP.LEFT_EYE:       "left_eye",
    KP.RIGHT_EYE:      "right_eye",
    KP.LEFT_EAR:       "left_ear",
    KP.RIGHT_EAR:      "right_ear",
    KP.LEFT_SHOULDER:  "left_shoulder",
    KP.RIGHT_SHOULDER: "right_shoulder",
    KP.LEFT_ELBOW:     "left_elbow",
    KP.RIGHT_ELBOW:    "right_elbow",
    KP.LEFT_WRIST:     "left_wrist",
    KP.RIGHT_WRIST:    "right_wrist",
    KP.LEFT_HIP:       "left_hip",
    KP.RIGHT_HIP:      "right_hip",
    KP.LEFT_KNEE:      "left_knee",
    KP.RIGHT_KNEE:     "right_knee",
    KP.LEFT_ANKLE:     "left_ankle",
    KP.RIGHT_ANKLE:    "right_ankle",
}

SKELETON_EDGES = [
    (KP.NOSE, KP.LEFT_EYE),  (KP.NOSE, KP.RIGHT_EYE),
    (KP.LEFT_EYE, KP.LEFT_EAR), (KP.RIGHT_EYE, KP.RIGHT_EAR),
    (KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER),
    (KP.LEFT_SHOULDER, KP.LEFT_ELBOW),   (KP.LEFT_ELBOW, KP.LEFT_WRIST),
    (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW), (KP.RIGHT_ELBOW, KP.RIGHT_WRIST),
    (KP.LEFT_SHOULDER, KP.LEFT_HIP),     (KP.RIGHT_SHOULDER, KP.RIGHT_HIP),
    (KP.LEFT_HIP, KP.RIGHT_HIP),
    (KP.LEFT_HIP, KP.LEFT_KNEE),   (KP.LEFT_KNEE, KP.LEFT_ANKLE),
    (KP.RIGHT_HIP, KP.RIGHT_KNEE), (KP.RIGHT_KNEE, KP.RIGHT_ANKLE),
]


@dataclass
class Keypoints:
    """Pose output for one person: 17 keypoints with (y_norm, x_norm, confidence)."""
    data: np.ndarray  # shape (17, 3)

    def confidence(self, kp: KP) -> float:
        return float(self.data[kp, 2])

    def xy_norm(self, kp: KP) -> tuple[float, float]:
        """Normalised (x, y) in [0, 1]."""
        return float(self.data[kp, 1]), float(self.data[kp, 0])

    def xy_px(self, kp: KP, frame_wh: tuple[int, int]) -> tuple[int, int]:
        w, h = frame_wh
        x, y = self.xy_norm(kp)
        return int(x * w), int(y * h)

    def visible(self, kp: KP, threshold: float = 0.3) -> bool:
        return self.confidence(kp) >= threshold

    def to_dict(self, conf_threshold: float = 0.3) -> dict[str, dict]:
        """Serialisable keypoint map for LLM context."""
        return {
            KP_NAMES[kp]: {
                "x": round(self.data[kp, 1], 4),
                "y": round(self.data[kp, 0], 4),
                "confidence": round(self.data[kp, 2], 3),
                "visible": self.visible(kp, conf_threshold),
            }
            for kp in KP
        }


# ---------------------------------------------------------------------------
# Layer 1 — Video Stream I/O
# ---------------------------------------------------------------------------

class VideoStream:
    """
    Context manager wrapping an OpenCV capture source.

    Yields BGR frames and provides display helpers.

    Usage:
        with VideoStream(0) as vs:
            for frame in vs.frames():
                vs.show(frame)
                if vs.key_pressed("q"):
                    break
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int  = 640,
        height: int = 480,
        fps: int    = 30,
        window_name: str = "MoveNet",
    ):
        self.source      = source
        self.width       = width
        self.height      = height
        self.fps         = fps
        self.window_name = window_name
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "VideoStream":
        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source!r}")
        return self

    def __exit__(self, *_) -> None:
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield BGR frames until the source ends or read fails."""
        assert self._cap is not None, "Use VideoStream as a context manager."
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self.window_name, frame)

    def key_pressed(self, key: str) -> bool:
        return cv2.waitKey(1) & 0xFF == ord(key)

    @property
    def frame_size(self) -> tuple[int, int]:
        assert self._cap is not None
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )


# ---------------------------------------------------------------------------
# Layer 2 — MoveNet Inference
# ---------------------------------------------------------------------------

class MoveNetInference:
    """
    Single-person pose estimation via MoveNet.

    Two backends:
      MoveNetInference.from_hub("lightning")         # TensorFlow Hub
      MoveNetInference.from_tflite("model.tflite")   # TFLite (edge)

    Usage:
        model = MoveNetInference.from_hub()
        kp    = model.predict(bgr_frame)
        frame = model.draw_skeleton(bgr_frame, kp)
    """

    _INPUT_SIZE = {"lightning": 192, "thunder": 256}
    _HUB_URLS   = {
        "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
        "thunder":   "https://tfhub.dev/google/movenet/singlepose/thunder/4",
    }

    def __init__(self, infer_fn: Callable[[np.ndarray], np.ndarray], input_size: int):
        self._infer      = infer_fn
        self._input_size = input_size

    @classmethod
    def from_hub(cls, variant: str = "lightning") -> "MoveNetInference":
        if not _TF_AVAILABLE:
            raise ImportError("tensorflow and tensorflow_hub required.")
        module  = hub.load(cls._HUB_URLS[variant])
        movenet = module.signatures["serving_default"]

        def infer(rgb: np.ndarray) -> np.ndarray:
            t = tf.cast(tf.expand_dims(rgb, 0), tf.int32)
            return movenet(t)["output_0"].numpy()

        return cls(infer, cls._INPUT_SIZE[variant])

    @classmethod
    def from_tflite(cls, model_path: str, variant: str = "lightning") -> "MoveNetInference":
        if _TFLITE_AVAILABLE:
            interp = tflite.Interpreter(model_path=model_path)
        elif _TF_AVAILABLE:
            interp = tf.lite.Interpreter(model_path=model_path)
        else:
            raise ImportError("tflite_runtime or tensorflow required.")

        interp.allocate_tensors()
        in_idx  = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]

        def infer(rgb: np.ndarray) -> np.ndarray:
            interp.set_tensor(in_idx, np.expand_dims(rgb, 0).astype(np.int32))
            interp.invoke()
            return interp.get_tensor(out_idx)

        return cls(infer, cls._INPUT_SIZE[variant])

    def predict(self, bgr_frame: np.ndarray) -> Keypoints:
        """Run inference on a BGR frame (OpenCV default). Returns Keypoints."""
        rgb     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._input_size, self._input_size))
        raw     = self._infer(resized)   # (1, 1, 17, 3)
        return Keypoints(data=raw[0, 0])

    def draw_skeleton(
        self,
        frame: np.ndarray,
        kp: Keypoints,
        conf_threshold: float = 0.3,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Return a copy of frame with skeleton overlay."""
        out  = frame.copy()
        h, w = out.shape[:2]
        wh   = (w, h)
        for a, b in SKELETON_EDGES:
            if kp.visible(a, conf_threshold) and kp.visible(b, conf_threshold):
                cv2.line(out, kp.xy_px(a, wh), kp.xy_px(b, wh), color, 2)
        for k in KP:
            if kp.visible(k, conf_threshold):
                cv2.circle(out, kp.xy_px(k, wh), 4, (0, 0, 255), -1)
        return out


# ---------------------------------------------------------------------------
# Layer 3 — Context Engine
# ---------------------------------------------------------------------------

@dataclass
class FrameContext:
    """
    One packaged snapshot ready to send to an LLM.

    image_b64   — base64-encoded JPEG of the (optionally annotated) frame
    keypoints   — serialised keypoint dict from Keypoints.to_dict()
    timestamp   — capture time (seconds since epoch)
    exercise    — optional exercise label provided by the user
    """
    image_b64:  str
    keypoints:  dict[str, dict]
    timestamp:  float
    exercise:   Optional[str] = None

    def to_anthropic_messages(self, system_prompt: str = "") -> list[dict]:
        """
        Return a list of Anthropic API message dicts (user turn) ready for
        client.messages.create(messages=...).

        Includes:
          - base64 image block (vision)
          - text block with keypoint data + exercise label
        """
        kp_lines = "\n".join(
            f"  {name}: x={v['x']:.3f}, y={v['y']:.3f}, "
            f"conf={v['confidence']:.2f}, visible={v['visible']}"
            for name, v in self.keypoints.items()
            if v["visible"]
        )
        exercise_line = f"Exercise: {self.exercise}\n" if self.exercise else ""
        text = (
            f"{exercise_line}"
            f"Timestamp: {self.timestamp:.2f}\n\n"
            f"Visible keypoints (normalised 0-1 coordinates):\n{kp_lines}"
        )
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       self.image_b64,
                        },
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]


class ContextEngine:
    """
    Converts a (frame, Keypoints) pair into a FrameContext for the LLM.

    Handles:
      - JPEG encoding + base64 of the frame
      - Throttling so you don't flood the LLM on every camera frame
      - Optional skeleton overlay on the image sent to the LLM

    Usage:
        engine  = ContextEngine(exercise="squat", send_interval=1.0)
        context = engine.build(frame, keypoints)
        if context:                             # None when throttled
            messages = context.to_anthropic_messages()
            # ... call LLM
    """

    def __init__(
        self,
        exercise: Optional[str]  = None,
        send_interval: float     = 1.0,     # seconds between LLM calls
        jpeg_quality: int        = 75,
        conf_threshold: float    = 0.3,
        annotate_image: bool     = True,    # draw skeleton on the image sent to LLM
    ):
        self.exercise       = exercise
        self.send_interval  = send_interval
        self.jpeg_quality   = jpeg_quality
        self.conf_threshold = conf_threshold
        self.annotate_image = annotate_image
        self._last_sent: float = 0.0

    def build(
        self,
        frame: np.ndarray,
        kp: Keypoints,
        model: Optional[MoveNetInference] = None,
    ) -> Optional[FrameContext]:
        """
        Build a FrameContext from the current frame and keypoints.

        Returns None if called before send_interval has elapsed since the
        last successful build (throttle). Pass `model` to draw the skeleton
        on the image sent to the LLM (requires annotate_image=True).
        """
        now = time.monotonic()
        if now - self._last_sent < self.send_interval:
            return None
        self._last_sent = now

        img = frame
        if self.annotate_image and model is not None:
            img = model.draw_skeleton(frame, kp, self.conf_threshold)

        image_b64 = self._encode_frame(img)
        keypoints = kp.to_dict(self.conf_threshold)

        return FrameContext(
            image_b64=image_b64,
            keypoints=keypoints,
            timestamp=time.time(),
            exercise=self.exercise,
        )

    def _encode_frame(self, frame: np.ndarray) -> str:
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ok, buf = cv2.imencode(".jpg", frame, params)
        if not ok:
            raise RuntimeError("Failed to JPEG-encode frame.")
        return base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# End-to-end demo  (wires all three layers together)
# ---------------------------------------------------------------------------

def run_demo(
    exercise: str    = "squat",
    variant: str     = "lightning",
    source:  int     = 0,
    interval: float  = 1.5,
) -> None:
    """
    Captures from webcam, runs MoveNet, and prints the FrameContext that
    would be sent to the LLM on each interval tick.

        python movenet.py --exercise squat --interval 1.5
    """
    model   = MoveNetInference.from_hub(variant)
    engine  = ContextEngine(exercise=exercise, send_interval=interval)

    print(f"Streaming '{exercise}' — press Q to quit.")
    print("On each tick a FrameContext is built and printed (replace with LLM call).\n")

    with VideoStream(source) as vs:
        for frame in vs.frames():
            kp      = model.predict(frame)
            context = engine.build(frame, kp, model=model)

            if context:
                visible = [n for n, v in context.keypoints.items() if v["visible"]]
                print(f"[{context.timestamp:.1f}] {len(visible)} keypoints visible "
                      f"| image {len(context.image_b64) // 1024}KB")
                # Replace the print above with your LLM call:
                # messages = context.to_anthropic_messages()
                # response = anthropic_client.messages.create(
                #     model="claude-opus-4-7",
                #     max_tokens=512,
                #     system="You are a gym coach. Analyse the pose and give form feedback.",
                #     messages=messages,
                # )

            annotated = model.draw_skeleton(frame, kp)
            vs.show(annotated)

            if vs.key_pressed("q"):
                break


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--exercise", default="squat")
    p.add_argument("--variant",  default="lightning", choices=["lightning", "thunder"])
    p.add_argument("--source",   default=0, type=int)
    p.add_argument("--interval", default=1.5, type=float,
                   help="Seconds between LLM context snapshots")
    args = p.parse_args()

    run_demo(args.exercise, args.variant, args.source, args.interval)
