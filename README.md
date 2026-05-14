# MoveNet Gym Tracking Backend

Real-time pose estimation pipeline that streams keypoints and annotated frames
to an LLM service for exercise coaching.

```
Browser camera ──JPEG──► WebSocket /stream
                               │
                        MoveNetInference        every frame, thread pool
                               │
                          FrameBuffer           2-second sliding window
                               │  every 2s
                          ContextBatch ──JSON──► your LLM service
                                                      │
                                               form feedback JSON
                                                      │
                                               back to frontend
```

---

## Files

| File | Responsibility |
|---|---|
| `movenet.py` | `VideoStream`, `MoveNetInference`, `ContextEngine`, `Keypoints` |
| `backend.py` | `FrameBuffer`, `StreamSession`, `ContextBatch`, FastAPI WebSocket server |

---

## Setup

```bash
pip install fastapi "uvicorn[standard]" opencv-python tensorflow tensorflow-hub numpy
uvicorn backend:app --host 0.0.0.0 --port 8000
```

MoveNet Lightning is downloaded from TF Hub on first startup (~15MB).
Use `MoveNetInference.from_tflite("model.tflite")` for edge/offline deployments.

---

## WebSocket Protocol

**Endpoint:** `ws://localhost:8000/stream`

### 1. Session config (first message, text JSON)

Sent by the client immediately after the connection opens.

```json
{
  "exercise":       "squat",
  "send_interval":  2.0,
  "window_seconds": 2.0,
  "sampled_images": 4,
  "jpeg_quality":   70,
  "conf_threshold": 0.3
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `exercise` | string | `null` | Exercise label passed through to the LLM |
| `send_interval` | float | `2.0` | Seconds between `ContextBatch` flushes |
| `window_seconds` | float | `2.0` | Duration of the sliding frame buffer |
| `sampled_images` | int | `4` | Number of annotated images included per batch |
| `jpeg_quality` | int | `70` | JPEG compression quality (1–100) |
| `conf_threshold` | float | `0.3` | Minimum keypoint confidence to include |

### 2. Frame stream (subsequent messages, binary)

Raw JPEG bytes, one message per animation frame. Send as fast as the camera
produces frames — the server handles throttling internally.

### 3. Exercise change (any time, text JSON)

```json
{ "exercise": "deadlift" }
```

Clears the frame buffer immediately so motion from the previous exercise
does not bleed into the next batch.

---

## ContextBatch — JSON contract

Sent by the server every `send_interval` seconds.

```json
{
  "exercise":           "squat",
  "timestamp":          1715000000.0,
  "window_seconds":     1.97,
  "frame_count":        59,
  "keypoints_sequence": [ ...],
  "sampled_images":     ["<base64 JPEG>", "<base64 JPEG>", "..."],
  "batch_index":        3
}
```

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `exercise` | string \| null | Label from the client |
| `timestamp` | float | Unix time the batch was built |
| `window_seconds` | float | Actual duration of captured frames |
| `frame_count` | int | Number of MoveNet frames in this window |
| `keypoints_sequence` | array | One keypoint dict per frame (see below) |
| `sampled_images` | array of string | Base64 JPEGs with skeleton overlay, evenly sampled |
| `batch_index` | int | Monotonically increasing per connection |

### `keypoints_sequence[i]` — one frame

Each element is a dict keyed by keypoint name (COCO-17).

```json
{
  "nose":           { "x": 0.512, "y": 0.083, "confidence": 0.91, "visible": true },
  "left_shoulder":  { "x": 0.421, "y": 0.231, "confidence": 0.87, "visible": true },
  "right_shoulder": { "x": 0.601, "y": 0.228, "confidence": 0.85, "visible": true },
  "left_elbow":     { "x": 0.389, "y": 0.371, "confidence": 0.78, "visible": true },
  "right_elbow":    { "x": 0.634, "y": 0.368, "confidence": 0.76, "visible": true },
  "left_wrist":     { "x": 0.361, "y": 0.501, "confidence": 0.71, "visible": true },
  "right_wrist":    { "x": 0.659, "y": 0.498, "confidence": 0.69, "visible": true },
  "left_hip":       { "x": 0.438, "y": 0.498, "confidence": 0.92, "visible": true },
  "right_hip":      { "x": 0.572, "y": 0.495, "confidence": 0.91, "visible": true },
  "left_knee":      { "x": 0.431, "y": 0.671, "confidence": 0.88, "visible": true },
  "right_knee":     { "x": 0.578, "y": 0.669, "confidence": 0.86, "visible": true },
  "left_ankle":     { "x": 0.428, "y": 0.841, "confidence": 0.82, "visible": true },
  "right_ankle":    { "x": 0.581, "y": 0.839, "confidence": 0.80, "visible": true },
  "...": "..."
}
```

| Field | Type | Description |
|---|---|---|
| `x` | float [0, 1] | Normalised horizontal position (0 = left edge) |
| `y` | float [0, 1] | Normalised vertical position (0 = top edge) |
| `confidence` | float [0, 1] | MoveNet detection confidence |
| `visible` | bool | `true` if confidence ≥ `conf_threshold` |

All 17 COCO keypoints are always present. Keypoints below `conf_threshold`
have `"visible": false` and their coordinates are unreliable — ignore them.

### COCO-17 keypoint names

```
nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder,
left_elbow, right_elbow,
left_wrist, right_wrist,
left_hip, right_hip,
left_knee, right_knee,
left_ankle, right_ankle
```

---

## Frontend Integration

Minimal browser snippet — no libraries required.

```javascript
const ws = new WebSocket("ws://localhost:8000/stream")

// 1. Send config on open
ws.onopen = () => {
  ws.send(JSON.stringify({
    exercise:       "squat",
    send_interval:  2.0,
    window_seconds: 2.0,
    sampled_images: 4,
  }))
}

// 2. Receive ContextBatch and forward to your LLM service
ws.onmessage = async (e) => {
  const batch = JSON.parse(e.data)
  const response = await fetch("/llm/feedback", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(batch),
  })
  const feedback = await response.json()
  showFeedback(feedback.text)
}

// 3. Stream camera frames
const video  = document.getElementById("video")
const canvas = document.getElementById("canvas")
const ctx    = canvas.getContext("2d")

function sendFrame() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
  canvas.toBlob(
    blob => ws.readyState === WebSocket.OPEN && ws.send(blob),
    "image/jpeg",
    0.75,
  )
  requestAnimationFrame(sendFrame)
}

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream
  video.onloadedmetadata = () => { video.play(); sendFrame() }
})

// 4. Change exercise mid-session
function setExercise(name) {
  ws.send(JSON.stringify({ exercise: name }))
}
```

---

## LLM Service Integration

The `ContextBatch` is designed to plug directly into the Anthropic API.

```python
import anthropic, json

client = anthropic.Anthropic()

def handle_batch(batch_json: str) -> str:
    from backend import ContextBatch
    batch = ContextBatch(**json.loads(batch_json))

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=256,
        system=(
            "You are a gym coach analysing a real-time video of someone exercising. "
            "You receive a short clip as a sequence of images and keypoint motion data. "
            "Identify what the person is doing, evaluate their form, and give one or two "
            "specific, actionable cues. Be brief and encouraging."
        ),
        messages=batch.to_anthropic_messages(),
    )
    return response.content[0].text
```

`to_anthropic_messages()` returns a list with:
- One image block per sampled frame (vision context)
- One text block with the keypoint motion trail and metadata

---

## Latency Budget

| Stage | Typical time |
|---|---|
| Browser JPEG encode | ~5ms |
| WebSocket send | ~1–5ms (LAN) |
| `cv2.imdecode` | ~2ms |
| MoveNet Lightning (CPU) | ~30–50ms |
| MoveNet Lightning (GPU) | ~5–10ms |
| JPEG encode (server) | ~3ms |
| `ContextBatch` JSON serialise | ~1ms |
| **LLM call (Claude)** | **~500–1500ms** |

MoveNet runs in a thread pool — it does not block the WebSocket event loop.
The LLM call dominates; `send_interval: 2.0` gives it comfortable headroom.

---

## Configuration Reference

### `ContextEngine` (movenet.py)

| Param | Default | Description |
|---|---|---|
| `exercise` | `None` | Exercise label attached to every context |
| `send_interval` | `1.0` | Seconds between flushes (overridden by `FrameBuffer` in backend) |
| `jpeg_quality` | `75` | JPEG quality for the image sent to LLM |
| `conf_threshold` | `0.3` | Keypoint visibility threshold |
| `annotate_image` | `True` | Draw skeleton on the image sent to LLM |

### `FrameBuffer` (backend.py)

| Param | Default | Description |
|---|---|---|
| `window_seconds` | `2.0` | Rolling buffer duration |
| `sampled_images` | `4` | Images evenly sampled from the window |
| `jpeg_quality` | `70` | JPEG quality for sampled images |
| `conf_threshold` | `0.3` | Keypoint visibility threshold |
