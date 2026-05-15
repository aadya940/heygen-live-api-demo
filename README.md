# Pose Coach

Real-time AI gym coach that watches your form and tells you what to fix — powered by HeyGen and Gemini.

## How it works

- **Pose detection** runs entirely in your browser via TensorFlow.js (MoveNet) — no video ever leaves your device
- **Joint angles** are computed client-side and sent to the backend every 500ms
- **Gemini 2.0 Flash** analyzes your angles against exercise-specific targets and gives honest, short coaching cues
- **HeyGen LiveAvatar** speaks the cues aloud in a corner widget
- **Ask questions** — tap the mic button to ask anything and get a spoken answer from the same coach

## Setup

**Requirements**

```
GEMINI_API_KEY=...
HEYGEN_API_KEY=...
```

**Install & run**

```bash
pip install -r requirements.txt
uvicorn backend:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` and allow camera access.

## Supported exercises

Squat · Deadlift · Push-up · Lunge · Plank · Bicep Curl · Shoulder Press · Row
