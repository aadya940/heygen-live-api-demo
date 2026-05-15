"""
Pose coaching backend.

Pose detection runs in the browser (MoveNet/WebGL).
This server handles:
  - Receiving joint-angle snapshots from the frontend (every 500ms)
  - Calling Gemini every ~4s for a short coaching cue
  - Streaming the cue back over WebSocket
  - Providing a LiveAvatar session for the avatar widget

WebSocket /stream protocol
--------------------------
Client → Server:
  {"exercise": "squat"}                         # session config / exercise change
  {"type": "keypoints", "text": "...angles..."}  # joint-angle snapshot

Server → Client:
  {"type": "coach", "text": "..."}  # every ~4s
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# ---------------------------------------------------------------------------
# Entrypoint (used by Railway / any host that sets $PORT)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port)

# ---------------------------------------------------------------------------
# Gemini coach
# ---------------------------------------------------------------------------

try:
    from google import genai as genai_lib
    _GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    _GEMINI_OK  = bool(_GEMINI_KEY)
except ImportError:
    genai_lib  = None  # type: ignore[assignment]
    _GEMINI_OK = False

_COACH_SYSTEM = (
    "You are a strict real-time gym coach. "
    "You receive joint angles in degrees for the user's current exercise. "
    "Typical good-form targets: squat knee 70-100°, hip 60-90°, spine lean <30°; "
    "deadlift hip 45-90°, knee 100-140°, spine lean <20°; "
    "pushup elbow 70-100° at bottom; lunge knee 80-100°; "
    "bicep curl elbow 30-60° at top; shoulder press elbow 160-180° at top. "
    "Be HONEST — if angles are off, say exactly what to fix. "
    "When form is reasonable (within ~15° of target), occasionally give warm encouragement: "
    "'Great depth!', 'Looking strong!', 'Nice form!', 'Keep it up!', 'Solid rep!'. "
    "Mix encouragement and corrections naturally — don't always correct, don't always praise. "
    "Reply 3 to 7 words ONLY. "
    "NEVER ask for more data or say you need angles — always give a coaching cue with what you have."
)

_QA_SYSTEM = (
    "You are a knowledgeable gym coach. "
    "Answer the user's question about exercise form, technique, or fitness. "
    "Be specific and helpful. Reply in 1-2 sentences only — short enough to speak aloud."
)

_COACH_INTERVAL = 4.0
_MAX_SNAPSHOTS  = 30

_gemini_client = None


async def _answer_question(question: str, exercise: Optional[str]) -> str:
    text = (
        f"{_QA_SYSTEM}\n\n"
        f"Current exercise: {exercise or 'unknown'}\n\n"
        f"User question: {question}"
    )
    resp = await _gemini_client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=text,
    )
    # Trim to 2 sentences for comfortable avatar speech
    raw = resp.text.replace("\n", " ").strip()
    parts = [s.strip() for s in raw.split(".") if s.strip()]
    return ". ".join(parts[:2]) + "."


async def _call_gemini(exercise: Optional[str], snapshots: list[str]) -> str:
    recent = snapshots[-1]
    text   = (
        f"{_COACH_SYSTEM}\n\n"
        f"Exercise: {exercise or 'unknown'}\n\n"
        f"Current joint angles (degrees):\n{recent}"
    )
    resp = await _gemini_client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=text,
    )
    first_line = next(
        (ln.strip() for ln in resp.text.splitlines() if ln.strip()), ""
    )
    return first_line


# ---------------------------------------------------------------------------
# LiveAvatar (HeyGen) — session management for the frontend avatar widget
# ---------------------------------------------------------------------------

_LA_BASE   = "https://api.liveavatar.com"
_LA_KEY    = os.getenv("HEYGEN_API_KEY")
_LA_AVATAR = os.getenv("LIVEAVATAR_AVATAR_ID", "65f9e3c9-d48b-4118-b73a-4ae2e3cbb8f0")
_LA_OK     = bool(_LA_KEY)


async def _la_create_session() -> dict:
    """Create a LiveAvatar FULL-mode session and return LiveKit credentials."""
    async with httpx.AsyncClient() as c:
        # Step 1: get session token
        r = await c.post(
            f"{_LA_BASE}/v1/sessions/token",
            headers={"X-API-KEY": _LA_KEY, "Content-Type": "application/json"},
            json={
                "mode": "FULL",
                "avatar_id": _LA_AVATAR,
                "avatar_persona": {"language": "en"},
            },
            timeout=15.0,
        )
        r.raise_for_status()
        token_data = r.json()["data"]
        session_id    = token_data["session_id"]
        session_token = token_data["session_token"]

        # Step 2: start session → get LiveKit URL + client token
        r2 = await c.post(
            f"{_LA_BASE}/v1/sessions/start",
            headers={"Authorization": f"Bearer {session_token}"},
            timeout=15.0,
        )
        r2.raise_for_status()
        start_data = r2.json()["data"]

    return {
        "session_id":    session_id,
        "livekit_url":   start_data["livekit_url"],
        "livekit_token": start_data["livekit_client_token"],
    }


# ---------------------------------------------------------------------------
# Per-connection coach session
# ---------------------------------------------------------------------------

class CoachSession:
    def __init__(self, exercise: Optional[str], coach_interval: float = _COACH_INTERVAL):
        self.exercise       = exercise
        self.coach_interval = coach_interval
        self._snapshots: deque[str] = deque(maxlen=_MAX_SNAPSHOTS)
        self._last_coach    = 0.0
        self._coach_task: Optional[asyncio.Task] = None
        self._coach_result: Optional[str]        = None

    def set_exercise(self, exercise: str) -> None:
        self.exercise = exercise
        self._snapshots.clear()

    def add_snapshot(self, text: str) -> None:
        self._snapshots.append(text)

    def start_coach_if_due(self) -> None:
        if not _GEMINI_OK or _gemini_client is None:
            return
        if self._coach_task and not self._coach_task.done():
            return
        now = time.monotonic()
        if now - self._last_coach < self.coach_interval:
            return
        if not self._snapshots:
            return
        self._last_coach = now
        snapshots, exercise = list(self._snapshots), self.exercise

        async def _run() -> None:
            try:
                self._coach_result = await _call_gemini(exercise, snapshots)
            except Exception as exc:
                print(f"Gemini error: {exc}")

        self._coach_task = asyncio.create_task(_run())

    def poll_coach(self) -> Optional[str]:
        if not self._coach_task or not self._coach_task.done():
            return None
        result = self._coach_result
        self._coach_result = None
        self._coach_task   = None
        return result


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="Pose Coach")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def _startup() -> None:
    global _gemini_client
    if _GEMINI_OK:
        _gemini_client = genai_lib.Client(api_key=_GEMINI_KEY)
        print("Gemini coach ready.")
    else:
        print("GEMINI_API_KEY not set — coaching disabled.")
    print(f"LiveAvatar {'ready' if _LA_OK else 'disabled'} — avatar={_LA_AVATAR!r}")


@app.get("/")
async def serve_ui():
    return FileResponse("index.html")


@app.get("/init")
async def get_init():
    return {"skeleton_edges": SKELETON_EDGES}


@app.get("/liveavatar/session")
async def liveavatar_session():
    if not _LA_OK:
        return {"error": "HEYGEN_API_KEY not set"}
    try:
        return await _la_create_session()
    except httpx.HTTPStatusError as e:
        return {"error": f"LiveAvatar {e.response.status_code}", "detail": e.response.text}
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        first = await websocket.receive()
        cfg   = json.loads(first.get("text") or (first.get("bytes") or b"{}").decode())
    except Exception as exc:
        print(f"Config error: {exc}")
        await websocket.close(code=1011)
        return

    session = CoachSession(
        exercise=cfg.get("exercise"),
        coach_interval=float(cfg.get("coach_interval", _COACH_INTERVAL)),
    )
    print(f"Session opened — exercise={session.exercise!r}")

    try:
        while True:
            message = await websocket.receive()
            if "text" not in message:
                continue

            update = json.loads(message["text"])

            if "exercise" in update:
                session.set_exercise(update["exercise"])

            elif update.get("type") == "question" and update.get("text"):
                if _GEMINI_OK and _gemini_client:
                    answer = await _answer_question(update["text"], session.exercise)
                    print(f"Q: {update['text']!r}  →  A: {answer!r}")
                    await websocket.send_text(json.dumps({"type": "answer", "text": answer}))

            elif update.get("type") == "keypoints" and update.get("text"):
                session.add_snapshot(update["text"])
                session.start_coach_if_due()
                coach_text = session.poll_coach()
                if coach_text:
                    print(f"Coach [{session.exercise}]: {coach_text!r}")
                    await websocket.send_text(
                        json.dumps({"type": "coach", "text": coach_text})
                    )

    except (WebSocketDisconnect, RuntimeError):
        print(f"Session closed — exercise={session.exercise!r}")
