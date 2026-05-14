"""
Test client — simulates the frontend by streaming webcam frames over WebSocket.

Usage:
    python test_client.py --url wss://4020-104-7-12-185.ngrok-free.app/stream
    python test_client.py --url ws://localhost:8000/stream   # local
"""

import argparse
import asyncio
import json
import time

import cv2
import websockets


async def stream(url: str, exercise: str, fps: int) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    frame_interval = 1.0 / fps

    async with websockets.connect(url) as ws:
        # 1. Send config
        await ws.send(json.dumps({
            "exercise":       exercise,
            "send_interval":  2.0,
            "window_seconds": 2.0,
            "sampled_images": 4,
        }))
        print(f"Connected → {url}")
        print(f"Exercise: {exercise} | Streaming at {fps}fps\n")

        async def receive_loop():
            async for message in ws:
                batch = json.loads(message)
                print(
                    f"[batch #{batch['batch_index']}] "
                    f"window={batch['window_seconds']:.1f}s  "
                    f"frames={batch['frame_count']}  "
                    f"images={len(batch['sampled_images'])}  "
                    f"keypoints_seq_len={len(batch['keypoints_sequence'])}"
                )
                # Print all keypoints from the first frame, sorted by confidence
                first = batch["keypoints_sequence"][0] if batch["keypoints_sequence"] else {}
                ranked = sorted(first.items(), key=lambda kv: kv[1]["confidence"], reverse=True)
                for name, v in ranked:
                    marker = "✓" if v["visible"] else "✗"
                    print(f"  {marker} {name:<20} x={v['x']:.3f}  y={v['y']:.3f}  conf={v['confidence']:.2f}")

        # Run receiver concurrently
        recv_task = asyncio.create_task(receive_loop())

        try:
            while True:
                t0 = time.monotonic()

                ok, frame = cap.read()
                if not ok:
                    break

                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                await ws.send(buf.tobytes())

                # Show local preview
                cv2.imshow("test client — press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Pace to target fps
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, frame_interval - elapsed))

        finally:
            recv_task.cancel()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url",      default="ws://localhost:8000/stream")
    p.add_argument("--exercise", default="squat")
    p.add_argument("--fps",      default=30, type=int)
    args = p.parse_args()

    asyncio.run(stream(args.url, args.exercise, args.fps))
