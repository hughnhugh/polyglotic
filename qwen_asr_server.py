"""
Qwen3-ASR transcription server - run by qwen_asr_client.py inside its own
venv (venv-qwen-asr/, created by setup.ps1), never imported by the app.

The qwen-asr package pins transformers 4.x and an older huggingface_hub,
which conflict with the main venv's faster-whisper stack, so it lives in a
separate environment and is reached over localhost instead.

    venv-qwen-asr\\Scripts\\python.exe qwen_asr_server.py MODEL_DIR PORT

POST /transcribe  body: little-endian float32 PCM, 16kHz mono
                  -> {"language": "Chinese", "text": "..."}
GET  /health      -> 200 once the model is loaded
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
import transformers
from qwen_asr import Qwen3ASRModel

SAMPLE_RATE = 16000

# Otherwise every segment logs a "Setting pad_token_id..." notice.
transformers.logging.set_verbosity_error()


def main():
    model_dir, port = sys.argv[1], int(sys.argv[2])
    model = Qwen3ASRModel.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=256,  # app segments are capped at a few seconds of speech
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._reply(200 if self.path == "/health" else 404, {"status": "ok"})

        def do_POST(self):
            if self.path != "/transcribe":
                self._reply(404, {"error": "not found"})
                return
            audio = np.frombuffer(self.rfile.read(int(self.headers["Content-Length"])), dtype="<f4")
            try:
                result = model.transcribe(audio=(audio, SAMPLE_RATE), language=None)[0]
                self._reply(200, {"language": result.language or "", "text": result.text or ""})
            except Exception as e:
                self._reply(500, {"error": f"{type(e).__name__}: {e}"})

        def _reply(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # one line per segment would just flood the log

    print(f"Qwen3-ASR ready on port {port}", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
