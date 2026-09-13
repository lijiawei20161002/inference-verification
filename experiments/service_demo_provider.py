"""Local integration-test provider, in a process separate from the verifier.

OpenAI-compatible SSE completions with optional exact IDs. This is test support,
not part of the verification service and not required on a production provider.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ivgym.sampling import gumbel_noise, position_seed, gumbel_max_sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--temperature", type=float)
    p.add_argument("--context-window", type=int)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to("cuda")
        .eval()
    )
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def do_POST(self):
            if self.path != "/v1/completions":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", "0"))
            if not 0 < n < 1024 * 1024:
                self.send_error(413)
                return
            body = json.loads(self.rfile.read(n))
            prompt = tok.encode(body["prompt"], add_special_tokens=False)
            effective = (
                prompt[-args.context_window :] if args.context_window else prompt
            )
            x = torch.tensor([effective], device="cuda")
            cache = None
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def emit(value):
                self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
                self.wfile.flush()

            with lock, torch.inference_mode():
                for i in range(body["max_tokens"]):
                    output = model(input_ids=x, past_key_values=cache, use_cache=True)
                    logits = output.logits[0, -1].float().cpu().numpy()
                    noise = gumbel_noise(
                        len(logits),
                        position_seed(
                            body.get("seed", 42), body.get("ivgym_prompt_id", 0), i
                        ),
                    )
                    actual_temperature = (
                        2.0
                        if body.get("model") == "temperature-2"
                        else (args.temperature or body["temperature"])
                    )
                    token = gumbel_max_sample(
                        logits,
                        actual_temperature,
                        noise,
                        body.get("top_k"),
                        body.get("top_p", 1.0),
                    )
                    emit(
                        dict(
                            prompt_token_ids=prompt,
                            choices=[
                                dict(
                                    index=0,
                                    text=tok.decode([token]),
                                    token_ids=[token],
                                    finish_reason=None,
                                )
                            ],
                        )
                    )
                    cache = output.past_key_values
                    x = torch.tensor([[token]], device="cuda")
            emit(dict(choices=[dict(index=0, text="", finish_reason="length")]))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Test provider ready on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
