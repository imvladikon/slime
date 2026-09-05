#!/usr/bin/env python3
"""Fail closed unless the GLM-5.3-Flash Slime runtime is internally consistent."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


EXPECTED_MEGATRON_COMMIT = "d53ff11c6036349f7f971a46c0c2baf48bee012b"
EXPECTED_SGLANG_COMMIT = "b33136a11b2eb129550f083adcc8d289cd20e677"
EXPECTED_SLIME_REPOSITORY = "https://github.com/imvladikon/slime.git"
EXPECTED_MEGATRON_REPOSITORY = "https://github.com/imvladikon/Megatron-LM.git"
EXPECTED_SGLANG_REPOSITORY = "https://github.com/imvladikon/sglang.git"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slime-root", type=Path, default=Path("/root/slime"))
    parser.add_argument("--megatron-root", type=Path, default=Path("/root/Megatron-LM"))
    parser.add_argument("--sglang-python-root", type=Path, default=Path("/root/sglang-source/python"))
    parser.add_argument("--skip-router-forwarding", action="store_true")
    parser.add_argument("--router-only", action="store_true")
    parser.add_argument("--expected-slime-commit")
    parser.add_argument("--expected-megatron-commit", default=EXPECTED_MEGATRON_COMMIT)
    parser.add_argument("--expected-sglang-commit", default=EXPECTED_SGLANG_COMMIT)
    parser.add_argument("--expected-slime-repository", default=EXPECTED_SLIME_REPOSITORY)
    parser.add_argument("--expected-megatron-repository", default=EXPECTED_MEGATRON_REPOSITORY)
    parser.add_argument("--expected-sglang-repository", default=EXPECTED_SGLANG_REPOSITORY)
    return parser.parse_args()


def assert_import_below(module_name: str, expected_root: Path) -> str:
    module = importlib.import_module(module_name)
    candidates = []
    if getattr(module, "__file__", None):
        candidates.append(Path(module.__file__).resolve())
    candidates.extend(Path(item).resolve() for item in getattr(module, "__path__", ()))
    root = expected_root.resolve()
    if not candidates or not all(candidate.is_relative_to(root) for candidate in candidates):
        raise RuntimeError(f"{module_name} imported from {candidates or ['<unknown>']}, expected it below {root}")
    return str(next(candidate for candidate in candidates if candidate.is_relative_to(root)))


def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def normalize_repository(repository: str) -> str:
    return repository.removesuffix(".git").rstrip("/")


def assert_source_revision(root: Path, expected_repository: str, expected_commit: str) -> str:
    if (root / ".git").exists():
        actual_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        remote = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(["git", "-C", str(root), "diff", "--quiet"], check=True)
    else:
        provenance_path = root / ".source-provenance.json"
        if not provenance_path.is_file():
            raise RuntimeError(f"{root} has neither git metadata nor {provenance_path.name}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        actual_commit = provenance.get("commit")
        remote = provenance.get("repository")
    if actual_commit != expected_commit:
        raise RuntimeError(f"{root} is at {actual_commit}, expected {expected_commit}")
    if normalize_repository(str(remote)) != normalize_repository(expected_repository):
        raise RuntimeError(f"{root} reports repository {remote}, expected {expected_repository}")
    return actual_commit


def wait_for_port(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"router exited before listening: code={process.returncode}\n{stdout}\n{stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"router did not listen on port {port}")


def assert_router_forwards_extension() -> None:
    observed: queue.Queue[tuple[str, dict]] = queue.Queue()

    class WorkerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            response = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            observed.put((self.path, body))
            response = b'{"text":"ok","meta_info":{}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    router_port = unused_port()
    worker = ThreadingHTTPServer(("127.0.0.1", 0), WorkerHandler)
    worker_port = int(worker.server_address[1])
    worker_thread = threading.Thread(target=worker.serve_forever, daemon=True)
    worker_thread.start()
    command = [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--worker-urls",
        f"http://127.0.0.1:{worker_port}",
        "--host",
        "127.0.0.1",
        "--port",
        str(router_port),
        "--policy",
        "round_robin",
        "--disable-health-check",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_port(router_port, process)
        payload = json.dumps(
            {
                "input_ids": [1, 2, 3],
                "return_logprob": True,
                "sampling_params": {"max_new_tokens": 1},
                "return_routed_experts": True,
            }
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{router_port}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"router returned HTTP {response.status}")
            response.read()
        forwarded_path, forwarded = observed.get(timeout=10)
        if forwarded_path != "/generate":
            raise RuntimeError(f"router forwarded an unexpected path: {forwarded_path}")
        if forwarded.get("input_ids") != [1, 2, 3]:
            raise RuntimeError(f"router forwarded an unexpected request: {forwarded}")
        if forwarded.get("return_routed_experts") is not True:
            raise RuntimeError("sglang-router dropped return_routed_experts before forwarding to the worker")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        worker.shutdown()
        worker.server_close()


def main() -> None:
    args = parse_args()
    if args.router_only:
        assert_router_forwards_extension()
        print("GLM53_ROUTER_FORWARDING_PROBE PASS")
        return
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

    origins = {
        "slime": assert_import_below("slime.backends.sglang_utils.arguments", args.slime_root),
        "megatron": assert_import_below("megatron.core", args.megatron_root),
        "sglang": assert_import_below("sglang.srt.server_args", args.sglang_python_root),
    }
    source_commit_path = args.slime_root / ".source-commit"
    image_slime_commit = source_commit_path.read_text(encoding="utf-8").strip() if source_commit_path.is_file() else None
    expected_slime_commit = args.expected_slime_commit or image_slime_commit
    if not expected_slime_commit:
        raise RuntimeError("--expected-slime-commit is required for an extracted source bundle")
    commits = {
        "slime": assert_source_revision(args.slime_root, args.expected_slime_repository, expected_slime_commit),
        "megatron": assert_source_revision(
            args.megatron_root,
            args.expected_megatron_repository,
            args.expected_megatron_commit,
        ),
        "sglang": assert_source_revision(
            args.sglang_python_root.parent,
            args.expected_sglang_repository,
            args.expected_sglang_commit,
        ),
    }
    if image_slime_commit is not None and image_slime_commit != commits["slime"]:
        raise RuntimeError(f"Slime image revision marker is {image_slime_commit}, git HEAD is {commits['slime']}")

    import torch
    import torch_memory_saver  # noqa: F401
    import transformer_engine.pytorch  # noqa: F401
    from transformers import Glm5NextForConditionalGeneration  # noqa: F401

    importlib.import_module("apex")
    importlib.import_module("sglang_router.sglang_router_rs")
    if not torch.version.cuda or torch.version.cuda.split(".")[0] != "13":
        raise RuntimeError(f"expected CUDA 13, got {torch.version.cuda}")
    if not torch.__version__.split("+")[0].startswith("2.13."):
        raise RuntimeError(f"expected PyTorch 2.13, got {torch.__version__}")
    if importlib.metadata.version("sglang-router") != "0.3.2":
        raise RuntimeError(f"expected sglang-router 0.3.2, got {importlib.metadata.version('sglang-router')}")

    from slime.backends.sglang_utils.arguments import add_sglang_arguments

    parser = argparse.ArgumentParser(add_help=False)
    add_sglang_arguments(parser)
    parsed = parser.parse_args(
        [
            "--sglang-nsa-prefill-backend",
            "torch",
            "--sglang-nsa-decode-backend",
            "torch",
        ]
    )
    deprecated_aliases = (
        parsed.sglang_dsa_prefill_backend,
        parsed.sglang_dsa_decode_backend,
    )
    if deprecated_aliases != ("torch", "torch"):
        raise RuntimeError(f"deprecated DSA alias parsing mismatch: {deprecated_aliases}")

    if not args.skip_router_forwarding:
        assert_router_forwards_extension()

    print(
        json.dumps(
            {
                "status": "pass",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "sglang_router": importlib.metadata.version("sglang-router"),
                "commits": commits,
                "origins": origins,
                "router_forwarded_return_routed_experts": not args.skip_router_forwarding,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("GLM53_EXACT_IMAGE_RUNTIME_PROBE PASS")


if __name__ == "__main__":
    main()
