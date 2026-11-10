import json
import os
import subprocess
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

shared = [line for line in Path("/proc/mounts").read_text().splitlines()
          if len(parts := line.split()) >= 3 and parts[2] == "vboxsf"]
if shared:
    raise SystemExit(f"FAIL: host folder remains mounted: {shared}")
print("PASS: no Windows folder is shared")

routes = subprocess.run(["ip", "-o", "route", "show", "default"], capture_output=True,
                        text=True, check=True).stdout.strip()
if routes:
    raise SystemExit(f"FAIL: default network route remains: {routes}")
print("PASS: no default network route")

if os.environ.get("SUPPORT_OLLAMA_URL", "") not in ("", "http://127.0.0.1:11434"):
    raise SystemExit("FAIL: unset SUPPORT_OLLAMA_URL; Ollama must be local to VM")
client = build_opener(ProxyHandler({}))
with client.open("http://127.0.0.1:11434/api/tags", timeout=15) as response:
    tags = json.load(response)
models = {model.get("name") for model in tags.get("models", [])}
if not {"qwen3.5:2b", "gemma3:4b"}.issubset(models):
    raise SystemExit(f"FAIL: required local models unavailable: {models}")
print("PASS: both models are installed inside VM")

try:
    client.open("https://example.org/", timeout=8)
except (URLError, TimeoutError, OSError):
    print("PASS: outbound internet unavailable")
else:
    raise SystemExit("FAIL: outbound internet reachable; disable all VM network adapters")
