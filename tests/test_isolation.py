import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support_agent.telemetry import Journal
from support_agent.llm import Ollama
from support_agent.settings import runtime_directory
from isolation.guest_firewall import policy


class IsolationTests(unittest.TestCase):
    def test_endpoint_restricted_to_guest_loopback(self):
        with tempfile.TemporaryDirectory() as folder:
            journal = Journal(Path(folder))
            with patch.dict(os.environ, {"SUPPORT_OLLAMA_URL": "http://127.0.0.1:11434"}):
                client = Ollama({}, journal)
                self.assertEqual(client.endpoint, "http://127.0.0.1:11434")
            for value in ("http://example.com:11435", "http://host.docker.internal:11435",
                          "http://192.168.56.1:11435", "https://localhost:11435",
                          "http://localhost:11435/api"):
                with patch.dict(os.environ, {"SUPPORT_OLLAMA_URL": value}):
                    with self.assertRaises(ValueError):
                        Ollama({}, journal)

    def test_runtime_can_be_placed_on_guest_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"SUPPORT_RUNTIME_DIR": folder}):
                self.assertEqual(runtime_directory(Path("/project")), Path(folder).resolve())

    def test_vm_firewall_allows_only_loopback(self):
        rendered = policy()
        self.assertIn("policy drop", rendered)
        self.assertIn('oifname "lo" accept', rendered)
        self.assertNotIn("tcp dport 11435", rendered)
        self.assertNotIn("udp dport 67", rendered)
        self.assertNotIn("0.0.0.0/0", rendered)


if __name__ == "__main__":
    unittest.main()
