import tempfile
import unittest
from pathlib import Path

from support_agent.single_process import single_process_lock


class SingleProcessTests(unittest.TestCase):
    def test_second_run_is_rejected_and_lock_releases_after_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "process.lock"
            with single_process_lock(path):
                with self.assertRaises(RuntimeError):
                    with single_process_lock(path):
                        pass
            with single_process_lock(path):
                self.assertTrue(path.exists())
                self.assertEqual(path.stat().st_size, 1)
