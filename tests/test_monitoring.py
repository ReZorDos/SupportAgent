import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.eval_suite import MemorySampler
from evaluation.eval_summary import summarize
from observability.monitor import latest_eval_alerts, quality_below_target, sample_once
from observability.setup_monitor import generate_units
from support_agent.policy import evidence_conflict
from support_agent.system_metrics import cpu_percent, memory_snapshot


class MonitoringTests(unittest.TestCase):
    def test_linux_meminfo_and_cpu_math(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meminfo"
            path.write_text("MemTotal: 6291456 kB\nMemAvailable: 1048576 kB\n"
                            "SwapTotal: 2097152 kB\nSwapFree: 1572864 kB\n", encoding="ascii")
            metrics = memory_snapshot(path)
            self.assertEqual(metrics["total_ram_gb"], 6)
            self.assertEqual(metrics["used_ram_gb"], 5)
            self.assertEqual(metrics["available_ram_gb"], 1)
            self.assertEqual(metrics["swap_used_gb"], 0.5)
            self.assertEqual(cpu_percent((100, 20), (200, 40)), 80)

    def test_eval_samples_linux_memory(self):
        fake = {"total_ram_gb": 6, "used_ram_gb": 4.5,
                "available_ram_gb": 1.5, "swap_used_gb": 0.25}
        with patch("eval_suite.memory_snapshot", return_value=fake):
            with MemorySampler() as sampler:
                self.assertTrue(sampler.samples)
            self.assertEqual(sampler.samples[0], 4.5)
            self.assertEqual(sampler.swap_samples[0], 0.25)

    def test_summary_alerts_on_vm_memory_pressure(self):
        report = {"cases": [{"case": {"id": "demo", "expected": "solution"},
                             "status": "solution", "checks": {"mode_matches": True},
                             "wall_seconds": 200, "peak_system_ram_gb": 5.5,
                             "total_system_ram_gb": 6, "min_available_ram_gb": 0.5,
                             "peak_swap_used_gb": 0.4}]}
        summary, alerts = summarize(report)
        self.assertEqual(summary["peak_system_ram_percent"], 91.67)
        self.assertEqual(summary["peak_swap_used_gb"], 0.4)
        self.assertTrue({"slow_case", "high_system_ram", "low_available_ram"}.issubset(
            {row["kind"] for row in alerts}))

    def test_monitor_alerts_only_on_change_and_for_new_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            def bad(_path, _cpu):
                return ({"timestamp": 1, "available_ram_gb": 0.4, "disk_free_gb": 1.2,
                         "cpu_percent": 10}, None)
            def good(_path, _cpu):
                return ({"timestamp": 2, "available_ram_gb": 3, "disk_free_gb": 6,
                         "cpu_percent": 10}, None)
            cpu, active, last = sample_once(runtime, sampler=bad, health=lambda: False)
            self.assertEqual(active, {"ollama_unavailable", "low_available_ram", "low_disk"})
            sample_once(runtime, cpu, active, last, sampler=bad, health=lambda: False)
            cpu, active, last = sample_once(runtime, cpu, active, last, sampler=good, health=lambda: True)
            self.assertFalse(active)
            alerts = [json.loads(line) for path in (runtime / "monitor").glob("alerts-*.jsonl")
                      for line in path.read_text().splitlines()]
            self.assertEqual(len(alerts), 6)
            self.assertEqual([row["state"] for row in alerts].count("resolved"), 3)
            run = runtime / "eval_one"
            run.mkdir()
            (run / "eval_summary.json").write_text("{}", encoding="utf-8")
            (run / "alerts.jsonl").write_text('{"kind":"automatic_check_failed","case":"demo"}\n', encoding="utf-8")
            imported, seen = latest_eval_alerts(runtime, last)
            self.assertEqual(imported[0]["kind"], "eval_automatic_check_failed")
            self.assertEqual(latest_eval_alerts(runtime, seen)[0], [])

    def test_quality_alert_requires_five_completed_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            for number in range(4):
                path = runtime / f"eval_{number}"
                path.mkdir()
                (path / "eval_results.json").write_text(json.dumps({
                    "selected_case": "oversize_pdf", "cases": [
                        {"status": "escalation", "checks": {"mode_matches": False}}]}), encoding="utf-8")
            self.assertIsNone(quality_below_target(runtime))
            path = runtime / "eval_4"
            path.mkdir()
            (path / "eval_results.json").write_text(json.dumps({
                "selected_case": "oversize_pdf", "cases": [
                    {"status": "solution", "checks": {"mode_matches": True}}]}), encoding="utf-8")
            self.assertTrue(quality_below_target(runtime))

    def test_monitor_ignores_partially_written_eval_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            path = runtime / "eval_partial"
            path.mkdir()
            (path / "eval_results.json").write_text("{", encoding="utf-8")
            (path / "eval_summary.json").write_text("{}", encoding="utf-8")
            (path / "alerts.jsonl").write_text('{"kind":', encoding="utf-8")
            self.assertIsNone(quality_below_target(runtime))
            self.assertEqual(latest_eval_alerts(runtime, None), ([], None))

    def test_timer_and_conflicting_evidence(self):
        units = generate_units("/home/test/project", "/home/test/venv/bin/python")
        self.assertIn("--skip-if-busy", units["support-eval.service"])
        self.assertIn("OnUnitActiveSec=6h", units["support-eval.timer"])
        conflict = {"evidence": [
            {"id": "get_upload_rules", "data": {"formats": ["pdf"], "max_size_mb": 20}},
            {"id": "check_file", "data": {"allowed_formats": ["docx"], "max_size_mb": 20,
                                           "errors": ["unsupported_format"]}}]}
        self.assertTrue(evidence_conflict(conflict))


if __name__ == "__main__":
    unittest.main()
