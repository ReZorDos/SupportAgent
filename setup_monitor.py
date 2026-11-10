import sys
from pathlib import Path


def generate_units(project, python):
    project = Path(project).resolve()
    python = Path(python).resolve()
    if " " in str(project) or " " in str(python):
        raise ValueError("Путь проекта и Python не должен содержать пробелов")
    monitor = f"""[Unit]
Description=Local support agent health monitor

[Service]
Type=simple
WorkingDirectory={project}
ExecStart={python} {project / 'monitor.py'} --interval 15
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
"""
    eval_service = f"""[Unit]
Description=Support agent periodic synthetic evaluation

[Service]
Type=oneshot
WorkingDirectory={project}
ExecStart={python} {project / 'eval_suite.py'} --case oversize_pdf --skip-if-busy
"""
    timer = """[Unit]
Description=Schedule synthetic support check while user session is active

[Timer]
OnBootSec=10min
OnUnitActiveSec=6h
Unit=support-eval.service

[Install]
WantedBy=timers.target
"""
    return {"support-monitor.service": monitor,
            "support-eval.service": eval_service,
            "support-eval.timer": timer}


def main():
    root = Path(__file__).resolve().parent
    directory = Path.home() / ".config/systemd/user"
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in generate_units(root, sys.executable).items():
        path = directory / name
        path.write_text(content, encoding="utf-8")
        print(path)
    print("Включение: systemctl --user daemon-reload")
    print("Включение: systemctl --user enable --now support-monitor.service support-eval.timer")


if __name__ == "__main__":
    main()
