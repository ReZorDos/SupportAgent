# Учебный симулятор технической поддержки

Локальная мультиагентная система на Python, LangGraph и Ollama. Диспетчер выбирает тему, диагност использует учебные инструменты, специалист готовит ответ, независимый валидатор проверяет его. Для основных ролей применяется `qwen3.5:2b`, для проверки  `gemma3:4b`. Модели работают последовательно. Реальная учебная платформа не подключена: статусы курсов, профилей, материалов и оценок синтетические.

## Запуск в изолированной VM

VM уже установлена и проверена. Один перенос нового пакета описан в [ISOLATION.md](ISOLATION.md). В VM:

```bash
cd "$HOME/SupportAgentLinux"
"$HOME/support-venv/bin/python" -m unittest discover -s tests -q
"$HOME/support-venv/bin/python" isolation/check_boundary.py
"$HOME/support-venv/bin/python" app.py
```

В диалоге вводите известные данные, остальные поля оставляйте пустыми. `/exit` завершает работу, `/resolved` после одобренного решения сохраняет подтверждённый обезличенный случай в локальную память. Используйте только вымышленные данные.

## Структура

- `app.py` — точка запуска; `support_agent/cli.py` — диалог с пользователем.
- `support_agent/workflow.py` — LangGraph; `models.py` — схемы данных.
- `support_agent/policy.py` — проверяемые ограничения и резервные ответы.
- `support_agent/tools.py` — учебные инструменты; `llm.py` — локальная Ollama.
- `support_agent/memory.py` — SQLite; `telemetry.py` — события и сигналы; `settings.py` — пути и ресурсы.
- `prompts/`, `skills/` и файлы `*_POLICY.md` — инструкции для ролей.
- `eval_suite.py`, `eval_summary.py`, `tests/` — оценка и автоматические тесты.
- `trace_view.py`, `retention.py`, `isolation/` — просмотр траекторий, срок хранения журналов, проверки границ VM.
- `monitor.py`, `setup_monitor.py` — локальный монитор и расписание контрольных прогонов.
- `compare_models.py` — одинаковые задания для сравнения трёх моделей.

`runtime/` создаётся при запуске внутри VM. Там находятся файлы диалогов и оценок, `events.jsonl`, `alerts.jsonl`, `monitor/` и SQLite. Папка не входит в архив. `python retention.py` показывает старые журналы; удаление запускается только с `--apply`.

## Проверка качества

Один пример с реальными моделями:

```bash
"$HOME/support-venv/bin/python" eval_suite.py --case oversize_pdf
```
