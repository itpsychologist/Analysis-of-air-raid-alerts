"""CLI: build AI_CONVERSATION_LOG.md, the log of the AI-assisted work on this project.

Reads the local Claude Code session transcripts of this repository
(~/.claude/projects/<repo path, non-alphanumerics as "-">/*.jsonl) and writes, per session:
the user's messages verbatim, the assistant's visible replies, and one line per tool
call (grouped under "Дії"). Tool output, hidden reasoning, system reminders and
subagent internals are left out. Secrets (GitHub tokens, the API token, e-mail
addresses) are masked; check the result before publishing anyway.

Usage:
    .venv/bin/python scripts/build_conversation_log.py [--out AI_CONVERSATION_LOG.md]
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
# Claude Code names the folder after the path with every non-alphanumeric character as "-"
TRANSCRIPTS = Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(ROOT))
# Titles of the sessions so far; a new session falls back to the title Claude Code gave it
SESSION_TITLES = {
    "76692862-afdf-4560-8d94-2d7e8a24a756": "Постановка задачі, план, етапи 1–2",
    "7bd6dcea-236b-4f40-88fb-671c47f97ee1": "Етапи 3–4: моделі, ціль B, текстові ознаки",
    "54e937ad-8d21-411a-92f2-bca932c3ddb0": "Етап 5: фінальний тест, звіт, GitHub Pages, дашборд",
}
KYIV = ZoneInfo("Europe/Kyiv")

SECRETS = [
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "ghp_***REDACTED***"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "github_pat_***REDACTED***"),
    (re.compile(r"(ALERTS_API_TOKEN\s*=\s*\")(?!your_token_here)[^\"]+(\")"), r"\1***REDACTED***\2"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{16,}"), r"\1***REDACTED***"),
    (re.compile(r"(?!noreply@anthropic\.com)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}\b"), "***@***"),
]


def sessions() -> list[tuple[str, str]]:
    """(session id, title) of every transcript, oldest first."""
    found = []
    for path in TRANSCRIPTS.glob("*.jsonl"):
        first, ai_title = None, None
        for line in path.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            first = first or d.get("timestamp")
            if d.get("type") == "ai-title" and not ai_title:
                ai_title = d.get("aiTitle") or d.get("title")
        if first:
            found.append((first, path.stem, SESSION_TITLES.get(path.stem) or ai_title or path.stem))
    return [(sid, title) for _, sid, title in sorted(found)]


def redact(text: str) -> str:
    for pattern, repl in SECRETS:
        text = pattern.sub(repl, text)
    return text


def ts(value: str | None) -> str:
    if not value:
        return ""
    t = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(KYIV)
    return t.strftime("%d.%m.%Y %H:%M")


def action(item: dict) -> str:
    """One line for a tool call."""
    name, inp = item.get("name", "?"), item.get("input") or {}
    if name == "Bash":
        what = inp.get("description") or inp.get("command", "").splitlines()[0][:100]
        bg = " (у фоні)" if inp.get("run_in_background") else ""
        return f"Bash{bg}: {what}"
    if name in ("Read", "Write", "Edit", "NotebookEdit"):
        path = str(inp.get("file_path", "")).replace(f"{ROOT}/", "")
        path = re.sub(r"/private/tmp/claude-501/[^ ]*/scratchpad/", "scratchpad/", path)
        return f"{name}: `{path}`"
    if name == "Agent":
        return f"Agent ({inp.get('subagent_type', 'general')}): {inp.get('description', '')}"
    if name == "SendMessage":
        return f"SendMessage → субагент: {inp.get('summary', '')}"
    if name == "Artifact":
        return f"Artifact: {inp.get('action', 'publish')}"
    if name in ("Monitor",):
        return f"Monitor: {inp.get('description', '')}"
    if name in ("ScheduleWakeup",):
        return f"ScheduleWakeup: {inp.get('reason', '')}"
    if name == "AskUserQuestion":
        qs = "; ".join(q.get("question", "") for q in inp.get("questions", []))
        return f"AskUserQuestion: {qs}"
    if name == "Skill":
        return f"Skill: {inp.get('skill', '')}"
    if name == "ToolSearch":
        return f"ToolSearch: {inp.get('query', '')}"
    if name == "TaskStop":
        return "TaskStop: зупинка фонового завдання"
    return name


def events(session_id: str):
    """Ordered (kind, timestamp, payload) for one session; streamed message chunks are de-duplicated."""
    seen = set()
    for line in (TRANSCRIPTS / f"{session_id}.jsonl").open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("timestamp")
        if d.get("isSidechain"):
            continue  # subagent internals; their reports reach the main thread as messages
        if d.get("type") == "user" and not d.get("isMeta"):
            c = (d.get("message") or {}).get("content")
            if isinstance(c, str) and "<command-name>" in c:
                # A slash command typed by the user, e.g. /init
                cmd = re.search(r"<command-name>(.*?)</command-name>", c, re.S).group(1).strip()
                args = re.search(r"<command-args>(.*?)</command-args>", c, re.S)
                yield "user", t, f"`{cmd}{(' ' + args.group(1).strip()) if args and args.group(1).strip() else ''}` (команда Claude Code)"
            elif isinstance(c, str) and c.strip() and not c.lstrip().startswith("<"):
                kind = "wakeup" if c.startswith("Fallback:") else "user"
                yield kind, t, c.strip()
        elif d.get("type") == "attachment":
            a = d.get("attachment") or {}
            if a.get("type") == "queued_command" and (a.get("origin") or {}).get("kind") == "human":
                yield "user_mid", t, str(a.get("prompt", "")).strip()
        elif d.get("type") == "assistant":
            m = d.get("message") or {}
            for i, item in enumerate(m.get("content") or []):
                key = (m.get("id"), i, item.get("type"))
                if key in seen:
                    continue
                seen.add(key)
                if item.get("type") == "text" and item.get("text", "").strip():
                    yield "text", t, item["text"].strip()
                elif item.get("type") == "tool_use":
                    yield "tool", t, action(item)


def session_models(session_id: str) -> set[str]:
    out = set()
    for line in (TRANSCRIPTS / f"{session_id}.jsonl").open(encoding="utf-8"):
        if '"assistant"' in line:
            try:
                model = (json.loads(line).get("message") or {}).get("model")
            except json.JSONDecodeError:
                continue
            if model and not model.startswith("<"):
                out.add(model)
    return out


def render() -> str:
    out = []
    totals = {"user": 0, "tool": 0}
    body = []
    toc = []
    all_sessions = sessions()
    span_days: list[str] = []
    models = set()
    for n, (sid, title) in enumerate(all_sessions, 1):
        evs = list(events(sid))
        models |= session_models(sid)
        stamps = [e[1] for e in evs if e[1]]
        span_days.extend(ts(x)[:10] for x in (stamps[0], stamps[-1]) if stamps)
        span = f"{ts(stamps[0])} – {ts(stamps[-1])}" if stamps else ""
        anchor = f"session-{n}"
        toc.append(f"{n}. [{title}](#{anchor}) — {span}")
        body.append(f'\n---\n\n<a id="{anchor}"></a>\n## Сесія {n}. {title}\n\n*{span} (Київ) · `{sid}`*\n')
        segments: list[tuple[str, object]] = []  # ("text", str) | ("tools", list[str]) in time order

        def flush():
            if not segments:
                return
            body.append("#### 🤖 Claude\n")
            for kind_, content in segments:
                if kind_ == "text":
                    body.append(redact(content) + "\n")
                else:
                    body.append(f"<details><summary>Дії ({len(content)})</summary>\n")
                    body.append("\n".join(f"- {redact(t)}" for t in content))
                    body.append("\n</details>\n")
            segments.clear()

        for kind, t, payload in evs:
            if kind in ("user", "user_mid", "wakeup"):
                flush()
                if kind == "wakeup":
                    body.append(f"\n> ⏰ *Автоматична самоперевірка, запланована Claude (не повідомлення користувача), {ts(t)}*\n")
                    continue
                totals["user"] += 1
                label = "👤 Користувач" + (" (надіслано посеред роботи)" if kind == "user_mid" else "")
                quoted = "\n".join("> " + line if line else ">" for line in redact(payload).splitlines())
                body.append(f"\n### {label} · {ts(t)}\n\n{quoted}\n")
            elif kind == "text":
                segments.append(("text", payload))
            elif kind == "tool":
                totals["tool"] += 1
                if segments and segments[-1][0] == "tools":
                    segments[-1][1].append(payload)
                else:
                    segments.append(("tools", [payload]))
        flush()

    generated = datetime.now(timezone.utc).astimezone(KYIV).strftime("%d.%m.%Y %H:%M")
    first_day, last_day = span_days[0], span_days[-1]
    out.append("# AI conversation log: прогнозування повітряних тривог\n")
    out.append(
        "Журнал розмов з AI-асистентом, з яким зроблено весь проєкт "
        "[Analysis-of-air-raid-alerts](https://github.com/itpsychologist/Analysis-of-air-raid-alerts): "
        "від постановки задачі до фінального тесту, звіту і дашборду.\n"
    )
    out.append(
        "| | |\n|---|---|\n"
        f"| Асистент | Claude Code (CLI), модель {', '.join(f'`{m}`' for m in sorted(models))} |\n"
        f"| Період | {first_day} – {last_day}, сесій: {len(all_sessions)} |\n"
        f"| Повідомлень користувача | {totals['user']} |\n"
        f"| Дій асистента (виклики інструментів) | {totals['tool']} |\n"
        f"| Згенеровано | {generated} (Київ), з локальних транскриптів Claude Code |\n"
    )
    out.append(
        "\n**Що в журналі.** Повідомлення користувача дослівно; відповіді асистента, які бачив користувач; "
        "дії асистента — по рядку на виклик інструмента (команда, файл, субагент), згорнуто в «Дії». "
        "**Чого немає:** вмісту виводу команд і файлів, прихованих роздумів моделі, системних нагадувань, "
        "внутрішньої роботи субагентів (їхні висновки асистент переказував у відповідях). "
        "Секрети (токени, email) замасковано як `***REDACTED***`.\n"
    )
    out.append("\n## Зміст\n\n" + "\n".join(toc) + "\n")
    out.extend(body)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "AI_CONVERSATION_LOG.md")
    args = ap.parse_args()
    if not TRANSCRIPTS.exists():
        raise SystemExit(f"no Claude Code transcripts for this repository in {TRANSCRIPTS}")
    text = render()
    args.out.write_text(text, encoding="utf-8")
    print(f"saved {args.out}: {len(text) / 1024:.0f} KB, {text.count(chr(10))} lines")


if __name__ == "__main__":
    main()
