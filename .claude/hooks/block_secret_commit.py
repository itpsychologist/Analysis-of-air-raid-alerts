#!/usr/bin/env python3
"""PreToolUse hook: block `git commit` if the commit would include the API token.

Blocks when the staged changes (or working-tree changes for `commit -a`) touch
.streamlit/secrets.toml or .env, or contain the token value from secrets.toml.
Exit code 2 blocks the tool call and feeds stderr back to Claude.
"""
import json
import os
import re
import subprocess
import sys

payload = json.load(sys.stdin)
command = payload.get("tool_input", {}).get("command", "")
if not re.search(r"\bgit\b.*\bcommit\b", command):
    sys.exit(0)

root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()


def git(*args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True).stdout


commit_all = re.search(r"\s(-a|--all|-[a-zA-Z]*a[a-zA-Z]*)\b", command)
diff_args = ["HEAD"] if commit_all else ["--cached"]

files = git("diff", *diff_args, "--name-only").split()
blocked_files = [f for f in files if f.endswith(".streamlit/secrets.toml") or os.path.basename(f) == ".env"]
if blocked_files:
    print(f"Blocked: commit includes secret files {blocked_files}. Unstage them first.", file=sys.stderr)
    sys.exit(2)

token = None
try:
    with open(os.path.join(root, ".streamlit", "secrets.toml"), encoding="utf-8") as fh:
        m = re.search(r'ALERTS_API_TOKEN\s*=\s*"([^"]+)"', fh.read())
        token = m.group(1) if m else None
except OSError:
    pass

if token and token in git("diff", *diff_args):
    print("Blocked: staged changes contain the ALERTS_API_TOKEN value. Remove it before committing.", file=sys.stderr)
    sys.exit(2)

sys.exit(0)
