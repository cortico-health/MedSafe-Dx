"""
Codex CLI backend: run a single-turn prompt through the locally installed `codex exec`
(ChatGPT plan auth) instead of the OpenRouter API.

Why: OpenRouter bills per token and rejected GPT-6 Astra requests with HTTP 402 when
credits ran low (run-2026-09). The Codex CLI is flat-rate on the ChatGPT plan.

Caveats (record these on any leaderboard row produced this way):
- Codex wraps the prompt in its own agent system prompt and global AGENTS.md; we send our
  system prompt and case as the user message, so the model sees extra scaffolding.
- Temperature is not controllable; reasoning effort is (`model_reasoning_effort`).
- The agent runs sandboxed read-only in an empty directory with an ephemeral session, so
  it cannot browse the repo or persist state, but it could in principle run shell commands.
"""
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional


def call_codex(
    model: str,
    messages: List[Dict[str, str]],
    reasoning_effort: str = "medium",
    timeout: int = 300,
) -> Optional[str]:
    """Return the model's final message text, or None on failure."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    prompt = (
        f"{system}\n\n{user}\n\n"
        "Do not run any commands or read any files. Reply with the JSON only."
    )
    model_slug = model.split("/", 1)[-1]  # accept "openai/gpt-6-astra" or "gpt-6-astra"

    with tempfile.TemporaryDirectory(prefix="medsafe-codex-") as workdir:
        out_path = os.path.join(workdir, "last_message.txt")
        cmd = [
            "codex", "exec",
            "--model", model_slug,
            "-c", f'model_reasoning_effort="{reasoning_effort}"',
            "--sandbox", "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "--cd", workdir,
            "--color", "never",
            "--output-last-message", out_path,
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "NO_COLOR": "1"},
            )
        except subprocess.TimeoutExpired:
            print(f"codex exec timed out after {timeout}s", file=sys.stderr)
            return None
        if proc.returncode != 0:
            print(f"codex exec failed (rc={proc.returncode}): {proc.stderr[-500:]}", file=sys.stderr)
            return None
        try:
            with open(out_path) as f:
                text = f.read().strip()
        except OSError:
            text = ""
        if not text:
            print(f"codex exec returned no final message; stderr: {proc.stderr[-300:]}", file=sys.stderr)
            return None
        return text
