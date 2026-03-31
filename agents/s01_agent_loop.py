#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess
import json
import urllib.error
import urllib.request

try:
    import readline
    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

from dotenv import load_dotenv

load_dotenv(override=True)

QWEN_API_KEY = (
    os.getenv("QWEN_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
QWEN_BASE_URL = (
    os.getenv("QWEN_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL = os.getenv("MODEL_ID", "qwen3.5-plus")

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- The core pattern: a while loop that calls tools until the model stops --
def qwen_chat(messages: list) -> dict:
    if not QWEN_API_KEY:
        raise RuntimeError("Missing QWEN_API_KEY/DASHSCOPE_API_KEY/OPENAI_API_KEY")
    url = f"{QWEN_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Qwen API HTTP {e.code}: {detail}") from e


def agent_loop(messages: list) -> str:
    while True:
        response = qwen_chat(messages)
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        assistant_message = {"role": "assistant", "content": message.get("content") or ""}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)
        if not tool_calls:
            return message.get("content") or ""
        for tool_call in tool_calls:
            fn = tool_call.get("function", {})
            if fn.get("name") != "bash":
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            command = args.get("command", "")
            print(f"\033[33m$ {command}\033[0m")
            output = run_bash(command)
            print(output[:200])
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": "bash",
                "content": output,
            })


if __name__ == "__main__":
    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        final_text = agent_loop(history)
        if final_text:
            print(final_text)
        print()
