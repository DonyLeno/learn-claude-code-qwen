#!/usr/bin/env python3
import json
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from types import SimpleNamespace


class QwenClient:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = (
            api_key
            or os.getenv("QWEN_API_KEY")
            or ""
        )
        self.base_url = (
            base_url
            or os.getenv("QWEN_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self.messages = _MessagesAPI(self)


class _MessagesAPI:
    def __init__(self, outer):
        self.outer = outer

    def create(self, model, messages, system=None, tools=None, max_tokens=8000, **kwargs):
        chat_messages = _to_chat_messages(system=system, messages=messages)
        chat_tools = _to_chat_tools(tools or [])
        payload = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
        }
        if chat_tools:
            payload["tools"] = chat_tools
            payload["tool_choice"] = "auto"
        for k in ("temperature", "top_p", "stream"):
            if k in kwargs:
                payload[k] = kwargs[k]
        request_url = f"{self.outer.base_url}/chat/completions"
        request_data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.outer.api_key}",
            "Content-Type": "application/json",
        }
        max_attempts = max(1, int(os.getenv("QWEN_MAX_ATTEMPTS", "4")))
        base_delay = float(os.getenv("QWEN_RETRY_BASE_DELAY", "0.8"))
        retryable_http = {408, 429, 500, 502, 503, 504}
        last_error = None
        for attempt in range(1, max_attempts + 1):
            req = urllib.request.Request(
                request_url,
                data=request_data,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    last_error = None
                    break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="ignore")
                if e.code in retryable_http and attempt < max_attempts:
                    last_error = RuntimeError(f"Qwen API HTTP {e.code}: {detail}")
                    time.sleep(base_delay * (2 ** (attempt - 1)))
                    continue
                raise RuntimeError(f"Qwen API HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, ssl.SSLError, socket.timeout, TimeoutError, ConnectionResetError, OSError) as e:
                last_error = e
                if attempt < max_attempts:
                    time.sleep(base_delay * (2 ** (attempt - 1)))
                    continue
                raise RuntimeError(f"Qwen API network error after {max_attempts} attempts: {e}") from e
        if last_error is not None:
            raise RuntimeError(f"Qwen API network error after {max_attempts} attempts: {last_error}")
        msg = data["choices"][0]["message"]
        content_blocks = []
        if msg.get("content"):
            content_blocks.append(SimpleNamespace(type="text", text=msg["content"]))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args_text = fn.get("arguments", "{}")
            try:
                args = json.loads(args_text) if isinstance(args_text, str) else (args_text or {})
            except json.JSONDecodeError:
                args = {}
            content_blocks.append(
                SimpleNamespace(
                    type="tool_use",
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    input=args,
                )
            )
        stop_reason = "tool_use" if (msg.get("tool_calls") or []) else "end_turn"
        return SimpleNamespace(content=content_blocks, stop_reason=stop_reason, raw=data)


def _to_chat_tools(tools):
    result = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and "function" in t:
            result.append(t)
            continue
        result.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return result


def _tool_result_to_chat_messages(parts):
    out = []
    for p in parts:
        if not isinstance(p, dict):
            if hasattr(p, "type") and getattr(p, "type") == "text":
                out.append({"role": "user", "content": getattr(p, "text", "")})
            continue
        if p.get("type") == "tool_result":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": p.get("tool_use_id", ""),
                    "content": str(p.get("content", "")),
                }
            )
        elif p.get("type") == "text":
            out.append({"role": "user", "content": str(p.get("text", ""))})
    return out


def _assistant_blocks_to_message(content):
    text_parts = []
    tool_calls = []
    for b in content:
        if hasattr(b, "type") and b.type == "text":
            text_parts.append(getattr(b, "text", ""))
        elif hasattr(b, "type") and b.type == "tool_use":
            tool_calls.append(
                {
                    "id": getattr(b, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(b, "name", ""),
                        "arguments": json.dumps(getattr(b, "input", {}), ensure_ascii=False),
                    },
                }
            )
        elif isinstance(b, dict) and b.get("type") == "text":
            text_parts.append(str(b.get("text", "")))
        elif isinstance(b, dict) and b.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": b.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": b.get("name", ""),
                        "arguments": json.dumps(b.get("input", {}), ensure_ascii=False),
                    },
                }
            )
    msg = {"role": "assistant", "content": "\n".join([x for x in text_parts if x])}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _to_chat_messages(system, messages):
    out = []
    if system:
        out.append({"role": "system", "content": str(system)})
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant" and isinstance(content, list):
            out.append(_assistant_blocks_to_message(content))
            continue
        if role == "user" and isinstance(content, list):
            out.extend(_tool_result_to_chat_messages(content))
            continue
        out.append({"role": role, "content": str(content)})
    return out
