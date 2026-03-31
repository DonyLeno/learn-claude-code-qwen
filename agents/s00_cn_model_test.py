#!/usr/bin/env python3
import json
import os
import sys
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(title: str, value="") -> None:
    if value == "":
        print(f"[{now_str()}] {title}")
    else:
        print(f"[{now_str()}] {title}: {value}")


def mask_key(key: str) -> str:
    k = (key or "").strip()
    if not k:
        return "(empty)"
    if len(k) <= 10:
        return "*" * len(k)
    return f"{k[:6]}...{k[-4:]}"


def pretty_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except Exception:
        return text


def request(method: str, url: str, headers: dict, payload: dict | None, timeout: int = 60) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
            return {
                "ok": True,
                "status": resp.getcode(),
                "elapsed_ms": int((time.time() - t0) * 1000),
                "headers": dict(resp.headers.items()),
                "text": data,
            }
    except urllib.error.HTTPError as e:
        data = e.read().decode("utf-8", errors="ignore")
        return {
            "ok": False,
            "status": e.code,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "headers": dict(e.headers.items()) if e.headers else {},
            "text": data,
        }
    except Exception as e:
        return {
            "ok": False,
            "status": -1,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "headers": {},
            "text": f"{type(e).__name__}: {e}",
        }


def main() -> int:
    load_dotenv(override=True)
    key = (
        os.getenv("QWEN_API_KEY")
        or ""
    )
    key_name = (
        "QWEN_API_KEY" if os.getenv("QWEN_API_KEY")
        else "NONE"
    )
    base_url = (
        os.getenv("QWEN_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    model = os.getenv("MODEL_ID", "qwen3.5-plus")

    log("开始模型连通性测试")
    log("工作目录", os.getcwd())
    log("密钥来源变量", key_name)
    log("密钥掩码", mask_key(key))
    log("模型", model)
    log("Base URL", base_url)

    if not key.strip():
        log("失败", "未检测到可用密钥，请设置 QWEN_API_KEY")
        return 2

    headers = {
        "Authorization": f"Bearer {key.strip()}",
        "Content-Type": "application/json",
    }

    models_url = f"{base_url}/models"
    log("请求 1", f"GET {models_url}")
    r1 = request("GET", models_url, headers, None, timeout=45)
    log("响应 1 状态码", r1["status"])
    log("响应 1 耗时(ms)", r1["elapsed_ms"])
    log("响应 1 头部数量", len(r1["headers"]))
    log("响应 1 内容", pretty_json(r1["text"])[:3000])

    chat_url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "只回复 OK"}],
        "temperature": 0,
    }
    log("请求 2", f"POST {chat_url}")
    log("请求 2 体", json.dumps(payload, ensure_ascii=False))
    r2 = request("POST", chat_url, headers, payload, timeout=90)
    log("响应 2 状态码", r2["status"])
    log("响应 2 耗时(ms)", r2["elapsed_ms"])
    log("响应 2 头部数量", len(r2["headers"]))
    log("响应 2 内容", pretty_json(r2["text"])[:5000])

    if r1["ok"] and r2["ok"]:
        log("结果", "连接成功，模型调用成功")
        return 0
    if r1["status"] == 401 or r2["status"] == 401:
        log("结果", "鉴权失败（401 invalid_api_key），请检查百炼 API Key 是否有效且可用")
        return 3
    log("结果", "连接失败，请根据上方详细日志排查")
    return 1


if __name__ == "__main__":
    sys.exit(main())
