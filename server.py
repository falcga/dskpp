import json
import re
import os
import uuid
import time
from urllib.parse import urlparse

from CloudflareBypasser import CloudflareBypasser
from DrissionPage import ChromiumPage, ChromiumOptions
from fastapi import FastAPI, HTTPException, Response, Request, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Optional

import argparse
from pyvirtualdisplay import Display
import uvicorn
import atexit
from asyncio import to_thread

from openai_compat import (
    AUTH_KEY,
    AVAILABLE_MODELS,
    ChatCompletionRequest,
    messages_to_prompt,
    model_to_config,
    generate_id,
    estimate_tokens,
    make_error_response,
)


DOCKER_MODE = os.getenv("DOCKERMODE", "false").lower() == "true"
SERVER_PORT = int(os.getenv("SERVER_PORT", 8021))

browser_path = "/usr/bin/google-chrome"
app = FastAPI(title="dskpp OpenAI Compatible API")


class CookieResponse(BaseModel):
    cookies: Dict[str, str]
    user_agent: str


# Auth

def verify_api_key(authorization: Optional[str], key_param: Optional[str]) -> bool:
    """Verify API key from Authorization header or query param.
    If AUTH_KEY is not set, all requests are accepted."""
    if not AUTH_KEY:
        return True

    if authorization:
        if authorization.startswith("Bearer "):
            token = authorization[7:]
            return token == AUTH_KEY
        return authorization == AUTH_KEY

    if key_param:
        return key_param == AUTH_KEY

    return False


def check_auth(authorization: Optional[str], key_param: Optional[str]):
    """Raise 401 if auth fails."""
    if not verify_api_key(authorization, key_param):
        raise HTTPException(
            status_code=401,
            detail=make_error_response(
                "Invalid API key",
                error_type="invalid_request_error",
                code="invalid_api_key",
            ),
        )


# URL safety check (original)

def is_safe_url(url: str) -> bool:
    parsed_url = urlparse(url)
    ip_pattern = re.compile(
        r"^(127\.0\.0\.1|localhost|0\.0\.0\.0|::1|10\.\d+\.\d+\.\d+|"
        r"172\.1[6-9]\.\d+\.\d+|172\.2[0-9]\.\d+\.\d+|"
        r"172\.3[0-1]\.\d+\.\d+|192\.168\.\d+\.\d+)$"
    )
    hostname = parsed_url.hostname

    if (hostname and ip_pattern.match(hostname)) or parsed_url.scheme == "file":
        return False
    return True


def verify_page_loaded(driver: ChromiumPage) -> bool:
    try:
        body = driver.ele('tag:body', timeout=10)
        return len(body.html) > 100
    except (ValueError, AttributeError):
        return False


def bypass_cloudflare(url: str, retries: int, log: bool, proxy: str = None) -> ChromiumPage:
    max_load_retries = 3

    for load_attempt in range(max_load_retries):

        options = ChromiumOptions().auto_port()

        if DOCKER_MODE:
            options.set_argument("--auto-open-devtools-for-tabs", "true")
            options.set_argument("--remote-debugging-port=9222")
            options.set_argument("--no-sandbox")
            options.set_argument("--disable-gpu")
            options.set_paths(browser_path=browser_path).headless(False)
        else:
            options.set_paths(browser_path=browser_path).headless(False)

        if proxy:
            options.set_proxy(proxy)

        driver = ChromiumPage(addr_or_opts=options)

        try:
            driver.get(url)
            time.sleep(5)

            if not verify_page_loaded(driver):
                driver.quit()
                if load_attempt < max_load_retries - 1:
                    time.sleep(3)
                    continue
                raise Exception("Page load failed")

            cf_bypasser = CloudflareBypasser(driver, retries, log)
            cf_bypasser.bypass()

            return driver

        except Exception as e:
            driver.quit()
            if load_attempt < max_load_retries - 1:
                time.sleep(3)
                continue
            raise e


# Original endpoints

@app.get("/cookies", response_model=CookieResponse)
async def get_cookies(url: str, retries: int = 5, proxy: str = None):

    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        driver = await to_thread(bypass_cloudflare, url, retries, True, proxy)

        cookies = {
            c.get("name", ""): c.get("value", "")
            for c in driver.cookies()
        }

        user_agent = driver.user_agent

        driver.quit()

        return CookieResponse(cookies=cookies, user_agent=user_agent)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/html")
async def get_html(url: str, retries: int = 5, proxy: str = None):

    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        driver = await to_thread(bypass_cloudflare, url, retries, True, proxy)

        html = driver.html

        cookies_json = {
            c.get("name", ""): c.get("value", "")
            for c in driver.cookies()
        }

        response = Response(content=html, media_type="text/html")
        response.headers["cookies"] = json.dumps(cookies_json)
        response.headers["user_agent"] = driver.user_agent

        driver.quit()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# OpenAI-compatible endpoints

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(
    authorization: Optional[str] = Header(None),
    key: Optional[str] = None,
):
    check_auth(authorization, key)
    return {
        "object": "list",
        "data": AVAILABLE_MODELS,
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None),
    key: Optional[str] = None,
):
    check_auth(authorization, key)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=make_error_response("Invalid JSON body"),
        )

    try:
        req = ChatCompletionRequest(**body)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=make_error_response(f"Invalid request: {str(e)}"),
        )

    if not req.messages:
        raise HTTPException(
            status_code=400,
            detail=make_error_response("messages is required"),
        )

    prompt = messages_to_prompt(req.messages)
    model_config = model_to_config(req.model)
    completion_id = generate_id()
    created = int(time.time())
    model_name = req.model

    if req.stream:
        return StreamingResponse(
            stream_chat_completion(
                prompt=prompt,
                completion_id=completion_id,
                created=created,
                model_name=model_name,
                model_config=model_config,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        full_content = ""
        async for chunk in _proxy_chat(prompt, model_config):
            if chunk.get("type") == "text":
                full_content += chunk.get("content", "")
            elif chunk.get("finish_reason") == "stop":
                break

        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(full_content)

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


# Streaming internals

def _load_cookies() -> Dict[str, str]:
    """Load DeepSeek cookies from disk."""
    cookies_path = os.path.join(os.path.dirname(__file__), "dsk", "dsk", "cookies.json")
    try:
        with open(cookies_path, "r") as f:
            cookie_data = json.load(f)
            return cookie_data.get("cookies", {})
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


async def _proxy_chat(prompt: str, model_config: dict):
    """Proxy chat to upstream DeepSeek WebSocket API."""
    import websockets

    uid = str(uuid.uuid4())
    ws_url = f"wss://chat.deepseek.com/api/v0/chat/ws?uid={uid}"

    cookies = _load_cookies()
    cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())

    try:
        async with websockets.connect(
            ws_url,
            additional_headers={
                "Cookie": cookie_str,
                "Origin": "https://chat.deepseek.com",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
        ) as ws:
            chat_request = {
                "type": "chat",
                "payload": {
                    "prompt": prompt,
                    "thinking_enabled": model_config.get("thinking_enabled", False),
                    "search_enabled": model_config.get("search_enabled", False),
                },
            }
            await ws.send(json.dumps(chat_request))

            async for message in ws:
                try:
                    raw = message if isinstance(message, str) else message.decode("utf-8")
                    data = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                msg_type = data.get("type", "")

                if msg_type == "chat_response":
                    payload = data.get("payload") or {}
                    content = payload.get("content") or ""
                    is_end = payload.get("is_end", False)

                    if content:
                        yield {"type": "text", "content": content, "finish_reason": None}

                    if is_end:
                        yield {"type": "text", "content": "", "finish_reason": "stop"}
                        return

                elif msg_type == "error":
                    error_msg = (data.get("payload") or {}).get("message", "Upstream error")
                    yield {"type": "error", "content": error_msg, "finish_reason": "stop"}
                    return

            yield {"type": "text", "content": "", "finish_reason": "stop"}

    except Exception as e:
        yield {"type": "error", "content": f"Connection error: {str(e)}", "finish_reason": "stop"}


async def stream_chat_completion(
    prompt: str,
    completion_id: str,
    created: int,
    model_name: str,
    model_config: dict,
):
    """Async generator yielding SSE-formatted OpenAI streaming chunks."""
    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    async for chunk in _proxy_chat(prompt, model_config):
        content = chunk.get("content", "")
        finish_reason = chunk.get("finish_reason")

        if content:
            data_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data_chunk)}\n\n"

        if finish_reason == "stop":
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="dskpp - OpenAI compatible API server")
    parser.add_argument("--nolog", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    display = None

    if args.headless or DOCKER_MODE:
        display = Display(visible=0, size=(1920, 1080))
        display.start()

        def cleanup():
            if display:
                display.stop()

        atexit.register(cleanup)

    print(f"\033[92m[dskpp] Starting OpenAI-compatible API server on port {SERVER_PORT}\033[0m")
    print("\033[92m[dskpp] Endpoints:\033[0m")
    print("\033[92m  GET  /v1/models\033[0m")
    print("\033[92m  POST /v1/chat/completions\033[0m")
    print("\033[92m  GET  /health\033[0m")
    print("\033[92m  GET  /cookies\033[0m")
    print("\033[92m  GET  /html\033[0m")
    if AUTH_KEY:
        print("\033[92m  Auth: enabled (AUTH_KEY set)\033[0m")
    else:
        print("\033[93m  Auth: disabled (AUTH_KEY not set)\033[0m")

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)