"""
OpenAI-compatible API models and helpers for dskpp.
"""

import uuid
import os
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


AUTH_KEY = os.getenv("AUTH_KEY", "")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "deepseek-chat")

AVAILABLE_MODELS = [
    {
        "id": "deepseek-chat",
        "object": "model",
        "created": 1700000000,
        "owned_by": "deepseek",
        "permission": [],
        "root": "deepseek-chat",
        "parent": None,
    },
    {
        "id": "deepseek-reasoner",
        "object": "model",
        "created": 1700000000,
        "owned_by": "deepseek",
        "permission": [],
        "root": "deepseek-reasoner",
        "parent": None,
    },
]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    messages: List[ChatMessage]
    stream: bool = Field(default=False)
    temperature: Optional[float] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    n: Optional[int] = Field(default=None)
    stop: Optional[Any] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None)
    frequency_penalty: Optional[float] = Field(default=None)
    user: Optional[str] = Field(default=None)


def generate_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert OpenAI messages array to a single prompt string."""
    parts = []
    for msg in messages:
        role = msg.role
        content = msg.content or ""
        if role == "system":
            parts.append(f"[System] {content}")
        elif role == "user":
            parts.append(f"[User] {content}")
        elif role == "assistant":
            parts.append(f"[Assistant] {content}")
        else:
            parts.append(f"[{role}] {content}")
    return "\n\n".join(parts)


def model_to_config(model_name: str) -> Dict[str, Any]:
    """Map OpenAI model name to DeepSeek-specific parameters."""
    model_lower = model_name.lower()

    if model_lower in ("deepseek-reasoner", "deepseek-reasoning"):
        return {"thinking_enabled": True, "search_enabled": False}

    return {"thinking_enabled": False, "search_enabled": False}


def make_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an OpenAI-format error response dict."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }