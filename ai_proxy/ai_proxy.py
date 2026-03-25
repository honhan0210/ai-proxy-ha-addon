import json
import os
import time
from threading import Lock
from typing import Any

import requests
from fastapi import FastAPI, HTTPException


HA_STATES_URLS = (
    "http://supervisor/core/api/states",
    "http://homeassistant:8123/api/states",
    "http://localhost:8123/api/states",
)
DEFAULT_GEMINI_URL = "http://localhost:8000/v1/chat/completions"
CACHE_TTL_SECONDS = 5
IMPORTANT_SENSOR_KEYWORDS = ("battery", "power", "grid")
IGNORED_ENTITY_PREFIXES = ("cell_voltage", "cell_resistance")
IGNORED_DOMAINS = {"number", "update"}

app = FastAPI(title="AI Proxy")
session = requests.Session()
cache_lock = Lock()
state_cache: dict[str, Any] = {"expires_at": 0.0, "states": []}
options_cache: dict[str, Any] | None = None


def load_addon_options() -> dict[str, Any]:
    global options_cache
    if options_cache is not None:
        return options_cache

    options_path = "/data/options.json"
    if os.path.exists(options_path):
        try:
            with open(options_path, "r", encoding="utf-8") as file:
                options_cache = json.load(file)
                return options_cache
        except (OSError, json.JSONDecodeError):
            pass

    options_cache = {}
    return options_cache


def get_home_assistant_token() -> str:
    options = load_addon_options()
    return (
        os.getenv("HOME_ASSISTANT_TOKEN")
        or options.get("home_assistant_token")
        or os.getenv("HASSIO_TOKEN")
        or os.getenv("SUPERVISOR_TOKEN")
        or ""
    )


def get_gemini_api_url() -> str:
    options = load_addon_options()
    return options.get("gemini_api_url") or os.getenv("GEMINI_API_URL") or DEFAULT_GEMINI_URL


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def state_to_text(state: dict[str, Any]) -> str:
    value = str(state.get("state", "")).strip()
    unit = str(state.get("unit", "")).strip()
    if unit:
        return f"{value} {unit}"
    return value


def simplify_states(states: list[dict[str, Any]]) -> list[dict[str, str]]:
    simplified: list[dict[str, str]] = []

    for item in states:
        entity_id = str(item.get("entity_id", "")).strip()
        raw_state = str(item.get("state", "")).strip()
        if not entity_id or raw_state.lower() in {"unknown", "unavailable"}:
            continue

        domain, _, object_id = entity_id.partition(".")
        domain = domain.lower()
        object_id = object_id.lower()

        if domain in IGNORED_DOMAINS:
            continue
        if object_id.startswith(IGNORED_ENTITY_PREFIXES):
            continue

        attributes = item.get("attributes") or {}
        friendly_name = str(attributes.get("friendly_name") or entity_id)
        search_text = f"{entity_id} {friendly_name}".lower()

        keep = domain in {"light", "switch", "climate"}
        if not keep and domain == "sensor":
            keep = any(keyword in search_text for keyword in IMPORTANT_SENSOR_KEYWORDS)

        if not keep:
            continue

        simplified.append(
            {
                "entity_id": entity_id,
                "name": friendly_name,
                "state": raw_state,
                "unit": str(attributes.get("unit_of_measurement", "")).strip(),
            }
        )

    simplified.sort(key=lambda item: item["name"].lower())
    return simplified


def fetch_home_assistant_states(token: str) -> list[dict[str, Any]]:
    errors: list[str] = []

    for url in HA_STATES_URLS:
        try:
            response = session.get(
                url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue
        except ValueError as exc:
            errors.append(f"{url}: invalid JSON response")
            continue

        if not isinstance(payload, list):
            errors.append(f"{url}: unexpected payload type")
            continue

        return payload

    raise HTTPException(
        status_code=502,
        detail="Failed to fetch Home Assistant states. Tried: " + " | ".join(errors),
    )


def get_states(force_refresh: bool = False) -> list[dict[str, str]]:
    now = time.monotonic()
    with cache_lock:
        if not force_refresh and now < state_cache["expires_at"]:
            return list(state_cache["states"])

    token = get_home_assistant_token()
    if not token:
        raise HTTPException(
            status_code=500,
            detail="Missing Home Assistant bearer token. Set home_assistant_token or HOME_ASSISTANT_TOKEN.",
        )

    payload = fetch_home_assistant_states(token)
    simplified = simplify_states(payload)

    with cache_lock:
        state_cache["states"] = simplified
        state_cache["expires_at"] = time.monotonic() + CACHE_TTL_SECONDS

    return list(simplified)


def format_context(states: list[dict[str, str]]) -> str:
    if not states:
        return "No useful Home Assistant entities available."
    return "\n".join(f"{item['name']}: {state_to_text(item)}" for item in states)


def build_system_prompt(messages: list[dict[str, Any]], context: str) -> str:
    existing_system_parts = [
        message_content_to_text(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    existing_system = "\n\n".join(part for part in existing_system_parts if part).strip()

    context_block = (
        "You are an assistant connected to Home Assistant.\n"
        "Use the Home Assistant context below when it is relevant to the user request.\n"
        "If the requested information is not present in the context, say so clearly.\n\n"
        f"Home Assistant context:\n{context}"
    )

    if existing_system:
        return f"{existing_system}\n\n{context_block}"
    return context_block


def build_forward_payload(body: dict[str, Any], context: str) -> dict[str, Any]:
    input_messages = body.get("messages")
    if not isinstance(input_messages, list) or not input_messages:
        raise HTTPException(status_code=400, detail="Request must include a non-empty messages list.")

    non_system_messages = [
        message for message in input_messages if isinstance(message, dict) and message.get("role") != "system"
    ]

    payload = dict(body)
    payload["messages"] = [
        {
            "role": "system",
            "content": build_system_prompt(input_messages, context),
        },
        *non_system_messages,
    ]
    payload["stream"] = False
    return payload


def extract_assistant_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0] or {}
    message = first_choice.get("message") or {}
    content = message.get("content", "")
    return message_content_to_text(content)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(body: dict[str, Any]) -> dict[str, Any]:
    context = format_context(get_states())
    payload = build_forward_payload(body, context)

    try:
        response = session.post(
            get_gemini_api_url(),
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        response_payload = response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Gemini proxy request failed: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Gemini proxy returned invalid JSON.") from exc

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": extract_assistant_content(response_payload),
                }
            }
        ]
    }
