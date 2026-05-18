from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.star_tools import StarTools

PLUGIN_NAME = "astrbot_plugin_vllm_embedding_provider"
PROVIDER_TYPE = "vllm_embedding"
BACKUP_FILE_NAME = "providers_backup.json"


def get_backup_path() -> Path:
    return StarTools.get_data_dir(PLUGIN_NAME) / BACKUP_FILE_NAME


def load_backup_providers() -> list[dict[str, Any]]:
    backup_path = get_backup_path()
    if not backup_path.exists():
        return []

    try:
        with backup_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning(
            "[vLLM Embedding Provider] 读取备份文件失败，将忽略当前备份: %s",
            exc,
        )
        return []

    providers = payload.get("providers", []) if isinstance(payload, dict) else []
    if not isinstance(providers, list):
        return []

    normalized: list[dict[str, Any]] = []
    for provider in providers:
        if isinstance(provider, dict) and provider.get("type") == PROVIDER_TYPE:
            normalized.append(deepcopy(provider))
    return normalized


def sync_backup_from_core_config(core_config: AstrBotConfig) -> list[dict[str, Any]]:
    providers = extract_vllm_embedding_providers(core_config)
    backup_path = get_backup_path()
    payload = {"providers": providers}
    with backup_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    logger.info(
        "[vLLM Embedding Provider] 已刷新 plugin_data 备份，当前备份 %s 个 provider。",
        len(providers),
    )
    return providers


def reseed_missing_core_providers(core_config: AstrBotConfig) -> list[str]:
    provider_list = ensure_provider_list(core_config)
    existing_ids = {
        str(provider.get("id", "") or "").strip()
        for provider in provider_list
        if isinstance(provider, dict)
    }

    restored_ids: list[str] = []
    for provider in load_backup_providers():
        provider_id = str(provider.get("id", "") or "").strip()
        if not provider_id or provider_id in existing_ids:
            continue
        provider_list.append(deepcopy(provider))
        existing_ids.add(provider_id)
        restored_ids.append(provider_id)

    return restored_ids


def extract_vllm_embedding_providers(core_config: AstrBotConfig) -> list[dict[str, Any]]:
    providers = ensure_provider_list(core_config)
    return [
        deepcopy(provider)
        for provider in providers
        if isinstance(provider, dict) and provider.get("type") == PROVIDER_TYPE
    ]


def ensure_provider_list(core_config: AstrBotConfig) -> list[dict[str, Any]]:
    providers = core_config.get("provider")
    if isinstance(providers, list):
        return providers

    core_config["provider"] = []
    return core_config["provider"]