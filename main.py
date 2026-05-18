from __future__ import annotations

from typing import Any

from astrbot.api import logger
from astrbot.api.star import Context, Star, register
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.provider.manager import ProviderManager
from astrbot.dashboard.routes.config import ConfigRoute
from astrbot.dashboard.routes.route import Response

from . import provider as _provider  # noqa: F401
from .backup_store import (
    PLUGIN_NAME,
    PROVIDER_TYPE,
    reseed_missing_core_providers,
    sync_backup_from_core_config,
)


PLUGIN_DESC = "为 AstrBot 提供独立的 vLLM Embedding 提供商，并将配置镜像备份到 plugin_data。"
TEMPLATE_DISPLAY_NAME = "vLLM Embedding"

_CONFIG_ROUTE_PATCH_MARKER = "__vllm_embedding_provider_template_alias_patch__"
_PROVIDER_MANAGER_PATCH_MARKER = "__vllm_embedding_provider_backup_sync_patch__"

_ORIGINAL_GET_ASTRBOT_CONFIG_ATTR = "__vllm_embedding_provider_original_get_astrbot_config__"
_ORIGINAL_GET_PROVIDER_TEMPLATE_ATTR = "__vllm_embedding_provider_original_get_provider_template__"
_ORIGINAL_CREATE_PROVIDER_ATTR = "__vllm_embedding_provider_original_create_provider__"
_ORIGINAL_UPDATE_PROVIDER_ATTR = "__vllm_embedding_provider_original_update_provider__"
_ORIGINAL_DELETE_PROVIDER_ATTR = "__vllm_embedding_provider_original_delete_provider__"


@register(
    PLUGIN_NAME,
    "Creeper3222",
    PLUGIN_DESC,
    "v0.1.0",
)
class VLLMEmbeddingProviderPlugin(Star):
    def __init__(
        self,
        context: Context,
        config: AstrBotConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(context)
        self.context = context
        self.config = config or {}

    async def initialize(self) -> None:
        _apply_runtime_patches()

        core_config = self.context.provider_manager.acm.default_conf
        restored_ids = reseed_missing_core_providers(core_config)
        if restored_ids:
            core_config.save_config()
            logger.warning(
                "[vLLM Embedding Provider] 已从 plugin_data 补种缺失的 provider: %s",
                ", ".join(restored_ids),
            )

        self.context.provider_manager.providers_config = core_config["provider"]
        sync_backup_from_core_config(core_config)

        logger.info("[vLLM Embedding Provider] 插件初始化完成。")

    async def terminate(self) -> None:
        logger.info(
            "[vLLM Embedding Provider] 插件停止。若需完全卸载运行时模板/备份同步补丁，请重启 AstrBot。"
        )


def _apply_runtime_patches() -> None:
    if not getattr(ConfigRoute, _CONFIG_ROUTE_PATCH_MARKER, False):
        setattr(
            ConfigRoute,
            _ORIGINAL_GET_ASTRBOT_CONFIG_ATTR,
            ConfigRoute._get_astrbot_config,
        )
        setattr(
            ConfigRoute,
            _ORIGINAL_GET_PROVIDER_TEMPLATE_ATTR,
            ConfigRoute.get_provider_template,
        )
        setattr(ConfigRoute, _CONFIG_ROUTE_PATCH_MARKER, True)

    if not getattr(ProviderManager, _PROVIDER_MANAGER_PATCH_MARKER, False):
        setattr(
            ProviderManager,
            _ORIGINAL_CREATE_PROVIDER_ATTR,
            ProviderManager.create_provider,
        )
        setattr(
            ProviderManager,
            _ORIGINAL_UPDATE_PROVIDER_ATTR,
            ProviderManager.update_provider,
        )
        setattr(
            ProviderManager,
            _ORIGINAL_DELETE_PROVIDER_ATTR,
            ProviderManager.delete_provider,
        )
        setattr(ProviderManager, _PROVIDER_MANAGER_PATCH_MARKER, True)

    ConfigRoute._get_astrbot_config = _patched_get_astrbot_config
    ConfigRoute.get_provider_template = _patched_get_provider_template
    ProviderManager.create_provider = _patched_create_provider
    ProviderManager.update_provider = _patched_update_provider
    ProviderManager.delete_provider = _patched_delete_provider


async def _patched_get_astrbot_config(self: ConfigRoute) -> dict[str, Any]:
    original_method = getattr(type(self), _ORIGINAL_GET_ASTRBOT_CONFIG_ATTR, None)
    if not callable(original_method):
        raise RuntimeError("ConfigRoute._get_astrbot_config 原始实现不存在。")

    result = await original_method(self)
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    provider_metadata = (
        metadata.get("provider_group", {})
        .get("metadata", {})
        .get("provider", {})
    )
    config_template = provider_metadata.get("config_template")
    if isinstance(config_template, dict) and PROVIDER_TYPE in config_template:
        config_template[TEMPLATE_DISPLAY_NAME] = config_template.pop(PROVIDER_TYPE)
    return result


async def _patched_get_provider_template(self: ConfigRoute) -> dict[str, Any]:
    config_bundle = await self._get_astrbot_config()
    metadata = config_bundle.get("metadata", {}) if isinstance(config_bundle, dict) else {}
    provider_schema = (
        metadata.get("provider_group", {})
        .get("metadata", {})
        .get("provider", {})
    )

    default_conf = self.core_lifecycle.provider_manager.acm.default_conf
    data = {
        "config_schema": {"provider": provider_schema},
        "providers": default_conf.get("provider", []),
        "provider_sources": default_conf.get("provider_sources", []),
    }
    return Response().ok(data=data).__dict__


async def _patched_create_provider(
    self: ProviderManager,
    new_config: dict,
) -> None:
    original_method = getattr(type(self), _ORIGINAL_CREATE_PROVIDER_ATTR, None)
    if not callable(original_method):
        raise RuntimeError("ProviderManager.create_provider 原始实现不存在。")

    await original_method(self, new_config)
    sync_backup_from_core_config(self.acm.default_conf)


async def _patched_update_provider(
    self: ProviderManager,
    origin_provider_id: str,
    new_config: dict,
) -> None:
    original_method = getattr(type(self), _ORIGINAL_UPDATE_PROVIDER_ATTR, None)
    if not callable(original_method):
        raise RuntimeError("ProviderManager.update_provider 原始实现不存在。")

    await original_method(self, origin_provider_id, new_config)
    sync_backup_from_core_config(self.acm.default_conf)


async def _patched_delete_provider(
    self: ProviderManager,
    provider_id: str | None = None,
    provider_source_id: str | None = None,
) -> None:
    original_method = getattr(type(self), _ORIGINAL_DELETE_PROVIDER_ATTR, None)
    if not callable(original_method):
        raise RuntimeError("ProviderManager.delete_provider 原始实现不存在。")

    await original_method(self, provider_id=provider_id, provider_source_id=provider_source_id)
    sync_backup_from_core_config(self.acm.default_conf)