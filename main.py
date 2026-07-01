from __future__ import annotations

from typing import Any

from astrbot.api import logger
from astrbot.api.star import Context, Star, register
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.provider.manager import ProviderManager

try:  # AstrBot <= 4.25 legacy dashboard route stack.
    from astrbot.dashboard.routes.config import ConfigRoute as _LegacyConfigRoute
    from astrbot.dashboard.routes.route import Response as _LegacyResponse
except ModuleNotFoundError:  # AstrBot >= 4.26 moved dashboard APIs to dashboard.api/services.
    _LegacyConfigRoute = None
    _LegacyResponse = None

from . import provider as _provider  # noqa: F401 - importing registers the provider class
from .backup_store import (
    PLUGIN_NAME,
    PROVIDER_TYPE,
    reseed_missing_core_providers,
    sync_backup_from_core_config,
)


PLUGIN_DESC = "为 AstrBot 提供独立的 vLLM Embedding Provider，并将配置镜像备份到 plugin_data。"
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
    "v0.1.1",
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
                "[vLLM Embedding Provider] Restored missing provider(s) from plugin_data backup: %s",
                ", ".join(restored_ids),
            )

        self.context.provider_manager.providers_config = core_config["provider"]
        loaded_ids = await _ensure_vllm_provider_instances(self.context.provider_manager, core_config)
        sync_backup_from_core_config(core_config)

        if loaded_ids:
            logger.info("[vLLM Embedding Provider] Loaded existing vLLM provider instance(s): %s", ", ".join(loaded_ids))
        logger.info("[vLLM Embedding Provider] Initialized. provider_type=%s", PROVIDER_TYPE)

    async def terminate(self) -> None:
        _remove_runtime_patches()
        await _provider.flush_pending_embedding_log_summaries()
        removed_provider = _provider.unregister_vllm_embedding_provider(
            _provider.VLLMEmbeddingProvider.__module__
        )
        if removed_provider:
            logger.info(
                "[vLLM Embedding Provider] Terminated; runtime patches restored and provider registry cleaned."
            )
        else:
            logger.info("[vLLM Embedding Provider] Terminated; runtime patches restored.")


async def _ensure_vllm_provider_instances(
    provider_manager: ProviderManager,
    core_config: AstrBotConfig,
) -> list[str]:
    loaded: list[str] = []
    providers = core_config.get("provider", [])
    if not isinstance(providers, list):
        return loaded

    for provider_config in providers:
        if not isinstance(provider_config, dict):
            continue
        if provider_config.get("type") != PROVIDER_TYPE:
            continue
        provider_id = str(provider_config.get("id") or "").strip()
        if not provider_id:
            continue
        if provider_id in getattr(provider_manager, "inst_map", {}):
            continue
        if not provider_config.get("enable", False):
            continue
        await provider_manager.load_provider(provider_config)
        if provider_id in getattr(provider_manager, "inst_map", {}):
            loaded.append(provider_id)
    return loaded


def _apply_runtime_patches() -> None:
    # AstrBot 4.26+ no longer exposes astrbot.dashboard.routes.*.  Provider
    # schemas are now generated from provider_registry directly, so the old
    # ConfigRoute alias patch is optional.  Keep it only for older AstrBot.
    if _LegacyConfigRoute is not None and not getattr(_LegacyConfigRoute, _CONFIG_ROUTE_PATCH_MARKER, False):
        setattr(
            _LegacyConfigRoute,
            _ORIGINAL_GET_ASTRBOT_CONFIG_ATTR,
            _LegacyConfigRoute._get_astrbot_config,
        )
        setattr(
            _LegacyConfigRoute,
            _ORIGINAL_GET_PROVIDER_TEMPLATE_ATTR,
            _LegacyConfigRoute.get_provider_template,
        )
        setattr(_LegacyConfigRoute, _CONFIG_ROUTE_PATCH_MARKER, True)
        _LegacyConfigRoute._get_astrbot_config = _patched_get_astrbot_config
        _LegacyConfigRoute.get_provider_template = _patched_get_provider_template

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
        ProviderManager.create_provider = _patched_create_provider
        ProviderManager.update_provider = _patched_update_provider
        ProviderManager.delete_provider = _patched_delete_provider


def _restore_original_method(
    owner: type[Any] | None,
    original_attr: str,
    method_name: str,
) -> None:
    if owner is None:
        return
    original_method = getattr(owner, original_attr, None)
    if callable(original_method):
        setattr(owner, method_name, original_method)
    if hasattr(owner, original_attr):
        delattr(owner, original_attr)


def _remove_runtime_patches() -> None:
    _restore_original_method(
        _LegacyConfigRoute,
        _ORIGINAL_GET_ASTRBOT_CONFIG_ATTR,
        "_get_astrbot_config",
    )
    _restore_original_method(
        _LegacyConfigRoute,
        _ORIGINAL_GET_PROVIDER_TEMPLATE_ATTR,
        "get_provider_template",
    )
    if _LegacyConfigRoute is not None and hasattr(_LegacyConfigRoute, _CONFIG_ROUTE_PATCH_MARKER):
        delattr(_LegacyConfigRoute, _CONFIG_ROUTE_PATCH_MARKER)

    _restore_original_method(
        ProviderManager,
        _ORIGINAL_CREATE_PROVIDER_ATTR,
        "create_provider",
    )
    _restore_original_method(
        ProviderManager,
        _ORIGINAL_UPDATE_PROVIDER_ATTR,
        "update_provider",
    )
    _restore_original_method(
        ProviderManager,
        _ORIGINAL_DELETE_PROVIDER_ATTR,
        "delete_provider",
    )
    if hasattr(ProviderManager, _PROVIDER_MANAGER_PATCH_MARKER):
        delattr(ProviderManager, _PROVIDER_MANAGER_PATCH_MARKER)


async def _patched_get_astrbot_config(self: Any) -> dict[str, Any]:
    original_method = getattr(type(self), _ORIGINAL_GET_ASTRBOT_CONFIG_ATTR, None)
    if not callable(original_method):
        raise RuntimeError("Original ConfigRoute._get_astrbot_config is unavailable.")

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


async def _patched_get_provider_template(self: Any) -> dict[str, Any]:
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
    if _LegacyResponse is not None:
        return _LegacyResponse().ok(data=data).__dict__
    return {"status": "ok", "message": None, "data": data}


async def _patched_create_provider(
    self: ProviderManager,
    new_config: dict,
) -> None:
    original_method = getattr(type(self), _ORIGINAL_CREATE_PROVIDER_ATTR, None)
    if not callable(original_method):
        raise RuntimeError("Original ProviderManager.create_provider is unavailable.")

    await original_method(self, new_config)
    sync_backup_from_core_config(self.acm.default_conf)


async def _patched_update_provider(
    self: ProviderManager,
    origin_provider_id: str,
    new_config: dict,
) -> None:
    original_method = getattr(type(self), _ORIGINAL_UPDATE_PROVIDER_ATTR, None)
    if not callable(original_method):
        raise RuntimeError("Original ProviderManager.update_provider is unavailable.")

    await original_method(self, origin_provider_id, new_config)
    sync_backup_from_core_config(self.acm.default_conf)


async def _patched_delete_provider(
    self: ProviderManager,
    provider_id: str | None = None,
    provider_source_id: str | None = None,
) -> None:
    original_method = getattr(type(self), _ORIGINAL_DELETE_PROVIDER_ATTR, None)
    if not callable(original_method):
        raise RuntimeError("Original ProviderManager.delete_provider is unavailable.")

    await original_method(self, provider_id=provider_id, provider_source_id=provider_source_id)
    sync_backup_from_core_config(self.acm.default_conf)

