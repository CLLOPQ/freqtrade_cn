from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from fastapi import Depends, HTTPException

from freqtrade.constants import Config
from freqtrade.enums import RunMode
from freqtrade.persistence import Trade
from freqtrade.persistence.models import _request_id_ctx_var
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG
from freqtrade.rpc.rpc import RPC, RPCException

from .webserver import ApiServer


def get_rpc_optional() -> RPC | None:
    if ApiServer._has_rpc:
        return ApiServer._rpc
    return None


async def get_rpc() -> AsyncIterator[RPC] | None:
    _rpc = get_rpc_optional()
    if _rpc:
        request_id = str(uuid4())
        ctx_token = _request_id_ctx_var.set(request_id)
        Trade.rollback()
        try:
            yield _rpc
        finally:
            Trade.session.remove()
            _request_id_ctx_var.reset(ctx_token)

    else:
        raise RPCException("机器人未处于正确状态")


def get_config() -> dict[str, Any]:
    return ApiServer._config


def get_api_config() -> dict[str, Any]:
    return ApiServer._config["api_server"]


def _generate_exchange_key(config: Config) -> str:
    """
    交易所键 - 用于缓存交易所对象。
    """
    return f"{config['exchange']['name']}_{config.get('trading_mode', 'spot')}"


def get_exchange(config=Depends(get_config)):
    exchange_key = _generate_exchange_key(config)
    if not (exchange := ApiBG.exchanges.get(exchange_key)):
        from freqtrade.resolvers import ExchangeResolver

        exchange = ExchangeResolver.load_exchange(config, validate=False, load_leverage_tiers=False)
        ApiBG.exchanges[exchange_key] = exchange
    return exchange


def get_message_stream():
    return ApiServer._message_stream


def is_webserver_mode(config=Depends(get_config)):
    if config["runmode"] != RunMode.WEBSERVER:
        raise HTTPException(status_code=503, detail="机器人未处于正确状态。")
    return None