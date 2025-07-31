import logging
from copy import deepcopy
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import HTTPException

from freqtrade import __version__
from freqtrade.data.history import get_datahandler
from freqtrade.enums import CandleType, RunMode, State, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.rpc import RPC
from freqtrade.rpc.api_server.api_pairlists import handleExchangePayload
from freqtrade.rpc.api_server.api_schemas import (
    AvailablePairs,
    Balances,
    BlacklistPayload,
    BlacklistResponse,
    Count,
    DailyWeeklyMonthly,
    DeleteLockRequest,
    DeleteTrade,
    Entry,
    ExchangeListResponse,
    Exit,
    ForceEnterPayload,
    ForceEnterResponse,
    ForceExitPayload,
    FreqAIModelListResponse,
    Health,
    HyperoptLossListResponse,
    ListCustomData,
    Locks,
    LocksPayload,
    Logs,
    MarketRequest,
    MarketResponse,
    MixTag,
    OpenTradeSchema,
    PairCandlesRequest,
    PairHistory,
    PerformanceEntry,
    Ping,
    PlotConfig,
    Profit,
    ResultMsg,
    ShowConfig,
    Stats,
    StatusMsg,
    StrategyListResponse,
    StrategyResponse,
    SysInfo,
    Version,
    WhitelistResponse,
)
from freqtrade.rpc.api_server.deps import get_config, get_exchange, get_rpc, get_rpc_optional
from freqtrade.rpc.rpc import RPCException


logger = logging.getLogger(__name__)

# API版本
# 1.1版本之前，未提供版本
# 版本增量应在“小”步骤中进行（如1.1、1.12等），除非发生重大变更。
# 1.11：forcebuy和forcesell支持订单类型
# 1.12：添加黑名单删除端点
# 1.13：forcebuy支持持仓金额
# 版本2.xx -> 期货/做空分支
# 2.14：在交易响应中添加入场/出场订单
# 2.15：添加回测历史端点
# 2.16：额外的每日指标
# 2.17：Forceentry - 杠杆、部分强制出场
# 2.20：添加WebSocket端点
# 2.21：添加new_candle消息类型
# 2.22：在回测中添加FreqAI
# 2.23：允许在Web服务器模式下请求绘图配置
# 2.24：添加cancel_open_order端点
# 2.25：在/status端点添加多个利润值
# 2.26：增加/balance输出
# 2.27：添加/trades/<id>/reload端点
# 2.28：将重载端点切换为Post
# 2.29：添加/exchanges端点
# 2.30：新的/pairlists端点
# 2.31：新的/backtest/history/删除端点
# 2.32：新的/backtest/history/补丁端点
# 2.33：额外的每周/每月指标
# 2.34：新的入场/出场/mix_tags端点
# 2.35：pair_candles和pair_history端点作为Post变体
# 2.40：添加hyperopt-loss端点
# 2.41：添加download-data端点
# 2.42：添加带实时数据的/pair_history端点
API_VERSION = 2.42

# 公开API，无需认证。
router_public = APIRouter()
# 私有API，受认证保护
router = APIRouter()


@router_public.get("/ping", response_model=Ping)
def ping():
    """简单的ping"""
    return {"status": "pong"}


@router.get("/version", response_model=Version, tags=["info"])
def version():
    """机器人版本信息"""
    return {"version": __version__}


@router.get("/balance", response_model=Balances, tags=["info"])
def balance(rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    """账户余额"""
    return rpc._rpc_balance(
        config["stake_currency"],
        config.get("fiat_display_currency", ""),
    )


@router.get("/count", response_model=Count, tags=["info"])
def count(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_count()


@router.get("/entries", response_model=list[Entry], tags=["info"])
def entries(pair: str | None = None, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_enter_tag_performance(pair)


@router.get("/exits", response_model=list[Exit], tags=["info"])
def exits(pair: str | None = None, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_exit_reason_performance(pair)


@router.get("/mix_tags", response_model=list[MixTag], tags=["info"])
def mix_tags(pair: str | None = None, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_mix_tag_performance(pair)


@router.get("/performance", response_model=list[PerformanceEntry], tags=["info"])
def performance(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_performance()


@router.get("/profit", response_model=Profit, tags=["info"])
def profit(rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    return rpc._rpc_trade_statistics(config["stake_currency"], config.get("fiat_display_currency"))


@router.get("/stats", response_model=Stats, tags=["info"])
def stats(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_stats()


@router.get("/daily", response_model=DailyWeeklyMonthly, tags=["info"])
def daily(
    timescale: int = Query(7, ge=1, description="要获取数据的天数"),
    rpc: RPC = Depends(get_rpc),
    config=Depends(get_config),
):
    return rpc._rpc_timeunit_profit(
        timescale, config["stake_currency"], config.get("fiat_display_currency", "")
    )


@router.get("/weekly", response_model=DailyWeeklyMonthly, tags=["info"])
def weekly(
    timescale: int = Query(4, ge=1, description="要获取数据的周数"),
    rpc: RPC = Depends(get_rpc),
    config=Depends(get_config),
):
    return rpc._rpc_timeunit_profit(
        timescale, config["stake_currency"], config.get("fiat_display_currency", ""), "weeks"
    )


@router.get("/monthly", response_model=DailyWeeklyMonthly, tags=["info"])
def monthly(
    timescale: int = Query(3, ge=1, description="要获取数据的月数"),
    rpc: RPC = Depends(get_rpc),
    config=Depends(get_config),
):
    return rpc._rpc_timeunit_profit(
        timescale, config["stake_currency"], config.get("fiat_display_currency", ""), "months"
    )


@router.get("/status", response_model=list[OpenTradeSchema], tags=["info"])
def status(rpc: RPC = Depends(get_rpc)):
    try:
        return rpc._rpc_trade_status()
    except RPCException:
        return []


# 在此处使用响应模型会导致响应时间增加约100%（从1秒增加到2秒）
# 在大型数据库中。正确的响应模型：response_model=TradeResponse，
@router.get("/trades", tags=["info", "trading"])
def trades(
    limit: int = Query(500, ge=1, description="要返回的不同交易的最大数量"),
    offset: int = Query(0, ge=0, description="分页跳过的交易数量"),
    order_by_id: bool = Query(
        True, description="按交易ID排序（默认：True）。如果为False，则按最新时间戳排序"
    ),
    rpc: RPC = Depends(get_rpc),
):
    return rpc._rpc_trade_history(limit, offset=offset, order_by_id=order_by_id)


@router.get("/trade/{tradeid}", response_model=OpenTradeSchema, tags=["info", "trading"])
def trade(tradeid: int = 0, rpc: RPC = Depends(get_rpc)):
    try:
        return rpc._rpc_trade_status([tradeid])[0]
    except (RPCException, KeyError):
        raise HTTPException(status_code=404, detail="未找到交易。")


@router.delete("/trades/{tradeid}", response_model=DeleteTrade, tags=["info", "trading"])
def trades_delete(tradeid: int, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete(tradeid)


@router.delete("/trades/{tradeid}/open-order", response_model=OpenTradeSchema, tags=["trading"])
def trade_cancel_open_order(tradeid: int, rpc: RPC = Depends(get_rpc)):
    rpc._rpc_cancel_open_order(tradeid)
    return rpc._rpc_trade_status([tradeid])[0]


@router.post("/trades/{tradeid}/reload", response_model=OpenTradeSchema, tags=["trading"])
def trade_reload(tradeid: int, rpc: RPC = Depends(get_rpc)):
    rpc._rpc_reload_trade_from_exchange(tradeid)
    return rpc._rpc_trade_status([tradeid])[0]


@router.get("/trades/open/custom-data", response_model=list[ListCustomData], tags=["trading"])
def list_open_trades_custom_data(
    key: str | None = Query(None, description="可选的键，用于相应过滤数据"),
    limit: int = Query(100, ge=1, description="要返回的不同交易的最大数量"),
    offset: int = Query(0, ge=0, description="分页跳过的交易数量"),
    rpc: RPC = Depends(get_rpc),
):
    """
    获取所有未结交易的自定义数据。
    如果提供了键，将用于相应过滤数据。
    分页通过`limit`和`offset`参数实现。
    """
    try:
        return rpc._rpc_list_custom_data(key=key, limit=limit, offset=offset)
    except RPCException as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/trades/{trade_id}/custom-data", response_model=list[ListCustomData], tags=["trading"])
def list_custom_data(trade_id: int, key: str | None = Query(None), rpc: RPC = Depends(get_rpc)):
    """
    获取特定交易的自定义数据。
    如果提供了键，将用于相应过滤数据。
    """
    try:
        return rpc._rpc_list_custom_data(trade_id, key=key)
    except RPCException as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/show_config", response_model=ShowConfig, tags=["info"])
def show_config(rpc: RPC | None = Depends(get_rpc_optional), config=Depends(get_config)):
    state: State | str = ""
    strategy_version = None
    if rpc:
        state = rpc._freqtrade.state
        strategy_version = rpc._freqtrade.strategy.version()
    resp = RPC._rpc_show_config(config, state, strategy_version)
    resp["api_version"] = API_VERSION
    return resp


# /forcebuy已被做空功能弃用。请使用/forceentry
@router.post("/forceenter", response_model=ForceEnterResponse, tags=["trading"])
@router.post("/forcebuy", response_model=ForceEnterResponse, tags=["trading"])
def force_entry(payload: ForceEnterPayload, rpc: RPC = Depends(get_rpc)):
    ordertype = payload.ordertype.value if payload.ordertype else None

    trade = rpc._rpc_force_entry(
        payload.pair,
        payload.price,
        order_side=payload.side,
        order_type=ordertype,
        stake_amount=payload.stakeamount,
        enter_tag=payload.entry_tag or "force_entry",
        leverage=payload.leverage,
    )

    if trade:
        return ForceEnterResponse.model_validate(trade.to_json())
    else:
        return ForceEnterResponse.model_validate(
            {"status": f"Error entering {payload.side} trade for pair {payload.pair}."}
        )


# /forcesell已被做空功能弃用。请使用/forceexit
@router.post("/forceexit", response_model=ResultMsg, tags=["trading"])
@router.post("/forcesell", response_model=ResultMsg, tags=["trading"])
def forceexit(payload: ForceExitPayload, rpc: RPC = Depends(get_rpc)):
    ordertype = payload.ordertype.value if payload.ordertype else None
    return rpc._rpc_force_exit(str(payload.tradeid), ordertype, amount=payload.amount)


@router.get("/blacklist", response_model=BlacklistResponse, tags=["info", "pairlist"])
def blacklist(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_blacklist()


@router.post("/blacklist", response_model=BlacklistResponse, tags=["info", "pairlist"])
def blacklist_post(payload: BlacklistPayload, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_blacklist(payload.blacklist)


@router.delete("/blacklist", response_model=BlacklistResponse, tags=["info", "pairlist"])
def blacklist_delete(pairs_to_delete: list[str] = Query([]), rpc: RPC = Depends(get_rpc)):
    """提供要从黑名单中删除的交易对列表"""

    return rpc._rpc_blacklist_delete(pairs_to_delete)


@router.get("/whitelist", response_model=WhitelistResponse, tags=["info", "pairlist"])
def whitelist(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_whitelist()


@router.get("/locks", response_model=Locks, tags=["info", "locks"])
def locks(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_locks()


@router.delete("/locks/{lockid}", response_model=Locks, tags=["info", "locks"])
def delete_lock(lockid: int, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete_lock(lockid=lockid)


@router.post("/locks/delete", response_model=Locks, tags=["info", "locks"])
def delete_lock_pair(payload: DeleteLockRequest, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete_lock(lockid=payload.lockid, pair=payload.pair)


@router.post("/locks", response_model=Locks, tags=["info", "locks"])
def add_locks(payload: list[LocksPayload], rpc: RPC = Depends(get_rpc)):
    for lock in payload:
        rpc._rpc_add_lock(lock.pair, lock.until, lock.reason, lock.side)
    return rpc._rpc_locks()


@router.get("/logs", response_model=Logs, tags=["info"])
def logs(limit: int | None = None):
    return RPC._rpc_get_logs(limit)


@router.post("/start", response_model=StatusMsg, tags=["botcontrol"])
def start(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_start()


@router.post("/stop", response_model=StatusMsg, tags=["botcontrol"])
def stop(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_stop()


@router.post("/pause", response_model=StatusMsg, tags=["botcontrol"])
@router.post("/stopentry", response_model=StatusMsg, tags=["botcontrol"])
@router.post("/stopbuy", response_model=StatusMsg, tags=["botcontrol"])
def pause(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_pause()


@router.post("/reload_config", response_model=StatusMsg, tags=["botcontrol"])
def reload_config(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_reload_config()


@router.get("/pair_candles", response_model=PairHistory, tags=["candle data"])
def pair_candles(pair: str, timeframe: str, limit: int | None = None, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_analysed_dataframe(pair, timeframe, limit, None)


@router.post("/pair_candles", response_model=PairHistory, tags=["candle data"])
def pair_candles_filtered(payload: PairCandlesRequest, rpc: RPC = Depends(get_rpc)):
    # 带列过滤的高级pair_candles端点
    return rpc._rpc_analysed_dataframe(
        payload.pair, payload.timeframe, payload.limit, payload.columns
    )


@router.get("/plot_config", response_model=PlotConfig, tags=["candle data"])
def plot_config(
    strategy: str | None = None,
    config=Depends(get_config),
    rpc: RPC | None = Depends(get_rpc_optional),
):
    if not strategy:
        if not rpc:
            raise RPCException("在Web服务器模式下，策略是必需的。")
        return PlotConfig.model_validate(rpc._rpc_plot_config())
    else:
        config1 = deepcopy(config)
        config1.update({"strategy": strategy})
    try:
        return PlotConfig.model_validate(RPC._rpc_plot_config_with_strategy(config1))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/strategies", response_model=StrategyListResponse, tags=["strategy"])
def list_strategies(config=Depends(get_config)):
    from freqtrade.resolvers.strategy_resolver import StrategyResolver

    strategies = StrategyResolver.search_all_objects(
        config, False, config.get("recursive_strategy_search", False)
    )
    strategies = sorted(strategies, key=lambda x: x["name"])

    return {"strategies": [x["name"] for x in strategies]}


@router.get("/strategy/{strategy}", response_model=StrategyResponse, tags=["strategy"])
def get_strategy(strategy: str, config=Depends(get_config)):
    if ":" in strategy:
        raise HTTPException(status_code=500, detail="不允许使用base64编码的策略。")

    config_ = deepcopy(config)
    from freqtrade.resolvers.strategy_resolver import StrategyResolver

    try:
        strategy_obj = StrategyResolver._load_strategy(
            strategy, config_, extra_dir=config_.get("strategy_path")
        )
    except OperationalException:
        raise HTTPException(status_code=404, detail="策略未找到")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {
        "strategy": strategy_obj.get_strategy_name(),
        "code": strategy_obj.__source__,
        "timeframe": getattr(strategy_obj, "timeframe", None),
    }


@router.get("/exchanges", response_model=ExchangeListResponse, tags=[])
def list_exchanges(config=Depends(get_config)):
    from freqtrade.exchange import list_available_exchanges

    exchanges = list_available_exchanges(config)
    return {
        "exchanges": exchanges,
    }


@router.get(
    "/hyperoptloss", response_model=HyperoptLossListResponse, tags=["hyperopt", "webserver"]
)
def list_hyperoptloss(
    config=Depends(get_config),
):
    import textwrap

    from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver

    loss_functions = HyperOptLossResolver.search_all_objects(config, False)
    loss_functions = sorted(loss_functions, key=lambda x: x["name"])

    return {
        "loss_functions": [
            {
                "name": x["name"],
                "description": textwrap.dedent((x["class"].__doc__ or "").strip()),
            }
            for x in loss_functions
        ]
    }


@router.get("/freqaimodels", response_model=FreqAIModelListResponse, tags=["freqai"])
def list_freqaimodels(config=Depends(get_config)):
    from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver

    models = FreqaiModelResolver.search_all_objects(config, False)
    models = sorted(models, key=lambda x: x["name"])

    return {"freqaimodels": [x["name"] for x in models]}


@router.get("/available_pairs", response_model=AvailablePairs, tags=["candle data"])
def list_available_pairs(
    timeframe: str | None = None,
    stake_currency: str | None = None,
    candletype: CandleType | None = None,
    config=Depends(get_config),
):
    dh = get_datahandler(config["datadir"], config.get("dataformat_ohlcv"))
    trading_mode: TradingMode = config.get("trading_mode", TradingMode.SPOT)
    pair_interval = dh.ohlcv_get_available_data(config["datadir"], trading_mode)

    if timeframe:
        pair_interval = [pair for pair in pair_interval if pair[1] == timeframe]
    if stake_currency:
        pair_interval = [pair for pair in pair_interval if pair[0].endswith(stake_currency)]
    if candletype:
        pair_interval = [pair for pair in pair_interval if pair[2] == candletype]
    else:
        candle_type = CandleType.get_default(trading_mode)
        pair_interval = [pair for pair in pair_interval if pair[2] == candle_type]

    pair_interval = sorted(pair_interval, key=lambda x: x[0])

    pairs = list({x[0] for x in pair_interval})
    pairs.sort()
    result = {
        "length": len(pairs),
        "pairs": pairs,
        "pair_interval": pair_interval,
    }
    return result


@router.get("/markets", response_model=MarketResponse, tags=["candle data", "webserver"])
def markets(
    query: Annotated[MarketRequest, Query()],
    config=Depends(get_config),
    rpc: RPC | None = Depends(get_rpc_optional),
):
    if not rpc or config["runmode"] == RunMode.WEBSERVER:
        # Web服务器模式
        config_loc = deepcopy(config)
        handleExchangePayload(query, config_loc)
        exchange = get_exchange(config_loc)
    else:
        exchange = rpc._freqtrade.exchange

    return {
        "markets": exchange.get_markets(
            base_currencies=[query.base] if query.base else None,
            quote_currencies=[query.quote] if query.quote else None,
        ),
        "exchange_id": exchange.id,
    }


@router.get("/sysinfo", response_model=SysInfo, tags=["info"])
def sysinfo():
    return RPC._rpc_sysinfo()


@router.get("/health", response_model=Health, tags=["info"])
def health(rpc: RPC = Depends(get_rpc)):
    return rpc.health()