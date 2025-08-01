# pragma pylint: disable=missing-docstring, too-many-arguments, too-many-ancestors,
# pragma pylint: disable=protected-access, C0103

import datetime
from unittest.mock import MagicMock

import pytest
from requests.exceptions import RequestException

from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from freqtrade.util.coin_gecko import FtCoinGeckoApi
from tests.conftest import log_has, log_has_re


def test_fiat_convert_is_singleton():
    fiat_convert = CryptoToFiatConverter({"a": 22})
    fiat_convert2 = CryptoToFiatConverter({})

    assert fiat_convert is fiat_convert2
    assert id(fiat_convert) == id(fiat_convert2)


def test_fiat_convert_is_supported():
    fiat_convert = CryptoToFiatConverter({})
    assert fiat_convert._is_supported_fiat(fiat="USD") is True
    assert fiat_convert._is_supported_fiat(fiat="usd") is True
    assert fiat_convert._is_supported_fiat(fiat="abc") is False
    assert fiat_convert._is_supported_fiat(fiat="ABC") is False


def test_fiat_convert_find_price(mocker):
    fiat_convert = CryptoToFiatConverter({})

    fiat_convert._coinlistings = {}
    fiat_convert._backoff = 0
    mocker.patch(
        "freqtrade.rpc.fiat_convert.CryptoToFiatConverter._load_cryptomap", return_value=None
    )
    assert fiat_convert.get_price(crypto_symbol="BTC", fiat_symbol="EUR") == 0.0

    with pytest.raises(ValueError, match=r"法币ABC不支持。"):
        fiat_convert._find_price(crypto_symbol="BTC", fiat_symbol="ABC")

    assert fiat_convert.get_price(crypto_symbol="XRP", fiat_symbol="USD") == 0.0

    mocker.patch(
        "freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price", return_value=12345.0
    )
    assert fiat_convert.get_price(crypto_symbol="BTC", fiat_symbol="USD") == 12345.0
    assert fiat_convert.get_price(crypto_symbol="btc", fiat_symbol="usd") == 12345.0

    mocker.patch(
        "freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price", return_value=13000.2
    )
    assert fiat_convert.get_price(crypto_symbol="BTC", fiat_symbol="EUR") == 13000.2


def test_fiat_convert_unsupported_crypto(mocker, caplog):
    mocker.patch("freqtrade.rpc.fiat_convert.CryptoToFiatConverter._coinlistings", return_value=[])
    fiat_convert = CryptoToFiatConverter({})
    assert fiat_convert._find_price(crypto_symbol="CRYPTO_123", fiat_symbol="EUR") == 0.0
    assert log_has("不支持的加密货币符号CRYPTO_123 - 返回0.0", caplog)


def test_fiat_convert_get_price(mocker):
    find_price = mocker.patch(
        "freqtrade.rpc.fiat_convert.CryptoToFiatConverter._find_price", return_value=28000.0
    )

    fiat_convert = CryptoToFiatConverter({})

    with pytest.raises(ValueError, match=r"法币us dollar不支持。"):
        fiat_convert.get_price(crypto_symbol="btc", fiat_symbol="US Dollar")

    # 检查方法返回的值
    pair_len = len(fiat_convert._pair_price)
    assert pair_len == 0
    assert fiat_convert.get_price(crypto_symbol="btc", fiat_symbol="usd") == 28000.0
    assert fiat_convert._pair_price["btc/usd"] == 28000.0
    assert len(fiat_convert._pair_price) == 1
    assert find_price.call_count == 1

    # 验证缓存被使用
    fiat_convert._pair_price["btc/usd"] = 9867.543
    assert fiat_convert.get_price(crypto_symbol="btc", fiat_symbol="usd") == 9867.543
    assert find_price.call_count == 1


def test_fiat_convert_same_currencies():
    fiat_convert = CryptoToFiatConverter({})

    assert fiat_convert.get_price(crypto_symbol="USD", fiat_symbol="USD") == 1.0


def test_fiat_convert_two_FIAT():
    fiat_convert = CryptoToFiatConverter({})

    assert fiat_convert.get_price(crypto_symbol="USD", fiat_symbol="EUR") == 0.0


def test_loadcryptomap():
    fiat_convert = CryptoToFiatConverter({})
    assert len(fiat_convert._coinlistings) == 2

    assert fiat_convert._get_gecko_id("btc") == "bitcoin"


def test_fiat_init_network_exception(mocker):
    # 因为CryptoToFiatConverter是单例，我们重置列表
    listmock = MagicMock(side_effect=RequestException)
    mocker.patch.multiple(
        "freqtrade.rpc.fiat_convert.FtCoinGeckoApi",
        get_coins_list=listmock,
    )
    # with pytest.raises(RequestEsxception):
    fiat_convert = CryptoToFiatConverter({})
    fiat_convert._coinlistings = {}
    fiat_convert._load_cryptomap()

    assert len(fiat_convert._coinlistings) == 0


def test_fiat_convert_without_network():
    # 因为CryptoToFiatConverter是单例，我们重置_coingecko的值

    fiat_convert = CryptoToFiatConverter({})

    cmc_temp = fiat_convert._coingecko
    fiat_convert._coingecko = None

    assert fiat_convert._coingecko is None
    assert fiat_convert._find_price(crypto_symbol="btc", fiat_symbol="usd") == 0.0
    fiat_convert._coingecko = cmc_temp


def test_fiat_too_many_requests_response(mocker, caplog):
    # 因为CryptoToFiatConverter是单例，我们重置列表
    req_exception = "429 Too Many Requests"
    listmock = MagicMock(return_value="{}", side_effect=RequestException(req_exception))
    mocker.patch.multiple(
        "freqtrade.rpc.fiat_convert.FtCoinGeckoApi",
        get_coins_list=listmock,
    )
    # with pytest.raises(RequestEsxception):
    fiat_convert = CryptoToFiatConverter({})
    fiat_convert._coinlistings = {}
    fiat_convert._load_cryptomap()

    assert len(fiat_convert._coinlistings) == 0
    assert fiat_convert._backoff > datetime.datetime.now().timestamp()
    assert log_has(
        "CoinGecko API请求过多，将稍后重试。", caplog
    )


def test_fiat_multiple_coins(caplog):
    fiat_convert = CryptoToFiatConverter({})
    fiat_convert._coinlistings = [
        {"id": "helium", "symbol": "hnt", "name": "Helium"},
        {"id": "hymnode", "symbol": "hnt", "name": "Hymnode"},
        {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
        {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
        {"id": "ethereum-wormhole", "symbol": "eth", "name": "Ethereum Wormhole"},
    ]

    assert fiat_convert._get_gecko_id("btc") == "bitcoin"
    assert fiat_convert._get_gecko_id("hnt") is None
    assert fiat_convert._get_gecko_id("eth") == "ethereum"

    assert log_has("在CoinGecko中发现hnt有多个映射。", caplog)


def test_fiat_invalid_response(mocker, caplog):
    # 因为CryptoToFiatConverter是单例，我们重置列表
    listmock = MagicMock(return_value=None)
    mocker.patch.multiple(
        "freqtrade.rpc.fiat_convert.FtCoinGeckoApi",
        get_coins_list=listmock,
    )
    # with pytest.raises(RequestEsxception):
    fiat_convert = CryptoToFiatConverter({})
    fiat_convert._coinlistings = []
    fiat_convert._load_cryptomap()

    assert len(fiat_convert._coinlistings) == 0
    assert log_has_re(
        "无法加载法币加密货币映射，原因如下：.*", caplog
    )


def test_convert_amount(mocker):
    mocker.patch("freqtrade.rpc.fiat_convert.CryptoToFiatConverter.get_price", return_value=12345.0)

    fiat_convert = CryptoToFiatConverter({})
    result = fiat_convert.convert_amount(crypto_amount=1.23, crypto_symbol="BTC", fiat_symbol="USD")
    assert result == 15184.35

    result = fiat_convert.convert_amount(crypto_amount=1.23, crypto_symbol="BTC", fiat_symbol="BTC")
    assert result == 1.23

    result = fiat_convert.convert_amount(
        crypto_amount="1.23", crypto_symbol="BTC", fiat_symbol="BTC"
    )
    assert result == 1.23


def test_FtCoinGeckoApi():
    ftc = FtCoinGeckoApi()
    assert ftc.extra_params is None
    assert ftc.api_base_url == "https://api.coingecko.com/api/v3/"

    # 默认使用演示模式
    ftc = FtCoinGeckoApi(api_key="123456")
    assert ftc.extra_params is not None
    assert ftc.extra_params["x_cg_demo_api_key"] == "123456"
    assert ftc.api_base_url == "https://api.coingecko.com/api/v3/"

    ftc = FtCoinGeckoApi(api_key="123456", is_demo=False)
    assert ftc.extra_params is not None
    assert ftc.extra_params["x_cg_pro_api_key"] == "123456"
    assert ftc.api_base_url == "https://pro-api.coingecko.com/api/v3/"