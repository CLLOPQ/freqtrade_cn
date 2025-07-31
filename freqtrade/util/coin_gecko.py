from pycoingecko import CoinGeckoAPI


class FtCoinGeckoApi(CoinGeckoAPI):
    """
    pycoingecko API的简单封装器，以支持演示API密钥。
    """

    def __init__(self, api_key: str = "", *, is_demo=True, retries=5):
        if api_key and is_demo:
            super().__init__(retries=retries, demo_api_key=api_key)
        else:
            super().__init__(api_key=api_key, retries=retries)