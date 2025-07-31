from optuna.distributions import FloatDistribution


class SKDecimal(FloatDistribution):
    def __init__(
        self,
        low: float,
        high: float,
        *,
        step: float | None = None,
        decimals: int | None = None,
        name=None,
    ):
        """
        固定步长的浮点数分布。
        步长（step）和小数位数（decimals）只能设置一个。
        :param low: 下限：下界
        :param high: 上限：上界
        :param step: 步长：步长大小（例如 0.001）
        :param decimals: 小数位数：要四舍五入的小数位数（例如 3）
        :param name: 名称：分布的名称
        """
        if decimals is not None and step is not None:
            raise ValueError("您只能设置步长（step）或小数位数（decimals）中的一个")
        if decimals is None and step is None:
            raise ValueError("您必须设置步长（step）或小数位数（decimals）中的一个")
        # Convert decimals to step
        self.step = step or (1 / 10**decimals if decimals else 1)
        self.name = name

        super().__init__(
            low=round(low, decimals) if decimals else low,
            high=round(high, decimals) if decimals else high,
            step=self.step,
        )