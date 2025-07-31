from pandas import DataFrame, Series


def get_tick_size_over_time(candles: DataFrame) -> Series:
    """
    计算一段时间内蜡烛图的有效数字位数。
    它使用每个月有效数字位数的月度最大值。
    :param candles: 包含 OHLCV 数据的 DataFrame
    :return: 每个月有效数字位数的平均值的 Series
    """
    # 计算开盘价和收盘价的有效数字位数
    for col in ["open", "high", "low", "close"]:
        candles[f"{col}_count"] = (
            candles[col].round(14).astype(str).str.extract(r"\.(\d*[1-9])")[0].str.len()
        )
    candles["max_count"] = candles[["open_count", "close_count", "high_count", "low_count"]].max(
        axis=1
    )

    candles1 = candles.set_index("date", drop=True)
    # 按月份分组并计算有效数字位数的平均值
    monthly_count_avg1 = candles1["max_count"].resample("MS").max()
    # monthly_open_count_avg
    # 将 monthly_open_count_avg 从 5.0 转换为 0.00001，4.0 转换为 0.0001，依此类推
    monthly_open_count_avg = 1 / 10**monthly_count_avg1

    return monthly_open_count_avg