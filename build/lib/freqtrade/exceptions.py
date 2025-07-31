class FreqtradeException(Exception):
    """
    Freqtrade的基础异常类。在最外层进行处理。
    所有其他异常类型都是此类的子类。
    """


class OperationalException(FreqtradeException):
    """
    需要人工干预并会停止机器人。
    大多数情况下，这是由无效的配置引起的。
    """


class ConfigurationError(OperationalException):
    """
    配置错误。通常由无效的配置引起。
    """


class DependencyException(FreqtradeException):
    """
    指示假设的依赖项未满足。
    这可能发生在账户中当前没有足够的资金时。
    """


class PricingError(DependencyException):
    """
    DependencyException的子类。
    指示无法确定价格。
    隐式涉及买入/卖出操作。
    """


class ExchangeError(DependencyException):
    """
    从交易所抛出的错误。
    有多个错误类型以确定适当的错误。
    """


class InvalidOrderException(ExchangeError):
    """
    当订单无效时返回。例如：
    如果交易所订单的止损被触发，然后尝试取消该订单，应返回此异常。
    """


class RetryableOrderError(InvalidOrderException):
    """
    当订单未找到时返回。
    此错误将以增加的退避时间重复（与DDosError一致）。
    """


class InsufficientFundsError(InvalidOrderException):
    """
    当交易所上没有足够的资金来创建订单时使用此错误。
    """


class TemporaryError(ExchangeError):
    """
    临时的网络或交易所相关错误。
    这可能发生在交易所拥堵、不可用，或者用户有网络问题时。通常会在一段时间后自行解决。
    """


class DDosProtection(TemporaryError):
    """
    由DDoS保护导致的临时错误。
    机器人将等待一秒钟后重试。
    """


class StrategyError(FreqtradeException):
    """
    检测到自定义用户代码错误。
    通常由策略中的错误引起。
    """