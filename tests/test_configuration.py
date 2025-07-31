# pragma pylint: disable=missing-docstring, protected-access, invalid-name
import json
import warnings
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from jsonschema import ValidationError

from freqtrade.commands import Arguments
from freqtrade.configuration import (
    Configuration,
    remove_exchange_credentials,
    sanitize_config,
    validate_config_consistency,
)
from freqtrade.configuration.config_validation import validate_config_schema
from freqtrade.configuration.deprecated_settings import (
    check_conflicting_settings,
    process_deprecated_setting,
    process_removed_setting,
    process_temporary_deprecated_settings,
)
from freqtrade.configuration.environment_vars import _flat_vars_to_nested_dict
from freqtrade.configuration.load_config import (
    load_config_file,
    load_file,
    load_from_files,
    log_config_error_range,
)
from freqtrade.constants import DEFAULT_DB_DRYRUN_URL, DEFAULT_DB_PROD_URL, ENV_VAR_PREFIX
from freqtrade.enums import RunMode
from freqtrade.exceptions import ConfigurationError, OperationalException
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    log_has,
    log_has_re,
    patched_configuration_load_config_file,
)


@pytest.fixture(scope="function")
def all_conf():
    config_file = Path(__file__).parents[1] / "config_examples/config_full.example.json"
    conf = load_config_file(str(config_file))
    return conf


def test_load_config_missing_attributes(default_conf) -> None:
    """测试加载配置时缺少必要属性的情况"""
    conf = deepcopy(default_conf)
    conf.pop("exchange")

    with pytest.raises(ValidationError, match=r".*'exchange' 是必需的属性.*"):
        validate_config_schema(conf)

    conf = deepcopy(default_conf)
    conf.pop("stake_currency")
    conf["runmode"] = RunMode.DRY_RUN
    with pytest.raises(ValidationError, match=r".*'stake_currency' 是必需的属性.*"):
        validate_config_schema(conf)


def test_load_config_incorrect_stake_amount(default_conf) -> None:
    """测试错误的stake_amount配置"""
    default_conf["stake_amount"] = "fake"

    with pytest.raises(ValidationError, match=r".*'fake' 与 'unlimited' 不匹配.*"):
        validate_config_schema(default_conf)


def test_load_config_file(default_conf, mocker, caplog) -> None:
    """测试加载配置文件功能"""
    del default_conf["user_data_dir"]
    default_conf["datadir"] = str(default_conf["datadir"])
    file_mock = mocker.patch(
        "freqtrade.configuration.load_config.Path.open",
        mocker.mock_open(read_data=json.dumps(default_conf)),
    )

    validated_conf = load_config_file("somefile")
    assert file_mock.call_count == 1
    assert validated_conf.items() >= default_conf.items()


def test_load_config_file_error(default_conf, mocker, caplog) -> None:
    """测试加载配置文件时的错误处理"""
    del default_conf["user_data_dir"]
    default_conf["datadir"] = str(default_conf["datadir"])
    filedata = json.dumps(default_conf).replace('"stake_amount": 0.001,', '"stake_amount": .001,')
    mocker.patch(
        "freqtrade.configuration.load_config.Path.open", mocker.mock_open(read_data=filedata)
    )
    mocker.patch.object(Path, "read_text", MagicMock(return_value=filedata))

    with pytest.raises(OperationalException, match=r".*请验证以下部分.*"):
        load_config_file("somefile")


def test_load_config_file_error_range(default_conf, mocker, caplog) -> None:
    """测试配置文件错误范围的日志记录"""
    del default_conf["user_data_dir"]
    default_conf["datadir"] = str(default_conf["datadir"])
    filedata = json.dumps(default_conf).replace('"stake_amount": 0.001,', '"stake_amount": .001,')
    mocker.patch.object(Path, "read_text", MagicMock(return_value=filedata))

    x = log_config_error_range("somefile", "Parse error at offset 64: Invalid value.")
    assert isinstance(x, str)
    assert (
        x == '{"max_open_trades": 1, "stake_currency": "BTC", '
        '"stake_amount": .001, "fiat_display_currency": "USD", '
        '"timeframe": "5m", "dry_run": true, "cance'
    )

    filedata = json.dumps(default_conf, indent=2).replace(
        '"stake_amount": 0.001,', '"stake_amount": .001,'
    )
    mocker.patch.object(Path, "read_text", MagicMock(return_value=filedata))

    x = log_config_error_range("somefile", "Parse error at offset 4: Invalid value.")
    assert isinstance(x, str)
    assert x == '  "max_open_trades": 1,\n  "stake_currency": "BTC",\n  "stake_amount": .001,'

    x = log_config_error_range("-", "")
    assert x == ""


def test_load_file_error(tmp_path):
    """测试加载不存在的文件时的错误"""
    testpath = tmp_path / "config.json"
    with pytest.raises(OperationalException, match=r"文件 .* 未找到!"):
        load_file(testpath)


def test__args_to_config(caplog):
    """测试将命令行参数转换为配置"""
    arg_list = ["trade", "--strategy-path", "TestTest"]
    args = Arguments(arg_list).get_parsed_arg()
    configuration = Configuration(args)
    config = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # 无警告情况
        configuration._args_to_config(config, argname="strategy_path", logstring="DeadBeef")
        assert len(w) == 0
        assert log_has("DeadBeef", caplog)
        assert config["strategy_path"] == "TestTest"

    configuration = Configuration(args)
    config = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # 有弃用警告情况
        configuration._args_to_config(
            config, argname="strategy_path", logstring="DeadBeef", deprecated_msg="即将移除!"
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "已弃用: 即将移除!" in str(w[-1].message)
        assert log_has("DeadBeef", caplog)
        assert config["strategy_path"] == "TestTest"


def test_load_config_max_open_trades_zero(default_conf, mocker, caplog) -> None:
    """测试max_open_trades为0的配置加载"""
    default_conf["max_open_trades"] = 0
    patched_configuration_load_config_file(mocker, default_conf)

    args = Arguments(["trade"]).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf["max_open_trades"] == 0
    assert "internals" in validated_conf


def test_load_config_combine_dicts(default_conf, mocker, caplog) -> None:
    """测试合并多个配置字典"""
    conf1 = deepcopy(default_conf)
    conf2 = deepcopy(default_conf)
    del conf1["exchange"]["key"]
    del conf1["exchange"]["secret"]
    del conf2["exchange"]["name"]
    conf2["exchange"]["pair_whitelist"] += ["NANO/BTC"]

    config_files = [conf1, conf2]

    configsmock = MagicMock(side_effect=config_files)
    mocker.patch("freqtrade.configuration.load_config.load_config_file", configsmock)

    arg_list = [
        "trade",
        "-c",
        "test_conf.json",
        "--config",
        "test2_conf.json",
    ]
    args = Arguments(arg_list).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    exchange_conf = default_conf["exchange"]
    assert validated_conf["exchange"]["name"] == exchange_conf["name"]
    assert validated_conf["exchange"]["key"] == exchange_conf["key"]
    assert validated_conf["exchange"]["secret"] == exchange_conf["secret"]
    assert validated_conf["exchange"]["pair_whitelist"] != conf1["exchange"]["pair_whitelist"]
    assert validated_conf["exchange"]["pair_whitelist"] == conf2["exchange"]["pair_whitelist"]

    assert "internals" in validated_conf


def test_from_config(default_conf, mocker, caplog) -> None:
    """测试从配置文件创建配置对象"""
    conf1 = deepcopy(default_conf)
    conf2 = deepcopy(default_conf)
    del conf1["exchange"]["key"]
    del conf1["exchange"]["secret"]
    del conf2["exchange"]["name"]
    conf2["exchange"]["pair_whitelist"] += ["NANO/BTC"]
    conf2["fiat_display_currency"] = "EUR"
    config_files = [conf1, conf2]
    mocker.patch("freqtrade.configuration.configuration.create_datadir", lambda c, x: x)

    configsmock = MagicMock(side_effect=config_files)
    mocker.patch("freqtrade.configuration.load_config.load_config_file", configsmock)

    validated_conf = Configuration.from_files(["test_conf.json", "test2_conf.json"])

    exchange_conf = default_conf["exchange"]
    assert validated_conf["exchange"]["name"] == exchange_conf["name"]
    assert validated_conf["exchange"]["key"] == exchange_conf["key"]
    assert validated_conf["exchange"]["secret"] == exchange_conf["secret"]
    assert validated_conf["exchange"]["pair_whitelist"] != conf1["exchange"]["pair_whitelist"]
    assert validated_conf["exchange"]["pair_whitelist"] == conf2["exchange"]["pair_whitelist"]
    assert validated_conf["fiat_display_currency"] == "EUR"
    assert "internals" in validated_conf
    assert isinstance(validated_conf["user_data_dir"], Path)


def test_from_recursive_files(testdatadir) -> None:
    """测试递归加载配置文件"""
    files = testdatadir / "testconfigs/testconfig.json"

    conf = Configuration.from_files([files])

    assert conf
    # 交易所配置来自"第一个配置"
    assert conf["exchange"]
    # 定价配置来自第二个配置
    assert conf["entry_pricing"]
    assert conf["entry_pricing"]["price_side"] == "same"
    assert conf["exit_pricing"]
    # 其他键来自pricing2，由pricing.json导入。pricing.json层级更高，因此优先
    assert conf["exit_pricing"]["price_side"] == "same"

    assert len(conf["config_files"]) == 4
    assert "testconfig.json" in conf["config_files"][0]
    assert "test_pricing_conf.json" in conf["config_files"][1]
    assert "test_base_config.json" in conf["config_files"][2]
    assert "test_pricing2_conf.json" in conf["config_files"][3]

    files = testdatadir / "testconfigs/recursive.json"
    with pytest.raises(OperationalException, match="检测到配置循环。"):
        load_from_files([files])


def test_print_config(default_conf, mocker, caplog) -> None:
    """测试打印配置功能"""
    conf1 = deepcopy(default_conf)
    # 从默认配置中删除非json元素
    del conf1["user_data_dir"]
    conf1["datadir"] = str(conf1["datadir"])
    config_files = [conf1]

    configsmock = MagicMock(side_effect=config_files)
    mocker.patch("freqtrade.configuration.configuration.create_datadir", lambda c, x: x)
    mocker.patch("freqtrade.configuration.configuration.load_from_files", configsmock)

    validated_conf = Configuration.from_files(["test_conf.json"])

    assert isinstance(validated_conf["user_data_dir"], Path)
    assert "user_data_dir" in validated_conf
    assert "original_config" in validated_conf
    assert isinstance(json.dumps(validated_conf["original_config"]), str)


def test_load_config_max_open_trades_minus_one(default_conf, mocker, caplog) -> None:
    """测试max_open_trades为-1的配置加载"""
    default_conf["max_open_trades"] = -1
    patched_configuration_load_config_file(mocker, default_conf)

    args = Arguments(["trade"]).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf["max_open_trades"] > 999999999
    assert validated_conf["max_open_trades"] == float("inf")
    assert "runmode" in validated_conf
    assert validated_conf["runmode"] == RunMode.DRY_RUN


def test_load_config_file_exception(mocker) -> None:
    """测试加载配置文件时的异常处理"""
    mocker.patch(
        "freqtrade.configuration.configuration.Path.open",
        MagicMock(side_effect=FileNotFoundError("文件未找到")),
    )

    with pytest.raises(OperationalException, match=r'.*配置文件 "somefile" 未找到!*'):
        load_config_file("somefile")


def test_load_config(default_conf, mocker) -> None:
    """测试基本配置加载"""
    del default_conf["strategy_path"]
    patched_configuration_load_config_file(mocker, default_conf)

    args = Arguments(["trade"]).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get("strategy_path") is None
    assert "edge" not in validated_conf


def test_load_config_with_params(default_conf, mocker) -> None:
    """测试带参数的配置加载"""
    patched_configuration_load_config_file(mocker, default_conf)

    arglist = [
        "trade",
        "--strategy",
        "TestStrategy",
        "--strategy-path",
        "/some/path",
        "--db-url",
        "sqlite:///someurl",
    ]
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get("strategy") == "TestStrategy"
    assert validated_conf.get("strategy_path") == "/some/path"
    assert validated_conf.get("db_url") == "sqlite:///someurl"

    # 测试配置中提供的生产环境db_url
    conf = default_conf.copy()
    conf["dry_run"] = False
    conf["db_url"] = "sqlite:///path/to/db.sqlite"
    patched_configuration_load_config_file(mocker, conf)

    arglist = ["trade", "--strategy", "TestStrategy", "--strategy-path", "/some/path"]
    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get("db_url") == "sqlite:///path/to/db.sqlite"

    # 测试配置中提供的模拟交易db_url
    conf = default_conf.copy()
    conf["dry_run"] = True
    conf["db_url"] = "sqlite:///path/to/db.sqlite"
    patched_configuration_load_config_file(mocker, conf)

    arglist = ["trade", "--strategy", "TestStrategy", "--strategy-path", "/some/path"]
    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get("db_url") == "sqlite:///path/to/db.sqlite"

    # 测试参数中提供的生产环境db_url
    conf = default_conf.copy()
    conf["dry_run"] = False
    del conf["db_url"]
    patched_configuration_load_config_file(mocker, conf)

    arglist = ["trade", "--strategy", "TestStrategy", "--strategy-path", "/some/path"]
    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get("db_url") == DEFAULT_DB_PROD_URL
    assert "runmode" in validated_conf
    assert validated_conf["runmode"] == RunMode.LIVE

    # 测试参数中提供的模拟交易db_url
    conf = default_conf.copy()
    conf["dry_run"] = True
    conf["db_url"] = DEFAULT_DB_PROD_URL
    patched_configuration_load_config_file(mocker, conf)

    arglist = ["trade", "--strategy", "TestStrategy", "--strategy-path", "/some/path"]
    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get("db_url") == DEFAULT_DB_DRYRUN_URL


@pytest.mark.parametrize(
    "config_value,expected,arglist",
    [
        (True, True, ["trade", "--dry-run"]),  # 保持配置不变
        (False, True, ["trade", "--dry-run"]),  # 覆盖配置
        (False, False, ["trade"]),  # 保持配置不变
        (True, True, ["trade"]),  # 保持配置不变
    ],
)
def test_load_dry_run(default_conf, mocker, config_value, expected, arglist) -> None:
    """测试模拟交易配置加载"""
    default_conf["dry_run"] = config_value
    patched_configuration_load_config_file(mocker, default_conf)

    configuration = Configuration(Arguments(arglist).get_parsed_arg())
    validated_conf = configuration.load_config()

    assert validated_conf["dry_run"] is expected
    assert validated_conf["runmode"] == (RunMode.DRY_RUN if expected else RunMode.LIVE)


def test_load_custom_strategy(default_conf, mocker, tmp_path) -> None:
    """测试加载自定义策略"""
    default_conf.update(
        {
            "strategy": "CustomStrategy",
            "strategy_path": f"{tmp_path}/strategies",
        }
    )
    patched_configuration_load_config_file(mocker, default_conf)

    args = Arguments(["trade"]).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get("strategy") == "CustomStrategy"
    assert validated_conf.get("strategy_path") == f"{tmp_path}/strategies"


def test_show_info(default_conf, mocker, caplog) -> None:
    """测试显示配置信息"""
    patched_configuration_load_config_file(mocker, default_conf)

    arglist = [
        "trade",
        "--strategy",
        "TestStrategy",
        "--db-url",
        "sqlite:///tmp/testdb",
    ]
    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    configuration.get_config()

    assert log_has('使用数据库: "sqlite:///tmp/testdb"', caplog)
    assert log_has("已启用模拟交易", caplog)


def test_setup_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    """测试无参数的配置设置"""
    patched_configuration_load_config_file(mocker, default_conf)

    arglist = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
    ]

    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()
    assert "max_open_trades" in config
    assert "stake_currency" in config
    assert "stake_amount" in config
    assert "exchange" in config
    assert "pair_whitelist" in config["exchange"]
    assert "datadir" in config
    assert "user_data_dir" in config
    assert log_has(f"使用数据目录: {config['datadir']} ...", caplog)
    assert "timeframe" in config
    assert not log_has("检测到参数 -i/--timeframe ...", caplog)

    assert "position_stacking" not in config
    assert not log_has("检测到参数 --enable-position-stacking ...", caplog)

    assert "timerange" not in config


def test_setup_configuration_with_arguments(mocker, default_conf, caplog, tmp_path) -> None:
    """测试带参数的配置设置"""
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.configuration.configuration.create_datadir", lambda c, x: x)
    mocker.patch(
        "freqtrade.configuration.configuration.create_userdata_dir",
        lambda x, *args, **kwargs: Path(x),
    )
    arglist = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--datadir",
        "/foo/bar",
        "--userdir",
        f"{tmp_path}/freqtrade",
        "--timeframe",
        "1m",
        "--enable-position-stacking",
        "--timerange",
        ":100",
        "--export",
        "trades",
        "--stake-amount",
        "unlimited",
    ]

    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()
    assert "max_open_trades" in config
    assert "stake_currency" in config
    assert "stake_amount" in config
    assert "exchange" in config
    assert "pair_whitelist" in config["exchange"]
    assert "datadir" in config
    assert log_has(f"使用数据目录: /foo/bar ...", caplog)
    assert log_has(f"使用用户数据目录: {tmp_path / 'freqtrade'} ...", caplog)
    assert "user_data_dir" in config

    assert "timeframe" in config
    assert log_has("检测到参数 -i/--timeframe ... 使用时间框架: 1m ...", caplog)

    assert "position_stacking" in config
    assert log_has("检测到参数 --enable-position-stacking ...", caplog)

    assert "timerange" in config
    assert log_has(f"检测到参数 --timerange: {config['timerange']} ...", caplog)

    assert "export" in config
    assert log_has(f"检测到参数 --export: {config['export']} ...", caplog)
    assert "stake_amount" in config
    assert config["stake_amount"] == "unlimited"


def test_setup_configuration_with_stratlist(mocker, default_conf, caplog) -> None:
    """测试带策略列表的配置设置"""
    patched_configuration_load_config_file(mocker, default_conf)

    arglist = [
        "backtesting",
        "--config",
        "config.json",
        "--timeframe",
        "1m",
        "--export",
        "trades",
        "--strategy-list",
        CURRENT_TEST_STRATEGY,
        "TestStrategy",
    ]

    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args, RunMode.BACKTEST)
    config = configuration.get_config()
    assert config["runmode"] == RunMode.BACKTEST
    assert "max_open_trades" in config
    assert "stake_currency" in config
    assert "stake_amount" in config
    assert "exchange" in config
    assert "pair_whitelist" in config["exchange"]
    assert "datadir" in config
    assert log_has(f"使用数据目录: {config['datadir']} ...", caplog)
    assert "timeframe" in config
    assert log_has("检测到参数 -i/--timeframe ... 使用时间框架: 1m ...", caplog)

    assert "strategy_list" in config
    assert log_has("使用包含2个策略的策略列表", caplog)

    assert "position_stacking" not in config

    assert "timerange" not in config

    assert "export" in config
    assert log_has(f"检测到参数 --export: {config['export']} ...", caplog)


def test_hyperopt_with_arguments(mocker, default_conf, caplog) -> None:
    """测试带参数的超参数优化配置"""
    patched_configuration_load_config_file(mocker, default_conf)

    arglist = [
        "hyperopt",
        "--epochs",
        "10",
        "--spaces",
        "all",
    ]
    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args, RunMode.HYPEROPT)
    config = configuration.get_config()

    assert "epochs" in config
    assert int(config["epochs"]) == 10
    assert log_has(
        "检测到参数 --epochs ... 将运行超参数优化，共10个周期 ...", caplog
    )

    assert "spaces" in config
    assert config["spaces"] == ["all"]
    assert log_has("检测到参数 -s/--spaces: ['all']", caplog)
    assert "runmode" in config
    assert config["runmode"] == RunMode.HYPEROPT


def test_cli_verbose_with_params(default_conf, mocker, caplog) -> None:
    """测试命令行详细日志参数"""
    patched_configuration_load_config_file(mocker, default_conf)

    # 阻止设置日志器
    mocker.patch("freqtrade.loggers.logging.config.dictConfig", MagicMock)
    arglist = ["trade", "-vvv"]
    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get("verbosity") == 3
    assert log_has("日志详细级别设置为3", caplog)


@pytest.mark.usefixtures("keep_log_config_loggers")
def test_set_logfile(default_conf, mocker, tmp_path):
    """测试设置日志文件"""
    default_conf["ft_tests_force_logging"] = True
    patched_configuration_load_config_file(mocker, default_conf)
    f = tmp_path / "test_file.log"
    assert not f.is_file()
    arglist = [
        "trade",
        "--logfile",
        str(f),
    ]
    args = Arguments(arglist).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf["logfile"] == str(f)
    assert f.is_file()
    try:
        f.unlink()
    except Exception:
        pass


def test_load_config_warn_forcebuy(default_conf, mocker, caplog) -> None:
    """测试加载包含强制买入配置的警告"""
    default_conf["force_entry_enable"] = True
    patched_configuration_load_config_file(mocker, default_conf)

    args = Arguments(["trade"]).get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get("force_entry_enable")
    assert log_has("`force_entry_enable` RPC消息已启用。", caplog)


def test_validate_default_conf(default_conf) -> None:
    """测试验证默认配置"""
    # 通过我们的验证器验证 - 允许设置默认值!
    validate_config_schema(default_conf)


@pytest.mark.parametrize("fiat", ["EUR", "USD", "", None])
def test_validate_fiat_currency_options(default_conf, fiat) -> None:
    """测试验证法币显示货币选项"""
    # 通过我们的验证器验证 - 允许设置默认值!
    if fiat is not None:
        default_conf["fiat_display_currency"] = fiat
    else:
        del default_conf["fiat_display_currency"]
    validate_config_schema(default_conf)


def test_validate_max_open_trades(default_conf):
    """测试验证最大开仓数配置"""
    default_conf["max_open_trades"] = float("inf")
    default_conf["stake_amount"] = "unlimited"
    with pytest.raises(
        OperationalException,
        match="`max_open_trades` 和 `stake_amount` 不能同时为无限。",
    ):
        validate_config_consistency(default_conf)


def test_validate_price_side(default_conf):
    """测试验证价格方向配置"""
    default_conf["order_types"] = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
    }
    # 默认配置应通过验证
    validate_config_consistency(default_conf)

    conf = deepcopy(default_conf)
    conf["order_types"]["entry"] = "market"
    with pytest.raises(
        OperationalException,
        match='市价入场订单需要 entry_pricing.price_side = "other"。',
    ):
        validate_config_consistency(conf)

    conf = deepcopy(default_conf)
    conf["order_types"]["exit"] = "market"
    with pytest.raises(
        OperationalException, match='市价出场订单需要 exit_pricing.price_side = "other"。',
    ):
        validate_config_consistency(conf)

    # 验证反向情况
    conf = deepcopy(default_conf)
    conf["order_types"]["exit"] = "market"
    conf["order_types"]["entry"] = "market"
    conf["exit_pricing"]["price_side"] = "bid"
    conf["entry_pricing"]["price_side"] = "ask"

    validate_config_consistency(conf)


def test_validate_tsl(default_conf):
    """测试验证追踪止损配置"""
    default_conf["stoploss"] = 0.0
    with pytest.raises(
        OperationalException,
        match="配置中的止损需要与0不同，以避免卖出订单出现问题。",
    ):
        validate_config_consistency(default_conf)
    default_conf["stoploss"] = -0.10

    default_conf["trailing_stop"] = True
    default_conf["trailing_stop_positive"] = 0
    default_conf["trailing_stop_positive_offset"] = 0

    default_conf["trailing_only_offset_is_reached"] = True
    with pytest.raises(
        OperationalException,
        match=r"配置中的 trailing_only_offset_is_reached 需要 "
        "配置中的 trailing_stop_positive_offset 大于0。",
    ):
        validate_config_consistency(default_conf)

    default_conf["trailing_stop_positive_offset"] = 0.01
    default_conf["trailing_stop_positive"] = 0.015
    with pytest.raises(
        OperationalException,
        match=r"配置中的 trailing_stop_positive_offset 需要 "
        "大于配置中的 trailing_stop_positive。",
    ):
        validate_config_consistency(default_conf)

    default_conf["trailing_stop_positive"] = 0.01
    default_conf["trailing_stop_positive_offset"] = 0.015
    validate_config_consistency(default_conf)

    # 0追踪止损正值 - 导致"订单将立即触发"
    default_conf["trailing_stop_positive"] = 0
    default_conf["trailing_stop_positive_offset"] = 0.02
    default_conf["trailing_only_offset_is_reached"] = False
    with pytest.raises(
        OperationalException,
        match="配置中的 trailing_stop_positive 需要与0不同，"
        "以避免卖出订单出现问题",
    ):
        validate_config_consistency(default_conf)


def test_validate_whitelist(default_conf):
    """测试验证白名单配置"""
    default_conf["runmode"] = RunMode.DRY_RUN
    # 测试常规情况 - 有白名单并使用StaticPairlist
    validate_config_consistency(default_conf)
    conf = deepcopy(default_conf)
    del conf["exchange"]["pair_whitelist"]
    # 测试错误情况
    with pytest.raises(
        OperationalException, match="StaticPairList需要设置pair_whitelist。"
    ):
        validate_config_consistency(conf)

    conf = deepcopy(default_conf)

    conf.update(
        {
            "pairlists": [
                {
                    "method": "VolumePairList",
                }
            ]
        }
    )
    # 动态白名单不应关心pair_whitelist
    validate_config_consistency(conf)
    del conf["exchange"]["pair_whitelist"]

    validate_config_consistency(conf)


def test_validate_ask_orderbook(default_conf, caplog) -> None:
    """测试验证卖单簿配置"""
    conf = deepcopy(default_conf)
    conf["exit_pricing"]["use_order_book"] = True
    conf["exit_pricing"]["order_book_min"] = 2
    conf["exit_pricing"]["order_book_max"] = 2

    validate_config_consistency(conf)
    assert log_has_re(r"已弃用: 请使用 `order_book_top` 代替.*", caplog)
    assert conf["exit_pricing"]["order_book_top"] == 2

    conf["exit_pricing"]["order_book_max"] = 5

    with pytest.raises(
        OperationalException, match=r"在exit_pricing中使用order_book_max != order_book_min.*"
    ):
        validate_config_consistency(conf)


def test_validate_time_in_force(default_conf, caplog) -> None:
    """测试验证订单有效期配置"""
    conf = deepcopy(default_conf)
    conf["order_time_in_force"] = {
        "buy": "gtc",
        "sell": "GTC",
    }
    validate_config_consistency(conf)
    assert log_has_re(r"已弃用: 使用 'buy' 和 'sell' 作为time_in_force的键.*", caplog)
    assert conf["order_time_in_force"]["entry"] == "gtc"
    assert conf["order_time_in_force"]["exit"] == "GTC"

    conf = deepcopy(default_conf)
    conf["order_time_in_force"] = {
        "buy": "GTC",
        "sell": "GTC",
    }
    conf["trading_mode"] = "futures"
    with pytest.raises(
        OperationalException,
        match=r"请将您的time_in_force设置迁移为使用 'entry' 和 'exit'\.",
    ):
        validate_config_consistency(conf)


def test__validate_order_types(default_conf, caplog) -> None:
    """测试验证订单类型配置"""
    conf = deepcopy(default_conf)
    conf["order_types"] = {
        "buy": "limit",
        "sell": "market",
        "forcesell": "market",
        "forcebuy": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    validate_config_consistency(conf)
    assert log_has_re(r"已弃用: 使用 'buy' 和 'sell' 作为order_types的键.*", caplog)
    assert conf["order_types"]["entry"] == "limit"
    assert conf["order_types"]["exit"] == "market"
    assert conf["order_types"]["force_entry"] == "limit"
    assert "buy" not in conf["order_types"]
    assert "sell" not in conf["order_types"]
    assert "forcebuy" not in conf["order_types"]
    assert "forcesell" not in conf["order_types"]

    conf = deepcopy(default_conf)
    conf["order_types"] = {
        "buy": "limit",
        "sell": "market",
        "forcesell": "market",
        "forcebuy": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    conf["trading_mode"] = "futures"
    with pytest.raises(
        OperationalException,
        match=r"请将您的order_types设置迁移为使用新的命名方式\.",
    ):
        validate_config_consistency(conf)


def test__validate_unfilledtimeout(default_conf, caplog) -> None:
    """测试验证未成交订单超时配置"""
    conf = deepcopy(default_conf)
    conf["unfilledtimeout"] = {
        "buy": 30,
        "sell": 35,
    }
    validate_config_consistency(conf)
    assert log_has_re(r"已弃用: 使用 'buy' 和 'sell' 作为unfilledtimeout的键.*", caplog)
    assert conf["unfilledtimeout"]["entry"] == 30
    assert conf["unfilledtimeout"]["exit"] == 35
    assert "buy" not in conf["unfilledtimeout"]
    assert "sell" not in conf["unfilledtimeout"]

    conf = deepcopy(default_conf)
    conf["unfilledtimeout"] = {
        "buy": 30,
        "sell": 35,
    }
    conf["trading_mode"] = "futures"
    with pytest.raises(
        OperationalException,
        match=r"请将您的unfilledtimeout设置迁移为使用新的命名方式\.",
    ):
        validate_config_consistency(conf)


def test__validate_pricing_rules(default_conf, caplog) -> None:
    """测试验证定价规则配置"""
    def_conf = deepcopy(default_conf)
    del def_conf["entry_pricing"]
    del def_conf["exit_pricing"]

    def_conf["ask_strategy"] = {
        "price_side": "ask",
        "use_order_book": True,
        "bid_last_balance": 0.5,
    }
    def_conf["bid_strategy"] = {
        "price_side": "bid",
        "use_order_book": False,
        "ask_last_balance": 0.7,
    }
    conf = deepcopy(def_conf)

    validate_config_consistency(conf)
    assert log_has_re(r"已弃用: 使用 'ask_strategy' 和 'bid_strategy' 已.*", caplog)
    assert conf["exit_pricing"]["price_side"] == "ask"
    assert conf["exit_pricing"]["use_order_book"] is True
    assert conf["exit_pricing"]["price_last_balance"] == 0.5
    assert conf["entry_pricing"]["price_side"] == "bid"
    assert conf["entry_pricing"]["use_order_book"] is False
    assert conf["entry_pricing"]["price_last_balance"] == 0.7
    assert "ask_strategy" not in conf
    assert "bid_strategy" not in conf

    conf = deepcopy(def_conf)

    conf["trading_mode"] = "futures"
    with pytest.raises(
        OperationalException, match=r"请将您的定价设置迁移为使用新的命名方式\."
    ):
        validate_config_consistency(conf)


def test__validate_freqai_include_timeframes(default_conf, caplog) -> None:
    """测试验证FreqAI包含的时间框架配置"""
    conf = deepcopy(default_conf)
    conf.update(
        {
            "freqai": {
                "enabled": True,
                "feature_parameters": {
                    "include_timeframes": ["1m", "5m"],
                    "include_corr_pairlist": [],
                },
                "data_split_parameters": {},
                "model_training_parameters": {},
            }
        }
    )
    with pytest.raises(OperationalException, match=r"主时间框架 .*"):
        validate_config_consistency(conf)
    # 验证通过
    conf.update({"timeframe": "1m"})
    validate_config_consistency(conf)

    # 确保基础时间框架在include_timeframes中
    conf["freqai"]["feature_parameters"]["include_timeframes"] = ["5m", "15m"]
    validate_config_consistency(conf)
    assert conf["freqai"]["feature_parameters"]["include_timeframes"] == ["1m", "5m", "15m"]

    conf.update({"analyze_per_epoch": True})
    with pytest.raises(
        OperationalException,
        match=r"使用analyze-per-epoch .* 不支持FreqAI策略。",
    ):
        validate_config_consistency(conf)


def test__validate_consumers(default_conf, caplog) -> None:
    """测试验证消息消费者配置"""
    conf = deepcopy(default_conf)
    conf.update({"external_message_consumer": {"enabled": True, "producers": []}})
    with pytest.raises(
        OperationalException, match="您必须指定至少1个要连接的生产者。"
    ):
        validate_config_consistency(conf)

    conf = deepcopy(default_conf)
    conf.update(
        {
            "external_message_consumer": {
                "enabled": True,
                "producers": [
                    {
                        "name": "default",
                        "host": "127.0.0.1",
                        "port": 8081,
                        "ws_token": "secret_ws_t0ken.",
                    },
                    {
                        "name": "default",
                        "host": "127.0.0.1",
                        "port": 8080,
                        "ws_token": "secret_ws_t0ken.",
                    },
                ],
            }
        }
    )
    with pytest.raises(
        OperationalException, match="生产者名称必须唯一。重复项: default"
    ):
        validate_config_consistency(conf)

    conf = deepcopy(default_conf)
    conf.update(
        {
            "process_only_new_candles": True,
            "external_message_consumer": {
                "enabled": True,
                "producers": [
                    {
                        "name": "default",
                        "host": "127.0.0.1",
                        "port": 8081,
                        "ws_token": "secret_ws_t0ken.",
                    }
                ],
            },
        }
    )
    validate_config_consistency(conf)
    assert log_has_re("为了使用外部数据获得最佳性能.*", caplog)


def test__validate_orderflow(default_conf) -> None:
    """测试验证订单流配置"""
    conf = deepcopy(default_conf)
    conf["exchange"]["use_public_trades"] = True
    with pytest.raises(
        ConfigurationError,
        match="使用公共交易时，Orderflow是必需的配置键。",
    ):
        validate_config_consistency(conf)

    conf.update(
        {
            "orderflow": {
                "scale": 0.5,
                "stacked_imbalance_range": 3,
                "imbalance_volume": 100,
                "imbalance_ratio": 3,
            }
        }
    )
    # 应该通过验证
    validate_config_consistency(conf)


def test_validate_edge_removal(default_conf):
    """测试验证Edge功能已移除"""
    default_conf["edge"] = {
        "enabled": True,
    }
    with pytest.raises(
        ConfigurationError,
        match="Edge不再受支持，并已在2025.6版本中从Freqtrade中移除。",
    ):
        validate_config_consistency(default_conf)


def test_load_config_test_comments() -> None:
    """测试加载带注释的配置"""
    config_file = Path(__file__).parents[0] / "config_test_comments.json"
    conf = load_config_file(str(config_file))

    assert conf


def test_load_config_default_exchange(all_conf) -> None:
    """测试加载默认交易所配置"""
    """
    config['exchange']子树包含必需的选项
    因此在配置中不能省略
    """
    del all_conf["exchange"]

    assert "exchange" not in all_conf

    with pytest.raises(ValidationError, match=r"'exchange' 是必需的属性"):
        validate_config_schema(all_conf)


def test_load_config_default_exchange_name(all_conf) -> None:
    """测试加载默认交易所名称配置"""
    """
    config['exchange']['name']选项是必需的
    因此在配置中不能省略
    """
    del all_conf["exchange"]["name"]

    assert "name" not in all_conf["exchange"]

    with pytest.raises(ValidationError, match=r"'name' 是必需的属性"):
        validate_config_schema(all_conf)


def test_load_config_stoploss_exchange_limit_ratio(all_conf) -> None:
    """测试加载止损交易所限价比例配置"""
    all_conf["order_types"]["stoploss_on_exchange_limit_ratio"] = 1.15

    with pytest.raises(ValidationError, match=r"1.15 大于最大值"):
        validate_config_schema(all_conf)


@pytest.mark.parametrize(
    "keys",
    [
        ("exchange", "key", ""),
        ("exchange", "secret", ""),
        ("exchange", "password", ""),
    ],
)
def test_load_config_default_subkeys(all_conf, keys) -> None:
    """测试加载默认子键配置"""
    """
    测试子路径中具有默认值的参数
    因此它们可以在配置中省略，并且默认值
    应该添加到配置中
    """
    # 获取第一级键
    key = keys[0]
    # 获取第二级键
    subkey = keys[1]

    del all_conf[key][subkey]

    assert subkey not in all_conf[key]

    validate_config_schema(all_conf)
    assert subkey in all_conf[key]
    assert all_conf[key][subkey] == keys[2]


def test_pairlist_resolving():
    """测试交易对列表解析"""
    arglist = ["download-data", "--pairs", "ETH/BTC", "XRP/BTC", "--exchange", "binance"]

    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args, RunMode.OTHER)
    config = configuration.get_config()

    assert config["pairs"] == ["ETH/BTC", "XRP/BTC"]
    assert config["exchange"]["pair_whitelist"] == ["ETH/BTC", "XRP/BTC"]
    assert config["exchange"]["name"] == "binance"


def test_pairlist_resolving_with_config(mocker, default_conf):
    """测试带配置的交易对列表解析"""
    patched_configuration_load_config_file(mocker, default_conf)
    arglist = [
        "download-data",
        "--config",
        "config.json",
    ]

    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()

    assert config["pairs"] == default_conf["exchange"]["pair_whitelist"]
    assert config["exchange"]["name"] == default_conf["exchange"]["name"]

    # 覆盖交易对
    arglist = [
        "download-data",
        "--config",
        "config.json",
        "--pairs",
        "ETH/BTC",
        "XRP/BTC",
    ]

    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()

    assert config["pairs"] == ["ETH/BTC", "XRP/BTC"]
    assert config["exchange"]["name"] == default_conf["exchange"]["name"]


def test_pairlist_resolving_with_config_pl(mocker, default_conf):
    """测试带配置文件的交易对列表解析"""
    patched_configuration_load_config_file(mocker, default_conf)

    arglist = [
        "download-data",
        "--config",
        "config.json",
        "--pairs-file",
        "tests/testdata/pairs.json",
    ]

    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()
    assert len(config["pairs"]) == 23
    assert "ETH/BTC" in config["pairs"]
    assert "XRP/BTC" in config["pairs"]
    assert config["exchange"]["name"] == default_conf["exchange"]["name"]


def test_pairlist_resolving_with_config_pl_not_exists(mocker, default_conf):
    """测试交易对文件不存在时的解析"""
    patched_configuration_load_config_file(mocker, default_conf)

    arglist = [
        "download-data",
        "--config",
        "config.json",
        "--pairs-file",
        "tests/testdata/pairs_doesnotexist.json",
    ]

    args = Arguments(arglist).get_parsed_arg()

    with pytest.raises(OperationalException, match=r"未找到路径为.*的交易对文件"):
        configuration = Configuration(args)
        configuration.get_config()


def test_pairlist_resolving_fallback(mocker, tmp_path):
    """测试交易对列表解析的回退机制"""
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    mocker.patch.object(Path, "open", MagicMock(return_value=MagicMock()))
    mocker.patch(
        "freqtrade.configuration.configuration.load_file",
        MagicMock(return_value=["XRP/BTC", "ETH/BTC"]),
    )
    arglist = ["download-data", "--exchange", "binance"]

    args = Arguments(arglist).get_parsed_arg()
    # 如果config.json存在，修复不稳定的测试
    args["config"] = None

    configuration = Configuration(args, RunMode.OTHER)
    config = configuration.get_config()

    assert config["pairs"] == ["ETH/BTC", "XRP/BTC"]
    assert config["exchange"]["name"] == "binance"
    assert config["datadir"] == tmp_path / "user_data/data/binance"


@pytest.mark.parametrize(
    "setting",
    [
        ("webhook", "webhookbuy", "testWEbhook", "webhook", "webhookentry", "testWEbhook"),
        (
            "ask_strategy",
            "ignore_buying_expired_candle_after",
            5,
            None,
            "ignore_buying_expired_candle_after",
            6,
        ),
    ],
)
def test_process_temporary_deprecated_settings(mocker, default_conf, setting, caplog):
    """测试处理临时弃用的设置"""
    patched_configuration_load_config_file(mocker, default_conf)

    # 为新设置和已弃用设置创建部分
    # (它们可能不存在于配置中)
    default_conf[setting[0]] = {}
    default_conf[setting[3]] = {}

    # 分配已弃用设置
    default_conf[setting[0]][setting[1]] = setting[2]
    # 分配新设置
    if setting[3]:
        default_conf[setting[3]][setting[4]] = setting[5]
    else:
        default_conf[setting[4]] = setting[5]

    # 新设置和已弃用设置是冲突的
    with pytest.raises(OperationalException, match=r"已弃用"):
        process_temporary_deprecated_settings(default_conf)

    caplog.clear()

    # 删除新设置
    if setting[3]:
        del default_conf[setting[3]][setting[4]]
    else:
        del default_conf[setting[4]]

    process_temporary_deprecated_settings(default_conf)
    assert log_has_re("已弃用", caplog)
    # 新设置的值应设置为
    # 已弃用设置的值
    if setting[3]:
        assert default_conf[setting[3]][setting[4]] == setting[2]
    else:
        assert default_conf[setting[4]] == setting[2]


@pytest.mark.parametrize(
    "setting",
    [
        ("experimental", "use_sell_signal", False),
        ("experimental", "sell_profit_only", True),
        ("experimental", "ignore_roi_if_buy_signal", True),
    ],
)
def test_process_removed_settings(mocker, default_conf, setting):
    """测试处理已移除的设置"""
    patched_configuration_load_config_file(mocker, default_conf)

    # 为新设置和已弃用设置创建部分
    # (它们可能不存在于配置中)
    default_conf[setting[0]] = {}
    # 分配已移除设置
    default_conf[setting[0]][setting[1]] = setting[2]

    # 新设置和已弃用设置是冲突的
    with pytest.raises(OperationalException, match=r"设置 .* 已移动"):
        process_temporary_deprecated_settings(default_conf)


def test_check_conflicting_settings(mocker, default_conf, caplog):
    """测试检查冲突设置"""
    patched_configuration_load_config_file(mocker, default_conf)

    # 为新设置和已弃用设置创建部分
    # (它们可能不存在于配置中)
    default_conf["sectionA"] = {}
    default_conf["sectionB"] = {}
    # 分配新设置
    default_conf["sectionA"]["new_setting"] = "valA"
    # 分配已弃用设置
    default_conf["sectionB"]["deprecated_setting"] = "valB"

    # 新设置和已弃用设置是冲突的
    with pytest.raises(OperationalException, match=r"已弃用"):
        check_conflicting_settings(
            default_conf, "sectionB", "deprecated_setting", "sectionA", "new_setting"
        )

    caplog.clear()

    # 删除新设置(已弃用设置存在)
    del default_conf["sectionA"]["new_setting"]
    check_conflicting_settings(
        default_conf, "sectionB", "deprecated_setting", "sectionA", "new_setting"
    )
    assert not log_has_re("已弃用", caplog)
    assert "new_setting" not in default_conf["sectionA"]

    caplog.clear()

    # 分配新设置
    default_conf["sectionA"]["new_setting"] = "valA"
    # 删除已弃用设置
    del default_conf["sectionB"]["deprecated_setting"]
    check_conflicting_settings(
        default_conf, "sectionB", "deprecated_setting", "sectionA", "new_setting"
    )
    assert not log_has_re("已弃用", caplog)
    assert default_conf["sectionA"]["new_setting"] == "valA"


def test_process_deprecated_setting(mocker, default_conf, caplog):
    """测试处理已弃用设置"""
    patched_configuration_load_config_file(mocker, default_conf)

    # 为新设置和已弃用设置创建部分
    # (它们可能不存在于配置中)
    default_conf["sectionA"] = {}
    default_conf["sectionB"] = {}
    # 分配已弃用设置
    default_conf["sectionB"]["deprecated_setting"] = "valB"

    # 新设置和已弃用设置都存在
    process_deprecated_setting(
        default_conf, "sectionB", "deprecated_setting", "sectionA", "new_setting"
    )
    assert log_has_re("已弃用", caplog)
    # 新设置的值应设置为
    # 已弃用设置的值
    assert default_conf["sectionA"]["new_setting"] == "valB"
    # 旧设置已移除
    assert "deprecated_setting" not in default_conf["sectionB"]

    caplog.clear()

    # 删除新设置(已弃用设置存在)
    del default_conf["sectionA"]["new_setting"]
    default_conf["sectionB"]["deprecated_setting"] = "valB"
    process_deprecated_setting(
        default_conf, "sectionB", "deprecated_setting", "sectionA", "new_setting"
    )
    assert log_has_re("已弃用", caplog)
    # 新设置的值应设置为
    # 已弃用设置的值
    assert default_conf["sectionA"]["new_setting"] == "valB"

    caplog.clear()

    # 分配新设置
    default_conf["sectionA"]["new_setting"] = "valA"
    # 删除已弃用设置
    default_conf["sectionB"].pop("deprecated_setting", None)
    process_deprecated_setting(
        default_conf, "sectionB", "deprecated_setting", "sectionA", "new_setting"
    )
    assert not log_has_re("已弃用", caplog)
    assert default_conf["sectionA"]["new_setting"] == "valA"

    caplog.clear()
    # 测试移动到根目录
    default_conf["sectionB"]["deprecated_setting2"] = "DeadBeef"
    process_deprecated_setting(default_conf, "sectionB", "deprecated_setting2", None, "new_setting")

    assert log_has_re("已弃用", caplog)
    assert default_conf["new_setting"]


def test_process_removed_setting(mocker, default_conf, caplog):
    """测试处理已移除设置"""
    patched_configuration_load_config_file(mocker, default_conf)

    # 为新设置和已弃用设置创建部分
    # (它们可能不存在于配置中)
    default_conf["sectionA"] = {}
    default_conf["sectionB"] = {}
    # 分配新设置
    default_conf["sectionB"]["somesetting"] = "valA"

    # 只有新设置存在(什么都不应该发生)
    process_removed_setting(default_conf, "sectionA", "somesetting", "sectionB", "somesetting")
    # 分配已移除设置
    default_conf["sectionA"]["somesetting"] = "valB"

    with pytest.raises(OperationalException, match=r"设置 .* 已移动"):
        process_removed_setting(default_conf, "sectionA", "somesetting", "sectionB", "somesetting")


def test_process_deprecated_ticker_interval(default_conf, caplog):
    """测试处理已弃用的ticker_interval设置"""
    message = "已弃用: 请使用 'timeframe' 代替 'ticker_interval。"
    config = deepcopy(default_conf)

    process_temporary_deprecated_settings(config)
    assert not log_has(message, caplog)

    del config["timeframe"]
    config["ticker_interval"] = "15m"
    with pytest.raises(
        OperationalException, match=r"已弃用: 检测到 'ticker_interval'。请使用.*"
    ):
        process_temporary_deprecated_settings(config)


def test_process_deprecated_protections(default_conf, caplog):
    """测试处理已弃用的protections设置"""
    message = "已弃用: 在配置中设置 'protections' 已被弃用。"
    config = deepcopy(default_conf)
    process_temporary_deprecated_settings(config)
    assert not log_has(message, caplog)

    config["protections"] = []
    with pytest.raises(ConfigurationError, match=message):
        process_temporary_deprecated_settings(config)


def test_flat_vars_to_nested_dict(caplog):
    """测试将扁平变量转换为嵌套字典"""
    test_args = {
        "FREQTRADE__EXCHANGE__SOME_SETTING": "true",
        "FREQTRADE__EXCHANGE__SOME_FALSE_SETTING": "false",
        "FREQTRADE__EXCHANGE__CONFIG__whatEver": "sometime",  # 小写
        # 保留ccxt_config的大小写
        "FREQTRADE__EXCHANGE__CCXT_CONFIG__httpsProxy": "something",
        "FREQTRADE__EXIT_PRICING__PRICE_SIDE": "bid",
        "FREQTRADE__EXIT_PRICING__cccc": "500",
        "FREQTRADE__STAKE_AMOUNT": "200.05",
        "FREQTRADE__TELEGRAM__CHAT_ID": "2151",
        "NOT_RELEVANT": "200.0",  # 将被忽略
        "FREQTRADE__ARRAY": '[{"name":"default","host":"xxx"}]',
        "FREQTRADE__EXCHANGE__PAIR_WHITELIST": '["BTC/USDT", "ETH/USDT"]',
        # 由于尾随逗号而失败
        "FREQTRADE__ARRAY_TRAIL_COMMA": '[{"name":"default","host":"xxx",}]',
        # 对象失败
        "FREQTRADE__OBJECT": '{"name":"default","host":"xxx"}',
    }
    expected = {
        "stake_amount": 200.05,
        "exit_pricing": {
            "price_side": "bid",
            "cccc": 500,
        },
        "exchange": {
            "config": {
                "whatever": "sometime",
            },
            "ccxt_config": {
                "httpsProxy": "something",
            },
            "some_setting": True,
            "some_false_setting": False,
            "pair_whitelist": ["BTC/USDT", "ETH/USDT"],
        },
        "telegram": {"chat_id": "2151"},
        "array": [{"name": "default", "host": "xxx"}],
        "object": '{"name":"default","host":"xxx"}',
        "array_trail_comma": '[{"name":"default","host":"xxx",}]',
    }
    res = _flat_vars_to_nested_dict(test_args, ENV_VAR_PREFIX)
    assert res == expected

    assert log_has("加载变量 'FREQTRADE__EXCHANGE__SOME_SETTING'", caplog)
    assert not log_has("加载变量 'NOT_RELEVANT'", caplog)


def test_setup_hyperopt_freqai(mocker, default_conf) -> None:
    """测试设置超参数优化与FreqAI"""
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.configuration.configuration.create_datadir", lambda c, x: x)
    mocker.patch(
        "freqtrade.configuration.configuration.create_userdata_dir",
        lambda x, *args, **kwargs: Path(x),
    )
    arglist = [
        "hyperopt",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--timerange",
        "20220801-20220805",
        "--freqaimodel",
        "LightGBMRegressorMultiTarget",
        "--analyze-per-epoch",
    ]

    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()
    config["freqai"] = {"enabled": True}
    with pytest.raises(
        OperationalException, match=r".*analyze-per-epoch参数不支持.*"
    ):
        validate_config_consistency(config)


def test_setup_freqai_backtesting(mocker, default_conf) -> None:
    """测试设置FreqAI回测"""
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.configuration.configuration.create_datadir", lambda c, x: x)
    mocker.patch(
        "freqtrade.configuration.configuration.create_userdata_dir",
        lambda x, *args, **kwargs: Path(x),
    )
    arglist = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--timerange",
        "20220801-20220805",
        "--freqaimodel",
        "LightGBMRegressorMultiTarget",
        "--freqai-backtest-live-models",
    ]

    args = Arguments(arglist).get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()
    config["runmode"] = RunMode.BACKTEST

    with pytest.raises(
        OperationalException, match=r".*--freqai-backtest-live-models参数仅.*"
    ):
        validate_config_consistency(config)

    conf = deepcopy(config)
    conf["freqai"] = {"enabled": True}
    with pytest.raises(
        OperationalException, match=r".* timerange参数不支持与 .*"
    ):
        validate_config_consistency(conf)

    conf["timerange"] = None
    conf["freqai_backtest_live_models"] = False

    with pytest.raises(
        OperationalException, match=r".* 如果您打算使用FreqAI，请传递--timerange .*"
    ):
        validate_config_consistency(conf)


def test_sanitize_config(default_conf_usdt):
    """测试清理配置（隐藏敏感信息）"""
    assert default_conf_usdt["exchange"]["key"] != "已屏蔽"
    res = sanitize_config(default_conf_usdt)
    # 未修改原始字典
    assert default_conf_usdt["exchange"]["key"] != "已屏蔽"
    assert "accountId" not in default_conf_usdt["exchange"]

    assert res["exchange"]["key"] == "已屏蔽"
    assert res["exchange"]["secret"] == "已屏蔽"
    # 未添加不存在的键
    assert "accountId" not in res["exchange"]

    res = sanitize_config(default_conf_usdt, show_sensitive=True)
    assert res["exchange"]["key"] == default_conf_usdt["exchange"]["key"]
    assert res["exchange"]["secret"] == default_conf_usdt["exchange"]["secret"]


def test_remove_exchange_credentials(default_conf) -> None:
    """测试移除交易所凭证"""
    conf = deepcopy(default_conf)
    remove_exchange_credentials(conf["exchange"], False)

    assert conf["exchange"]["key"] != ""
    assert conf["exchange"]["secret"] != ""

    remove_exchange_credentials(conf["exchange"], True)
    assert conf["exchange"]["key"] == ""
    assert conf["exchange"]["secret"] == ""
    assert conf["exchange"].get("password", "") == ""
    assert conf["exchange"].get("uid", "") == ""