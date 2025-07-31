"""PyTorch模型基类"""

import logging
from abc import ABC, abstractmethod

import torch

from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor


logger = logging.getLogger(__name__)


class BasePyTorchModel(IFreqaiModel, ABC):
    """
    PyTorch类型模型的基类。
    用户*必须*继承此类并实现fit()、predict()方法以及
    data_convertor属性。
    """

    def __init__(self,** kwargs):
        super().__init__(config=kwargs["config"])
        self.dd.model_type = "pytorch"
        self.device = (
            "mps"
            if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        test_size = self.freqai_info.get("data_split_parameters", {}).get("test_size")
        self.splits = ["train", "test"] if test_size != 0 else ["train"]
        self.window_size = self.freqai_info.get("conv_width", 1)

    @property
    @abstractmethod
    def data_convertor(self) -> PyTorchDataConvertor:
        """
        负责将`*_features`和`*_labels` pandas数据帧
        转换为pytorch张量的类。
        """
        raise NotImplementedError("抽象属性")