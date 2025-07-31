"""Freqai多输出分类器"""

import numpy as np
from sklearn.base import is_classifier
from sklearn.multioutput import MultiOutputClassifier, _fit_estimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import has_fit_parameter, validate_data

from freqtrade.exceptions import OperationalException


class FreqaiMultiOutputClassifier(MultiOutputClassifier):
    def fit(self, X, y, sample_weight=None, fit_params=None):
        """
        对数据拟合模型，每个输出变量单独拟合。
        参数
        ----------
        X : 形状为 (n_samples, n_features) 的 {数组-like, 稀疏矩阵}
            输入数据。
        y : 形状为 (n_samples, n_outputs) 的 {数组-like, 稀疏矩阵}
            多输出目标。指示矩阵用于多标签估计。
        sample_weight : 形状为 (n_samples,) 的数组-like, 默认=None
            样本权重。如果为 `None`，则样本权重相等。
            仅在基础分类器支持样本权重时有效。
        fit_params : 用于fit_params的字典列表
            传递给每个步骤的 ``estimator.fit`` 方法的参数。
            每个字典可以包含相同或不同的值（例如不同的评估集或初始模型）
            .. versionadded:: 0.23
        返回
        -------
        self : 对象
            返回拟合后的实例。
        """

        if not hasattr(self.estimator, "fit"):
            raise ValueError("基础估计器应实现fit方法")

        y = validate_data(self, X="no_validation", y=y, multi_output=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError(
                "对于多输出回归，y必须至少有两个维度，但只有一个维度。"
            )

        if sample_weight is not None and not has_fit_parameter(self.estimator, "sample_weight"):
            raise ValueError("基础估计器不支持样本权重。")

        if not fit_params:
            fit_params = [None] * y.shape[1]

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(self.estimator, X, y[:, i], sample_weight,** fit_params[i])
            for i in range(y.shape[1])
        )

        self.classes_ = []
        for estimator in self.estimators_:
            self.classes_.extend(estimator.classes_)
        if len(set(self.classes_)) != len(self.classes_):
            raise OperationalException(
                f"跨目标的类别标签必须唯一: {self.classes_}"
            )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict_proba(self, X):
        """
        获取预测概率并将数组水平堆叠
        """
        results = np.hstack(super().predict_proba(X))
        return np.squeeze(results)

    def predict(self, X):
        """
        获取预测结果并压缩为二维数组
        """
        results = super().predict(X)
        return np.squeeze(results)