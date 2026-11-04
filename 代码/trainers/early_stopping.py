"""
早停机制模块
参考 RARE-UNet 的设计
"""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    早停机制

    监控验证指标，当指标不再改善时停止训练
    """

    def __init__(
        self,
        patience: int = 50,
        delta: float = 1e-4,
        criterion: str = "dice",
        verbose: bool = True,
    ):
        """
        Args:
            patience: 耐心轮数（多少轮没有改善则停止）
            delta: 最小改善量
            criterion: 监控的指标名称
            verbose: 是否打印日志
        """
        self.patience = patience
        self.delta = delta
        self.criterion = criterion
        self.verbose = verbose

        self.counter = 0
        self.early_stop = False

        # 判断是最大化还是最小化
        if criterion == "loss":
            self.mode = "min"
            self.best_score = float("inf")
        else:
            self.mode = "max"
            self.best_score = -float("inf")

    def __call__(self, val_loss: float, val_metric: float) -> tuple:
        """
        检查是否应该早停

        Args:
            val_loss: 验证损失
            val_metric: 验证指标

        Returns:
            (early_stop: bool, improved: bool)
        """
        # 选择要比较的值
        score = val_loss if self.mode == "min" else val_metric

        # 判断是否改善
        if self._is_improved(score):
            if self.verbose:
                logger.info(
                    f"{self.criterion} improved: {self.best_score:.4f} -> {score:.4f}"
                )
            self.best_score = score
            self.counter = 0
            improved = True
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"No improvement in {self.criterion} "
                    f"({self.counter}/{self.patience})"
                )
            improved = False

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered")

        return self.early_stop, improved

    def _is_improved(self, score: float) -> bool:
        """判断是否有改善"""
        if self.mode == "min":
            return score < self.best_score - self.delta
        else:
            return score > self.best_score + self.delta
