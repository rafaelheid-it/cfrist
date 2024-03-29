from typing import Union
from config.train import TrainConfig
from config.test import TestConfig

class GlobalConfig:
    # Either config.train.TrainConfig or config.test.TestConfig
    config = None

    @staticmethod
    def set(config: Union[TrainConfig, TestConfig]):
        """Set used config to TrainConfig or TestConfig."""
        GlobalConfig.config = config