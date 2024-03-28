from config.train import TrainConfig

class GlobalConfig:
    config = TrainConfig()

    @staticmethod
    def set(config):
        GlobalConfig.config = config