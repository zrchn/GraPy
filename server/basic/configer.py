import yaml
from consts import CONFIG_PATH
from omegaconf import OmegaConf
from loguru import logger

class Configer:
    omega = None

    def __init__(self):
        self.reload_configs()

    def reload_configs(self):
        logger.debug(f'loading configs from {CONFIG_PATH}')
        try:
            self.omega = OmegaConf.load(CONFIG_PATH)
        except:
            raise ValueError(f'检查是否有提供configs.yaml并在consts中配置路径')
        logger.debug(f'configs: {self.omega}')

    def __getattr__(self, name):
        return getattr(self.omega, name)
configer = Configer()