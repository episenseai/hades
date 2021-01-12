import hydra


class _Conf_Helper(type):
    @property
    def conf(cls):
        return cls._conf

    @property
    def app(cls):
        return cls._conf.app

    @property
    def server(cls):
        return cls._conf.server

    @property
    def redis(cls):
        return cls._conf.redis

    @property
    def mongo(cls):
        return cls._conf.mongo

    @property
    def stages(cls):
        return cls._conf.app.stages

    @property
    def ssl(cls):
        return {"cert": cls.server.ssl["cert"], "key": cls.server.ssl["key"]}


class Conf(metaclass=_Conf_Helper):
    _conf = None


# Parse hydra config from `conf` folder in the project root
@hydra.main(config_path="conf/config.yaml")
def hydra_cfg_parse(cfg):
    Conf._conf = cfg


hydra_cfg_parse()

__version__ = "0.1.0"
