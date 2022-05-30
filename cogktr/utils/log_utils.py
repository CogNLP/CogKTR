import logging
import logging.config
import os
import datetime

logger=logging.getLogger()

def init_logger(log_file=None):
    """
    根据给定的log文件配置logger，日志信息同时输出至文件与命令行
    :param log_file: 日志输出位置
    """
    log_format = logging.Formatter("[\033[032m%(asctime)s\033[0m %(levelname)s] %(module)s.%(funcName)s %(message)s")
    LOGGING_DICT = {
        'version': 1,
        'disable_existing_loggers': True, # 是否禁用已经存在的日志器
        'formatters': {                   # 配置输出格式
            'standard': {
                'format': "[\033[032m%(asctime)s\033[0m %(levelname)s] %(module)s.%(funcName)s %(message)s"
            },
        },
        'filters': {},
        'handlers': {                     # 处理器设置
            'stream': {                   # 命令行输出处理器
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
            'file': {                     # 文件输出处理器
                'level': 20,
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'standard',
                'filename': log_file,
                'maxBytes': 1024 * 1024 * 5,
                'backupCount': 5,
                'encoding': 'utf-8',
            },
        },

        'loggers': {
            '': {
                'handlers': ['stream', 'file'],
                'level': 'INFO',
                'propagate': True,
            },
        },
    }
    if log_file is None:
        LOGGING_DICT['loggers']['']['handlers'] = ['stream']
        del LOGGING_DICT['handlers']['file']
    logging.config.dictConfig(LOGGING_DICT)
    logger = logging.getLogger()
    return logger





