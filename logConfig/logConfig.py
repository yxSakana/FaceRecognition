#!/usr/local/anaconda3/bin/python3.9
# -*- coding: UTF-8 -*-
# @Project name: KdSystem
# @Filename: logConfig.py
# @Author: sxy
# @Date: 2023-05-29 16:21

import sys
sys.path.append("/home/sxy/.conda/envs/logger/lib/python3.9/site-packages/")
import colorlog


level_config = {
    "logger": {
        "root": "DEBUG",
        "KdSystem": "INFO"
    },
    "console": {
        "root": "DEBUG",
        "KdSystem": "INFO"
    }
}


log_config = {
    "version": 1,
    "formatters": {
        "KdSystemFormatter": {
            "format": "[%(asctime)s]%(name)s[%(levelname)s]: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "coloredFormatter": {
            "()": "colorlog.ColoredFormatter",
            "format": "${log_color}[${asctime}]${name_log_color}${name}${levelname_log_color}[${levelname}]: ${message_log_color}${message}",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "log_colors": {
                'DEBUG': 'white',
                'INFO': 'white',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            },
            "secondary_log_colors": {
                "message": {
                    "INFO": "blue"
                },
                "name": {
                    "INFO": "purple"
                },
                "levelname": {
                    "INFO": "green"
                }
            },
            "style": "$"
        }
    },
    "filters": {},
    "handlers": {
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "level": level_config["console"]["root"],
            "formatter": "KdSystemFormatter",
            "stream": "ext://sys.stdout"
        },
        "coloredHandler": {
            "class": "logging.StreamHandler",
            "level": level_config["console"]["KdSystem"],
            "formatter": "coloredFormatter",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": level_config["logger"]["root"],
        "handlers": ["coloredHandler"]
    },
    "loggers": {
        "KdSystem": {
            "level": level_config["logger"]["KdSystem"],
            "propagate": 0,
            "handlers": ["coloredHandler"]
        }
    },
    "incremental": False,
    "disable_existing_loggers": False
}
