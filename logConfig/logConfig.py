#!/usr/local/anaconda3/bin/python3.9
# -*- coding: UTF-8 -*-
# @Project name: FaceRecognition
# @Filename: logConfig.py
# @Author: sxy
# @Date: 2023-05-29 16:21

import sys
sys.path.append("/home/sxy/.conda/envs/logger/lib/python3.9/site-packages/")


level_config = {
    "logger": {
        "root": "DEBUG",
        "FaceRecognition": "INFO"
    },
    "console": {
        "root": "DEBUG",
        "FaceRecognition": "INFO"
    }
}


log_config = {
    "version": 1,
    "formatters": {
        "FaceRecognitionFormatter": {
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
            "formatter": "FaceRecognitionFormatter",
            "stream": "ext://sys.stdout"
        },
        "coloredHandler": {
            "class": "logging.StreamHandler",
            "level": level_config["console"]["FaceRecognition"],
            "formatter": "coloredFormatter",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": level_config["logger"]["root"],
        "handlers": ["coloredHandler"]
    },
    "loggers": {
        "FaceRecognition": {
            "level": level_config["logger"]["FaceRecognition"],
            "propagate": 0,
            "handlers": ["coloredHandler"]
        }
    },
    "incremental": False,
    "disable_existing_loggers": False
}
