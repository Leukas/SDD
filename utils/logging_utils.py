# logging_utils.py
import os
import sys
import time
import datetime
import logging
from transformers.trainer_callback import TrainerCallback
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

logger = logging.getLogger(__name__)


def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"[GPU: {info.used//1024**2}/{info.total//1024**2} MB]"

class LogFlushCallback(TrainerCallback):
    """ Like printer callback, but with logger and flushes the logs every call """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            self.logger.info(logs)
            sys.stdout.flush()


def load_logger(logger):
    class LogFormatter():
        def __init__(self):
            self.start_time = time.time()

        def format(self, msg):
            time_passed = round(msg.created - self.start_time)
            gpu_usage = get_gpu_utilization()
            prefix = "[%s - %s]" % (
                time.strftime('%x %X'),
                datetime.timedelta(seconds=time_passed)
            )
            prefix = prefix + gpu_usage
            msg_text = msg.getMessage()
            return "%s %s" % (prefix, msg_text) if msg_text else ''

    logger.setLevel(logging.INFO)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(LogFormatter())
    logger.addHandler(log_handler)
    return logger