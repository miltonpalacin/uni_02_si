# coding: UTF-8

import time
from datetime import datetime
import logging
import logging.handlers
import inspect
from alglib.help import config

TRACE = 100


class LocalLogger:

    def __init__(self):
        self.__logger_log = None
        self.__logger_trace = None
        self.__logger_debug = None
        self.__logger_line = None
        self.__logger_csv = None
        # __formatter__ = logging.Formatter('%(levelname)s:%(message)s')

    @property
    def logger_log(self):
        if self.__logger_log:
            return self.__logger_log
        __log_filename__ = config.SETTING.LOG_PATH + "\\" + config.SETTING.APP + ".log"
        __formatter_log__ = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        __handler_log = logging.handlers.TimedRotatingFileHandler(__log_filename__, when="D", interval=1)
        __handler_log.setFormatter(__formatter_log__)
        self.__logger_log = logging.getLogger(config.SETTING.APP+"_LOG")
        self.__logger_log.addHandler(__handler_log)
        self.__logger_log.setLevel(logging.INFO)
        return self.__logger_log

    # @logger_log.setter
    # def logger_log(self, val):
    #     self.logger_log = val

    @property
    def logger_trace(self):
        if self.__logger_trace:
            return self.__logger_trace
        __trace_filename__ = config.SETTING.LOG_PATH + "\\" + config.SETTING.APP + ".trace"
        __formatter_trace__ = logging.Formatter('%(asctime)s %(message)s')
        __handler_trace = logging.handlers.TimedRotatingFileHandler(__trace_filename__, when="D", interval=1)
        __handler_trace.setFormatter(__formatter_trace__)
        self.__logger_trace = logging.getLogger(config.SETTING.APP+"_TRACE")
        self.__logger_trace.addHandler(__handler_trace)
        self.__logger_trace.setLevel(logging.INFO)
        return self.__logger_trace

    @property
    def logger_debug(self):
        if self.__logger_debug:
            return self.__logger_debug
        __debug_filename__ = config.SETTING.LOG_PATH + "\\" + config.SETTING.APP + ".debug"
        __formatter_debug__ = logging.Formatter('%(asctime)s %(message)s')
        __handler_debug = logging.handlers.TimedRotatingFileHandler(__debug_filename__, when="D", interval=1)
        __handler_debug.setFormatter(__formatter_debug__)
        self.__logger_debug = logging.getLogger(config.SETTING.APP+"_DEBUG")
        self.__logger_debug.addHandler(__handler_debug)
        self.__logger_debug.setLevel(logging.INFO)
        return self.__logger_debug

    @property
    def logger_line(self):
        if self.__logger_line:
            return self.__logger_line
        __line_filename__ = config.SETTING.LOG_PATH + "\\" + config.SETTING.APP + ".line"
        __formatter_line__ = logging.Formatter('%(asctime)s %(message)s')
        __handler_line = logging.handlers.TimedRotatingFileHandler(__line_filename__, when="D", interval=1)
        __handler_line.setFormatter(__formatter_line__)
        self.__logger_line = logging.getLogger(config.SETTING.APP+"_LINE")
        self.__logger_line.addHandler(__handler_line)
        self.__logger_line.setLevel(logging.INFO)
        return self.__logger_line

    @property
    def logger_csv(self):
        if self.__logger_csv:
            return self.__logger_csv
        __csv_filename__ = config.SETTING.LOG_PATH + "\\" + config.SETTING.APP + ".csv"
        __formatter_csv__ = logging.Formatter('%(message)s')
        __handler_csv = logging.handlers.TimedRotatingFileHandler(__csv_filename__, when="D", interval=1)
        __handler_csv.setFormatter(__formatter_csv__)
        self.__logger_csv = logging.getLogger(config.SETTING.APP+"_CSV")
        self.__logger_csv.addHandler(__handler_csv)
        self.__logger_csv.setLevel(logging.INFO)
        return self.__logger_csv


LOG = LocalLogger()


def trace(*msg):
    # (frame (0), filename (1), line_number (2), function_name (3), lines (4), index (5)) = inspect.getouterframes(inspect.currentframe())[1]
    if config.SETTING.TRACE_ENABLE:
        val = inspect.getouterframes(inspect.currentframe())[1]
        log_base(val, TRACE, msg)


def debug(*msg):
    if config.SETTING.DEBUG_ENABLE:
        val = inspect.getouterframes(inspect.currentframe())[1]
        log_base(val, logging.DEBUG, msg)


def info(*msg):
    if config.SETTING.LOG_ENABLE:
        val = inspect.getouterframes(inspect.currentframe())[1]
        log_base(val, logging.INFO, msg)


def warning(*msg):
    if config.SETTING.LOG_ENABLE:
        val = inspect.getouterframes(inspect.currentframe())[1]
        log_base(val, logging.WARNING, msg)


def error(*msg):
    if config.SETTING.LOG_ENABLE:
        val = inspect.getouterframes(inspect.currentframe())[1]
        log_base(val, logging.ERROR, msg)


def log(*msg):
    if config.SETTING.LOG_ENABLE:
        val = inspect.getouterframes(inspect.currentframe())[1]
        log_base(val, logging.NOTSET, msg)


def line(*msg):
    parts = [str(x) for x in msg]
    msg_string = " ".join(parts)
    LOG.logger_line.info(msg_string)


def line_csv(*msg):
    if len(msg) == 0:
        return
    parts = [str(x) for x in msg]
    msg_string = datetime.now().strftime('%d/%m/%Y %H:%M:%S.%f')[:-3] + ";" + ";".join(parts)
    LOG.logger_csv.info(msg_string)


def log_base(val, typ, *msg):
    parts = [str(x) for x in msg[0]]
    msg_string = "[" + str(val[1]) + "," + str(val[2]) + "," + str(val[3]) + "] "+" ".join(parts)
    if typ == logging.INFO:
        LOG.logger_log.info(msg_string)
    elif typ == logging.WARNING:
        LOG.logger_log.warning(msg_string)
    elif typ == logging.ERROR:
        LOG.logger_log.error(msg_string)
    elif typ == logging.DEBUG:
        LOG.logger_debug.error(msg_string)
    elif typ == TRACE:
        LOG.logger_trace.info(msg_string)
    else:
        LOG.logger_log.info(msg_string)


START_TIME = time.process_time()
COUNTER_TIME = 0
END_TIME = START_TIME


def debug_timer(*message):
    global END_TIME, START_TIME
    END_TIME = time.process_time()
    diff = END_TIME - START_TIME
    if diff >= COUNTER_TIME:
        START_TIME = time.process_time()
        date = datetime.now().strftime('%d/%m/%Y %H:%M:%S.%f')[:-3]
        print(date, "INFO:", message)
