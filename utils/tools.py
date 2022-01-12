import logging
import os

def get_logger(level=logging.INFO):
    log = logging.getLogger("PARCEL")
    log.setLevel(level)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s - THREAD: %(threadName)s - %(name)s] : %(message)s")
    log_stream_handler = logging.StreamHandler()
    log_stream_handler.setFormatter(log_formatter)
    log.addHandler(log_stream_handler)
    return log

def writetotimelog(time, printmessage, logfile):

    '''
    Write information on process duration to a timelog file

    :param time: execution time
    :param printmessage: message that should be accompanied by the time
    :param logfile: path to txt file containing timelog (if not existing, it is created)
    :return: /
    '''
    remainingtime = time

    days = int((remainingtime) / 86400)
    remainingtime -= (days * 86400)
    hours = int((remainingtime) / 3600)
    remainingtime -= (hours * 3600)
    minutes = int((remainingtime) / 60)
    remainingtime -= (minutes * 60)
    seconds = round((remainingtime) % 60, 1)

    if time < 60:
        finalprintmessage = printmessage + " " + str(seconds) + " seconds."
    elif time < 3600:
        finalprintmessage = printmessage + " " + str(minutes) + " minutes and " + str(seconds) + " seconds."
    elif time < 86400:
        finalprintmessage = printmessage + " " + str(hours) + " hours and " + str(minutes) + " minutes and " + str(
            seconds) + " seconds."
    elif time >= 86400:
        finalprintmessage = printmessage + " " + str(days) + " days, " + str(hours) + " hours and " + str(
            minutes) + " minutes and " + str(seconds) + " seconds."

    if not os.path.exists(logfile):
        mode = 'w'
    else:
        mode = 'a'
    with open(logfile, mode) as timelogger:
        timelogger.write(finalprintmessage + '\n')