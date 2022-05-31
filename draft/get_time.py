# Copyright (C) 2021 Andrii Sonsiadlo

from datetime import datetime


def getTime(unit: str):
    time = datetime.now()

    timeSecond = time.second
    timeMinute = time.minute
    timeHour = time.hour
    timeDay = time.day
    timeMonth = time.month
    timeYear = time.year

    if (unit == "second"):
        if (timeSecond < 10):
            timeSecond = "0" + str(timeSecond)
        else:
            timeSecond = str(timeSecond)
        return timeSecond

    if (unit == "minute"):
        if (timeMinute < 10):
            timeMinute = "0" + str(timeMinute)
        else:
            timeMinute = str(timeMinute)
        return timeMinute

    if (unit == "hour"):
        if (timeHour < 10):
            timeHour = "0" + str(timeHour)
        else:
            timeHour = str(timeHour)
        return timeHour

    if (unit == "day"):
        if (timeDay < 10):
            timeDay = "0" + str(timeDay)
        else:
            timeDay = str(timeDay)
        return timeDay

    if (unit == "month"):
        if (timeMonth < 10):
            timeMonth = "0" + str(timeMonth)
        else:
            timeMonth = str(timeMonth)
        return timeMonth

    if (unit == "year"):
        timeYear = str(timeYear)
        return timeYear


def get_date():
    return str(getTime("day") + '.' + getTime("month") + '.' + getTime("year"))


def get_time():
    return str(getTime("hour") + ':' + getTime("minute") + ':' + getTime("second"))
