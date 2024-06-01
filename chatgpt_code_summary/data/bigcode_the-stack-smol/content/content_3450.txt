import linelib
import datetime
import signal


def handler(x, y):
    pass

signal.signal(signal.SIGUSR1, handler)
signal.signal(signal.SIGALRM, handler)

while True:
    linelib.sendblock("date", {"full_text": datetime.datetime.now().strftime(
        "%Y-%m-%e %H:%M:%S"
    )})
    linelib.sendPID("date")
    linelib.waitsig(1)
