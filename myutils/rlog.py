# Nonexecable
from datetime import datetime

def _timedprint(mark, s):
    print(datetime.now().strftime("%H:%M:%S {}> {}".format(mark, s)))

def log(s):
    _timedprint('LOG', s)

def warn(s):
    _timedprint('WARN', s)

def err(s):
    _timedprint('ERR', s)
