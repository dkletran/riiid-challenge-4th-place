import sys
class CaptureStdoutToFile(object):
    def __init__(self, logfile):
        self.file = open(logfile, 'w')
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, _type, _value, _traceback):
        sys.stdout = self.stdout
        self.file.close()