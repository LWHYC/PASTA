import time
import os
import shutil
import logging

def ignore_these_dirs(dirname, filenames):
    ignore_dirs = ['.git', '__pycache__', 'weights', 'GeodisTK-master', 'runs']
    return [dir for dir in ignore_dirs if dir in filenames]

class Logger(object):
    def __init__(self, log_path=None):
        import sys
        if not log_path:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_path = os.path.join(
                'runs', current_time + '_' + socket.gethostname()+'/default.log')
        self.terminal = sys.stdout
        self.log = open(log_path+'/default.log', "w", buffering=64, encoding="utf-8")
        if os.path.exists(log_path + '/code'):
            shutil.rmtree(log_path + '/code')
        # shutil.copytree('.', log_path + '/code', ignore=ignore_these_dirs)
 
    def print(self, *message):
        message = ",".join([str(it) for it in message])
        self.terminal.write(str(message) + "\n")
        self.log.write(str(message) + "\n")
 
    def flush(self):
        self.terminal.flush()
        self.log.flush()
 
    def close(self):
        self.log.close()