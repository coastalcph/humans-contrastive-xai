import numpy as np
from os import listdir, makedirs
from os.path import isdir


AUTH_TOKEN = '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'

def set_up_dir(path):
    if not isdir(path):
        makedirs(path)
       
        
        
