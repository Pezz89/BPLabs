import os
import re
import errno
import fnmatch
from natsort import natsorted
from collections import namedtuple
def prepareOutDir(folder):
    '''
    Check that the specified output directory exists and remove any
    pre-existing files and folders from it
    '''
    try:
        os.mkdir(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def globDir(directory, pattern):
    '''
    Return all files in a directory matching the unix glob pattern, ignoring
    case
    '''
    def absoluteFilePaths(directory):
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))
    speech_file_pattern = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    filepaths = []
    for item in absoluteFilePaths(directory):
        if bool(speech_file_pattern.match(os.path.basename(item))):
            filepaths.append(item)
    filepaths = natsorted(filepaths)
    return filepaths

def organiseWavs(wavFiles):
    '''
    Returns a dictionary of FileTuple('filename', 'filepath') objects.
    Each dictionary key represents a column from a-e
    '''
    wavFiles = natsorted(wavFiles)
    # create dictionary of {wav filename: wav filepath} key, value pairs
    fileTuple = namedtuple('FileTuple', ['filename', 'filepath'])
    wavFileTuples = [fileTuple._make([os.path.basename(os.path.splitext(a)[0]), a]) for a in wavFiles]
    # Separate gap files
    gaps = wavFileTuples[:3]
    wavFileTuples = wavFileTuples[3:]
    # Sort files into columns
    columnNames = ['a', 'b', 'c', 'd', 'e']
    i = 0
    wavFileMatrix = {}
    for col in columnNames:
        wavFileMatrix[col] = wavFileTuples[i:i+100]
        i += 100
    return wavFileMatrix
