#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import webview
from time import sleep
from server import run_server
import config
from threading import Thread, Lock
from loggerops import create_logger

socketio = config.socketio

server_lock = Lock()

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def url_ok(url, port):
    # Use httplib on Python 2
    try:
        from http.client import HTTPConnection
    except ImportError:
        from httplib import HTTPConnection

    try:
        conn = HTTPConnection(url, port)
        conn.request("GET", "/")
        r = conn.getresponse()
        return r.status == 200
    except:
        logger.error("Server not started")
        return False

def create_new_window():
    webview.create_window("Participant view",
                          "http://127.0.0.1:23948/participant_home",
                          width=1000, height=700, min_size=(1000, 500))


if __name__ == '__main__':
    logger = create_logger(
        logger_streamlevel=10,
        log_filename='./logs/main.log',
        logger_filelevel=10
    )
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log = logging.getLogger('engineio')
    log.setLevel(logging.ERROR)
    logger.debug("Starting server")
    # Run server in seperate thread
    # socketio.start_background_task(run_server)
    t = Thread(target=run_server)
    t.daemon = True
    t.start()
    logger.debug("Checking server")

    # Check server is up and running
    while not url_ok("127.0.0.1", 23948):
        sleep(0.1)

    logger.debug("Server started")
    # Create browser window for user interaction with GUI

    # Create clinician view
    t2 = Thread(target=create_new_window)
    t2.start()
    # socketio.start_background_task(create_new_window)
    webview.create_window("BPLabs",
                          "http://127.0.0.1:23948/home",
                          width=1000, height=700, min_size=(1000, 500))
    webview.start(debug=True)


