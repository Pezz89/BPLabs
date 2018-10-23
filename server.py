#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from flask import Flask, url_for, render_template, jsonify, request, make_response, g
from flask_socketio import emit

import webview
import webbrowser
import app
import time
from threading import Thread, Event

import config

server = config.server
socketio = config.socketio

@server.after_request
def add_header(response):
    # Disable caching? unsure why...
    response.headers['Cache-Control'] = 'no-store'
    return response


@server.route("/")
def landing():
    """
    Render index.html
    """
    return render_template("index.html")


@server.route("/choose/path")
def choose_path():
    """
    Invokes a folder selection dialog
    """
    dirs = webview.create_file_dialog(webview.FOLDER_DIALOG)
    if dirs and len(dirs) > 0:
        directory = dirs[0]
        if isinstance(directory, bytes):
            directory = directory.decode("utf-8")

        response = {"status": "ok", "directory": directory}
    else:
        response = {"status": "cancel"}

    return jsonify(response)

@server.route("/fullscreen")
def fullscreen():
    webview.toggle_fullscreen()
    return jsonify({})

@server.route('/home')
def homepage():
    title = "Welcome"
    paragraph = [
        "This web app was developed for the generation of stimulus and "
        "running of experiments for the PhD project \"Predicting speech "
        "in noise performance using evoked responses\".",
        "Use the drop down menus to access the various modules of this app."
    ]

    try:
        return render_template("home.html", title = title, paragraph=paragraph)
    except Exception as e:
        return str(e)


@server.route('/mat_dec_stim')
def matDecStim():
    return render_template("matrix_decode_stim.html")

thread = Thread()
thread_stop_event = Event()

class StimGenThread(Thread):
    '''
    Thread object for asynchronous processing of data in Python without locking
    up the GUI
    '''
    def __init__(self):
        super(StimGenThread, self).__init__()


    def process_stimulus(self):
        '''
        An example process
        '''
        for participant_n in range(15):
            percent = ((participant_n+1) / 15)*100.
            # Emit a message to update the progress bar during execution of the
            # python process (see relevant javascript code in
            # matrix_decode_stim.html)
            socketio.emit('update-progress', {'data': '{}%'.format(percent)}, namespace='/main')
            time.sleep(1)


    def run(self):
        '''
        This function is called when the thread starts
        '''
        self.process_stimulus()
        socketio.emit('processing-complete', {'data': ''}, namespace='/main')


@socketio.on('run_mat_stim_gen', namespace='/main')
def generateStim(msg):
    '''
    When process buton is clicked in GUI, start an asynchronous thread to run
    process
    '''
    thread = StimGenThread()
    thread.start()

@socketio.on('open_save_dialog', namespace='/main')
def openSaveDialog():
    # Open a file dialog interface for selecting a directory
    dirs = webview.create_file_dialog(webview.FOLDER_DIALOG)
    if dirs and len(dirs) > 0:
        directory = dirs[0]
        if isinstance(directory, bytes):
            directory = directory.decode("utf-8")
    # TODO: Add filepath checking here...
    # Send message with selected directory to the GUI
    socketio.emit('save-dialog-resp', {'data': directory}, namespace='/main')

@socketio.on('open_mat_dialog', namespace='/main')
def openMatDialog():
    # Open a file dialog interface for selecting a directory
    dirs = webview.create_file_dialog(webview.FOLDER_DIALOG)
    if dirs and len(dirs) > 0:
        directory = dirs[0]
        if isinstance(directory, bytes):
            directory = directory.decode("utf-8")
    # TODO: Add filepath checking here...
    # Send message with selected directory to the GUI
    socketio.emit('mat-dialog-resp', {'data': directory}, namespace='/main')

@server.route('/click_stim')
def clickStim():
    return render_template("click_stim.html")

@server.route('/da_stim')
def daStim():
    return render_template("da_stim.html")

def run_server():
    '''
    Start the Flask server
    '''
    # SocketIO objects are defined in config.py
    socketio.init_app(server)
    socketio.run(server, host="127.0.0.1", port=23948)

if __name__ == "__main__":
    run_server()

