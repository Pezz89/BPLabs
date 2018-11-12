from flask import Flask
from flask_socketio import SocketIO, emit
import os

gui_dir = os.path.join(os.getcwd(), "gui")  # development path
if not os.path.exists(gui_dir):  # frozen executable path
    gui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui")

server = Flask(__name__)
server.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1  # disable caching
socketio = SocketIO(async_mode='threading')
socketio.init_app(server)
