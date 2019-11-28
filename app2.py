__author__ = 'yangbin1729'

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello!"