from flask import Flask, jsonify, abort, request, make_response, url_for,redirect, render_template
#from flask.ext.httpauth import HTTPBasicAuth
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path = "")

auth = HTTPBasicAuth()

@app.route("/")
def main():
    
    return render_template("main.html")

if __name__ == '__main__':
    app.run(debug = True, host= '127.0.0.1')
