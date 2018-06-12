import pickle
import flask
from flask import request

app = flask.Flask(__name__)

#loading my model
model = pickle.load(open("model.pkl","r"))

@app.route('/', methods=['POST'])
def index():
    response = {}
    feature_array = request.get_json()['feature_array']
    response['predictions'] = model.predict([feature_array]).tolist()
    return flask.jsonify(response)