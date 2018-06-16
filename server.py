import pickle
import flask
from flask import request

app = flask.Flask(__name__)

#loading my model
model = pickle.load(open("model.pkl","r"))

#defining a route for only post requests
@app.route('/', methods=['POST'])
def index():
    #getting an array of features from the post request's body
    feature_array = request.get_json()['feature_array']

    #creating a response object
    #storing the model's prediction in the object
    response = {}
    response['predictions'] = model.predict([feature_array]).tolist()

    #returning the response object as json
    return flask.jsonify(response)