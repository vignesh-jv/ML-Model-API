from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS, cross_origin

import pickle
import numpy as np
from model import NLPModel

app = Flask(__name__)
api = Api(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = NLPModel()

clf_path = 'models/logit.pkl'
with open(clf_path, 'rb') as f:
    model.logit = pickle.load(f)

vec_path = 'models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictSarcasm(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        if user_query is not None:

            # vectorize the user's query and make a prediction
            uq_vectorized = model.vectorizer_transform(np.array([user_query]))
            #print(uq_vectorized)
            prediction = model.predict(uq_vectorized)
            pred_proba = model.predict_proba(uq_vectorized)

            # Output either 'Not Sarcastic' or 'Sarcastic' along with the score
            if prediction == 0:
                pred_text = 'Not sarcastic'
            else:
                pred_text = 'Sarcastic'

            # round the predict proba value and set to new variable
            confidence = round(pred_proba[0], 3)

            # create JSON object
            output = {'prediction': pred_text, 'confidence': confidence}

            return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSarcasm, '/')


if __name__ == '__main__':
    app.run(debug=True)
