from flask import Flask, jsonify, make_response, request, abort
import pandas as pd
# import catboost
import pickle
from flask_cors import CORS,cross_origin

#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
import xgboost as xgb
warnings.filterwarnings(action='ignore')





model = pickle.load(open( "finalized_model_final.sav", "rb"))

import pickle
filename = 'finalized_model_last.sav'
pickle.dump(model, open(filename, 'wb'))



app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)
@app.errorhandler(404)

def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route("/")
def hello():
  return "Hello World!"

@app.route("/get_prediction", methods=['POST','OPTIONS'])
@cross_origin()
def get_prediction():
    if not request.json:
        abort(400)
    df = pd.DataFrame(request.json, index=[0])
    cols=["item code","year", "month",  "dayofyear",    "dayofweek",    "monthly_avg"]
    df = df[cols]
    matrix_test = xgb.DMatrix(df)
    sales = model.predict(matrix_test)
    val = [round(int(value)) for value in sales]

    # print(val[0])
    return jsonify({'Sales': round(val[0])})

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5012,debug = True)
