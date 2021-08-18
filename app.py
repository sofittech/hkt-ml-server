from flask import Flask, jsonify, make_response, request, abort
import pandas as pd
# import catboost
import pickle
from flask_cors import CORS,cross_origin

#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
import warnings
import xgboost as xgb
from datetime import timedelta, date
# import datetime
from datetime import datetime


warnings.filterwarnings(action='ignore')
model = pickle.load(open( "finalized_model_final.sav", "rb"))

# filename = 'finalized_model_last.sav'
# pickle.dump(model, open(filename, 'wb'))
date_list = [] 
def daterange(date1, date2):
    
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def datesList(start_dt, end_dt):
    for dt in daterange(start_dt, end_dt):
        date_list.append(dt.strftime("%Y-%m-%d"))
    return date_list 
        # print(dt.strftime("%Y-%m-%d"))



app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)
@app.errorhandler(404)

def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

# def assembleData():
#     start_dt = date(2015, 12, 20)
#     end_dt = date(2016, 1, 11)


@app.route("/")
def hello():
  return "Hello World!"

@app.route("/get_prediction", methods=['POST','OPTIONS'])
@cross_origin()
def get_prediction():
    if not request.json:
        abort(400)
    payload  = request.json
    # print(payload,"-----------------")

    Sku_ids = payload['item code']
    dates = payload['dates']
    monthly_avgs = payload['monthly_avg']
    # print(dates[0].strip('"'),"---------------------------------------")
    # print(Sku_ids)
    # print(monthly_avgs)


    start_dt = start_dt = dates[0].strip('"')

    start_dt = datetime.strptime(start_dt, '%Y-%m-%d')
    # start_dt = start_dt.replace("-", ",")
    # print(start_dt,"--------------")
    # start_dt = datetime.date(int(start_dt))

    end_dt = dates[1].strip('"')
    end_dt = datetime.strptime(end_dt, '%Y-%m-%d')
    # end_dt = end_dt.replace("-", ",")
    # end_dt = datetime.date(int(end_dt))


    # end_dt = datetime.date(dates[1].strip('"'))
    date_list  = datesList(start_dt, end_dt)
    # print(date_list)

    salesData = []
    # sale = {}

    # jsonData = assembleData(request.json)
    # df = pd.DataFrame(jsonData, index=[0])
    for s_id, avg in zip(Sku_ids, monthly_avgs):

        sale = []

        for date in date_list:
            newdate = datetime.strptime(date, '%Y-%m-%d')
            year = int(newdate.strftime("%Y"))
            month = int(newdate.strftime("%m"))
            dayofyear = int(newdate.strftime("%j"))
            dayofweek = int(newdate.weekday())

            # print((s_id))
            # print((avg))
            # print((year))
            # print((month))
            # print((dayofweek))

            jsonData = {'item code':s_id,'year': year, 'month':month,  'dayofyear':dayofyear, 'dayofweek':dayofweek,  'monthly_avg':avg} 
            df = pd.DataFrame(jsonData, index=[0])

            cols=["item code","year", "month",  "dayofyear",  "dayofweek",   "monthly_avg"]
            df = df[cols]
            matrix_test = xgb.DMatrix(df)
            sales = model.predict(matrix_test)
            val = [round(int(value)) for value in sales]

            # sale[date] = round(val[0])
            sale.append({"date": date, "sale":round(val[0])})

        salesData.append({"sku": s_id, "sales": sale})
            # salesData["sku"] = s_id
            # salesData[s_id] = sale


        
#     print(salesData)




    # print(val[0])
    return jsonify(salesData)

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5012,debug = True)
