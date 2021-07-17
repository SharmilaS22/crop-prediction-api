from flask import Flask, request
app = Flask(__name__)

import pandas as pd

from joblib import load
model = load('./model/model.joblib')

# temperature = {
#     '10 to 15': 0,
#     '15 to 20': 1,
#     '20 to 25': 2,
#     '25 to 30': 3,
#     '30 to 35': 4,
#     '35 to 40': 5
# }
## Annual Rainfall cms
#     '25 to 50': 0,
#     '50 to 75': 1,
#     '75 to 100': 2,
#     '100 to 150': 3,
#     '150 to 200': 4,
#     '200 to 250': 5,
#     'more than 250': 6
## Ph
#     '4.5 to 5.0': 0,
#     '5.0 to 5.5': 1,
#     '5.5 to 6.0': 2,
#     '6.0 to 6.5': 3,
#     '6.5 to 7.0': 4,
#     '7.0 to 7.5': 5



soil_type_dummies = {
    'soil_alkaline': [0],
    'soil_alluvial': [0],
    'soil_clay': [0],
    'soil_brown': [0],
    'soil_black': [0],
    'soil_clayey': [0],
    'soil_gravel': [0],
    'soil_hill slope': [0],
    'soil_laterite': [0],
    'soil_loam': [0],
    'soil_loamy': [1], #
    'soil_red': [0],
    'soil_silt': [0],
    'soil_valconic': [0],    
}

cluster_dict = {
    0: ['rice', ' wheat', ' sugarcane', 'coffee', 'cotton'],
    1: ['rice', ' sugarcane', 'coffee']
}

@app.get('/')
def root_route():
    return 'Hello World'


# /predict?temperature_c=2&annual_rainfall_cms=2&ph=3&soil_loamy=1
@app.get('/predict')
def predict():

    predict_data = dict(request.args)
    for k, v in predict_data.items():
        predict_data[k] = [int(v[0])]

    user_input = pd.DataFrame(data={**soil_type_dummies, **predict_data})
    
    prediction = model.predict(user_input)[0]

    # TODO add it to database

    response_data = {
        'suggested_crops': cluster_dict[prediction]
    }

    return response_data

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000)