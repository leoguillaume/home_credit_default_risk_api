from fastapi import FastAPI
import pandas as pd
import numpy as np
import os, joblib

INPUT_PATH = 'inputs'

DATA_NAME = 'data_sample.f'
FEATURE_DICT_NAME = 'feature_dict.p'
MODEL_NAME = 'model.p'

app = FastAPI()

feature_dict = joblib.load(os.path.join(INPUT_PATH, FEATURE_DICT_NAME))
model = joblib.load(os.path.join(INPUT_PATH, MODEL_NAME))

feature_ids = [int(f.split('_')[1]) for f in feature_dict]

data = pd.read_feather(os.path.join(INPUT_PATH, DATA_NAME))
data.set_index('SK_ID_CURR', inplace=True)
users_ids = list(data.index)

X = data.drop(columns='TARGET')
y = data.TARGET

@app.get('/')
async def root():
    response = {
        "Available endpoints": [
            "/user/user_list", 
            "/user/user_data/{user_id}",
            '/user/user_data/{user_id}/{feature_id}',
            '/model/feature_list',
            '/model/feature_name/{feature_id}',
            '/model/predict/{user_id}'
            ]
            }
    return response


@app.get('/user/user_list')
async def get_user_id_list():
    
    response = {'user_id_list': users_ids}

    return response

@app.get('/user/user_data/{user_id}')
async def get_user_data(user_id:int):
    assert user_id in users_ids, 'This user ID is not available.'

    user_data = X.loc[user_id].fillna('null')
    user_data = dict(user_data)

    response = {'user_id': user_id, 'user_data': user_data}

    return response

@app.get('/user/user_data/{user_id}/{feature_id}')
async def get_user_feature_value(user_id:int, feature_id:str):
    assert user_id in users_ids, 'This user ID is not available.'
    assert feature_id in feature_ids, 'This feature ID is not available.'

    response = await  get_user_data(user_id)
    feature_value = response['user_data']['feature_' + str(feature_id)]

    response = {'user_id': user_id, 'feature_id': feature_id, 'feature_value': feature_value}

    return response

@app.get('/model/feature_list')
async def get_feature_id_list():

    response = {'feature_id_list': feature_ids}

    return response

@app.get('/model/feature_name/{feature_id}')
async def get_feature_name(feature_id:int):

    response = {'feature_id': feature_id, 'feature_name': feature_dict['feature_' + str(feature_id)]}

    return response

@app.get('/model/feature_data/{feature_id}')
async def get_feature_data(feature_id:int):
    assert feature_id in feature_ids, 'This feature ID is not available.'

    negative_data = list(X.loc[X.index.isin(y[y == 0].index), 'feature_' + str(feature_id)].fillna('null'))
    positive_data = list(X.loc[X.index.isin(y[y == 1].index), 'feature_' + str(feature_id)].fillna('null'))

    response = {'feature_id': feature_id, 'negative_data': negative_data, 'positive_data': positive_data}
    
    return response

@app.get('/model/predict/{user_id}')
async def get_prediction(user_id:int): 
    assert user_id in users_ids, 'This user ID is not available.'

    response = await get_user_data(user_id)
    user_data = pd.Series(response['user_data']).to_frame().T
    user_data.index = [response['user_id']]
    user_data.replace('null', np.NaN, inplace=True)

    prediction = model.predict_proba(user_data)[0]

    response = {'user_id': user_id, 'negative_pred': prediction[0], 'positive_pred': prediction[1]}
    return response


