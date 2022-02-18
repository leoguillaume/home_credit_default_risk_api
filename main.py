from fastapi import FastAPI
import shap
import pandas as pd
import numpy as np
import os, joblib

INPUT_PATH = 'inputs'

app = FastAPI()

feature_dict = joblib.load(os.path.join(INPUT_PATH, 'feature_dict.p'))
model_feature_dict = joblib.load(os.path.join(INPUT_PATH, 'model_feature_dict.p'))
model = joblib.load(os.path.join(INPUT_PATH, 'model.p'))
data = pd.read_feather(os.path.join(INPUT_PATH, 'train_sample.f'))
application = pd.read_feather(os.path.join(INPUT_PATH, 'application_sample.f'))

application = application.loc[application.SK_ID_CURR.isin(list(data.SK_ID_CURR))]
users_ids = list(data.SK_ID_CURR)
application.set_index('SK_ID_CURR', inplace=True)
data.set_index('SK_ID_CURR', inplace=True)

@app.get('/')
async def root():
    response = 'Welcome back.'
    return response


@app.get('/user/user_list')
async def get_user_id_list():
    
    response = {'user_id_list': users_ids}

    return response

@app.get('/model/user_data/{user_id}')
async def get_model_user_data(user_id:int):
    assert user_id in users_ids, 'This user ID is not available.'
    
    user_data = data.loc[user_id].fillna('null')
    user_data = dict(user_data)

    response = {'user_id': user_id, 'user_data': user_data}

    return response

@app.get('/data/feature_list')
async def get_feature_list():

    response = {'feature_list': feature_dict}

    return response

@app.get('/data/user_data/{user_id}')
async def get_user_data(user_id:int):
    assert user_id in users_ids, 'This user ID is not available.'

    user_data = application.loc[user_id].fillna('null').astype(str)
    user_data = dict(user_data)

    response = {'user_id': user_id, 'user_data': user_data}

    return response

@app.get('/data/feature_data/{feature_id}')
async def get_feature_data(feature_id:int):
    assert feature_id in feature_dict, 'This feature ID is not available.'

    negative_data = list(application.loc[application.TARGET == 0, feature_dict[feature_id]['name']].astype(object).fillna('null'))
    positive_data = list(application.loc[application.TARGET == 1, feature_dict[feature_id]['name']].astype(object).fillna('null'))

    response = {'feature_id': feature_id, 'negative_data': negative_data, 'positive_data': positive_data}
    
    return response

@app.get('/model/predict/{user_id}')
async def get_prediction(user_id:int): 
    assert user_id in users_ids, 'This user ID is not available.'

    response = await get_model_user_data(user_id)
    user_data = pd.Series(response['user_data']).to_frame().T
    user_data.replace('null', np.NaN, inplace=True)
    prediction = model.predict_proba(user_data)[0]

    response = {'user_id': user_id, 'negative_pred': prediction[0], 'positive_pred': prediction[1]}

    return response

@app.get('/model/shap_values/{user_id}')
async def get_shap_values(user_id:int):
    assert user_id in users_ids, 'This user ID is not available.'

    user_data = data.loc[user_id].fillna('null')
    user_data.index = [model_feature_dict[i] for i in user_data.index]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    explained_values = list(shap_values[1][np.arange(0, len(data))[data.index == user_id][0], :])
    expected_value = explainer.expected_value[1]

    response = {'user_id': user_id, 'explained_values': explained_values, 'expected_value': expected_value, 'user_data': dict(user_data)}

    return response
