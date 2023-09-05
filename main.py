#!/usr/bin/env python
# coding: utf-8
from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request, jsonify


import pandas as pd
from tensorflow.keras.models import load_model
import pandas as pd
import json


# sess =tf.Session()
# graph = tf.get_default_graph()

# with sess.as_default():
#     with graph.as_default():
#load the trained model
model2 = load_model(f"./Models/modelANN2.h5")

app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



# this is our categorical columns
categorical_columns = ['country', 'sub_continent', 'metro', 'category', 'mobile_brand_name', 'operating_system_version', 'medium','source']

# this is the encoded labels 
countryEncode = ['Other', "'Indonesia'", "'Pakistan'", "'Philippines'", "'Egypt'", "'Brazil'", "'India'", "'United States'", "'Vietnam'", "'South Africa'", "'Russia'", "'Myanmar (Burma)'", "'Nigeria'", "'Cambodia'", "'Iraq'"]
sub_continentEncode = ["'Southeast Asia'", "'Southern Asia'", "'South America'", "'Northern Africa'", 'Other', "'Northern America'", "'Western Asia'", "'Eastern Europe'", "'Western Africa'", "'Southern Africa'"]
metroEncode = ["'(not set)'", 'Other']
mobile_brand_nameEncode = ["'Samsung'", 'Other', "'OPPO'", "'Xiaomi'", "'Vivo'", "'Realme'", "'Infinix'", "'Tecno'"]
operating_system_versionEncode = ["'Android 11'", "'Android 12'", "'Android 10'", "'Android 13'", "'Android 9'", "'Android 8.1.0'", 'Other', "'Android 6.0.1'", "'Android 8.0.0'", "'Android 7.1.1'"]
mediumEncode = ["'organic'", "'(none)'", "'cpc'"]
sourceEncode =["'google-play'", "'(direct)'", "'google'"]
categoryEncode = ["'mobile'", "'tablet'"]
# Combine them into a dictionary
encoded_dict = {
    'country': countryEncode,
    'sub_continent': sub_continentEncode,
    'category' : categoryEncode,
    'metro': metroEncode,
    'mobile_brand_name': mobile_brand_nameEncode,
    'operating_system_version': operating_system_versionEncode,
    'medium': mediumEncode,
    'source': sourceEncode
}

# Manually drop columns
def drop_columns(df, columns):
    return df.drop(columns=columns)

# Manually filter categorical values
def filter_categorical(df, encoded_dict):
    for col, valid_values in encoded_dict.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if x in valid_values else 'Other')
    return df

# Manually perform one-hot encoding
def one_hot_encode(df, columns, encoded_dict):
    for col in columns:
        if col in df.columns:
            for unique_val in encoded_dict[col]:
                df[f"{col}_{unique_val}"] = (df[col] == unique_val).astype(int)
            df.drop(columns=[col], inplace=True)
    return df

def ModelPredict(df):
    return model2.predict(df)

# now test the function, we weill move this function to fastAPI file
def json_to_df(json_sample):
    # Parse the JSON string to remove escape characters
    parsed_json = json.loads(json_sample)
    # Convert JSON to DataFrame
    df = pd.DataFrame([parsed_json])
    return df
# Make steps in the pipeline which:
# Convert the value in column to 'Others' if it value no in the corresponding encode list
# use One Hot Encoded which provided encoded list, which the value Other
def Predict(item):
    df = pd.DataFrame([list(item.values())], columns=list(item.keys()))
    # Filter categorical values
    df = filter_categorical(df, encoded_dict)
    
    # One-hot encode
    categorical_columns = ['country', 'sub_continent', 'metro', 'category', 'mobile_brand_name', 'operating_system_version', 'medium','source']
    df = one_hot_encode(df, categorical_columns, encoded_dict)
    
    predictions = ModelPredict(df)
    
    print(predictions)
    # Find the class index with the maximum score for each sample
    if predictions >= 0.5:
        class_indices = 1
    else:
        class_indices = 0
    print(class_indices)
    return predictions,class_indices



@app.route('/', methods=['POST'] )
@cross_origin(origins='*')
def a():
    data = request.json
    try:
        predictions,class_indices = Predict(data)
        return {"predictions":float(predictions), "label": int(class_indices)}
    except Exception as ex:
        print(ex)
        return {"err":f"Error: {ex}"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')