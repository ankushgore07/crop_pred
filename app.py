# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 05:40:58 2021

@author: Ankush
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction

    return render_template('index.html', prediction_text='The Crop that gives u Best Result in your soil/farm is  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True) 