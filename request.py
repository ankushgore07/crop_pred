# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:37:13 2021

@author: Ankush
"""


import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'N':90,'P':42,'K':43,'temperature':20.87974371,'humidity':82.00274423,'ph':6.502985292000001,'rainfall':202.9355362})

print(r.json())