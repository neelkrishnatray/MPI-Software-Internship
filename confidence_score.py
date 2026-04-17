#creating a confidence score using the json file delivered via the search
#---------------------------------------Libraries--------------------------
import numpy as np #using the e-function in the sigmoid activation function
import os
#---------------------------------------input------------------------------
#---------------------------------------Sigmoid activation function--------
def sigmoid_activ(papers,k=1.0): 
    weights = {
    "Systematic review & meta-analaysis":1.0,
    "Randomised controlled trials (RCTs)":5/6,
    "Observational / epidmiological studies": 4/6, 
    "Animal model studies (in vivo)": 3/6,
    "Cell culture / in vitro studies": 2/6,
    "In silico / computational predictions": 1/6
    }
    result = {"positive":1,"negative":-1,"neutral":-0.5,"unclear":0.05}
    sum = 0
    for paper in papers: 
        w = weights.get(paper['study_type'])
        r = result.get(paper['result'])
    sum += (w*r) 
    score = 1/(1+np.exp(-k*sum))
    return score 


