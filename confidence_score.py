#creating a confidence score using the json file delivered via the search
#---------------------------------------Libraries--------------------------
import numpy as np #using the e-function in the sigmoid activation function
import json
#---------------------------------------input------------------------------
with open('data/processed/classified_papers.json','r', encoding = 'utf-8') as file: 
    data = json.load(file)
#---------------------------------------Sigmoid activation function--------
def sigmoid_activ(data): 
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
    for paper in data['papers']: 
        w = weights.get(paper['study_type'])
        r = result.get(paper['study_result'])
    sum += (w*r) 
    score = 1/(1+np.exp(-1*sum))
    data['confidence_score'] = score
    with open('data/processed/classified_papers.json','w', encoding = 'utf-8') as file: 
        json.dump(data,file,indent=4)
sigmoid_activ(data)