#creating a confidence score using the json file delivered via the search
#---------------------------------------Libraries--------------------------
import numpy as np #using the e-function in the sigmoid activation function
import json
#---------------------------------------input------------------------------

#---------------------------------------Sigmoid activation function--------
def sigmoid_activ(): 
    print("calculating confidence score")
    with open('data/processed/classified_papers.json','r', encoding = 'utf-8') as file: 
        data = json.load(file)
    weights = {
    "Systematic review & meta-analaysis":1.0,
    "Randomised controlled trials (RCTs)":5/6,
    "Observational / epidmiological studies": 4/6, 
    "Animal model studies (in vivo)": 3/6,
    "Cell culture / in vitro studies": 2/6,
    "In silico / computational predictions": 1/6
    }
    effect_type = {
    "lifespan": 1.0, 
    "healthspan": 5/6, 
    "functional" : 4/6,
    "biomarker" : 3/6, 
    "mechanistic": 2/6, 
    "computational": 1/6, 
    "unclear" : 0
    }
    result = {"positive":1.0,"negative":-1.0,"neutral":-0.5,"unclear":0.05}
    sum = 0
    n = 0
    for paper in data['papers']: 
        w = weights.get(paper['study_type'])  
        r = result.get(paper['study_result'])
        e = effect_type.get(paper['effect_type'])
        #calculation the score
        if (w== None) | (r == None) | (e==None):
            continue
        sum += (w*r*e)
        n += w
    #normalize score 
    sum = (sum)/(n**0.5)
    score = 1/(1+np.exp(-1*sum))
    data['confidence_score'] = score
    with open('data/processed/classified_papers.json','w', encoding = 'utf-8') as file: 
        json.dump(data,file,indent=4)
    print(score)

sigmoid_activ()
