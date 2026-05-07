#-----------------------libraries---------------------------------
import os
import time
#-----------------------files-------------------------------------
import mini_programm as retrieval
#-----------------------folders-----------------------------------
os.makedirs("data/raw/gemini", exist_ok = True)
os.makedirs("data/processed", exist_ok = True)
os.makedirs("data/processed/keywords",exist_ok=True)
#-----------------------running mini_programm---------------------
def run_retrieval(intervention): 
    retrieval.data_retrieval(intervention)
    retrieval.validate_data()
    print("waiting 30 seconds") 
    time.sleep(30)
    retrieval.classify_papers()
    time.sleep(60)
    print("waiting 60")
    retrieval.assess_qualities()
    print("waiting 60 seconds")
    time.sleep(60)
    retrieval.add_relations() 
    print("sucess")
run_retrieval("rapamycin longevity")
#-----------------------runing confidence score-------------------
#import confidence_score
#confidence_score.sigmoid_activ()
#-----------------------social media aspect-----------------------
#-----------------------creating report---------------------------
