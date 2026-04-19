#-----------------------libraries---------------------------------
import os
#-----------------------files-------------------------------------
import mini_programm
#-----------------------folders-----------------------------------
os.makedirs("data/raw/gemini", exist_ok = True)
os.makedirs("data/processed", exist_ok = True)
os.makedirs("data/processed/keywords",exist_ok=True)
#-----------------------running mini_programm---------------------
#ageing_intervention = input("what ageing intervention would you like to research? ")
#mini_programm.main(ageing_intervention)
#-----------------------runing confidence score-------------------
#import confidence_score
#confidence_score.sigmoid_activ()
#-----------------------social media aspect-----------------------
#-----------------------creating report---------------------------
