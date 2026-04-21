from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US', tz=360) #connecting to google servers via api
kw_list = ['Ginger']
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='',gprop='')
print(pytrends.interest_over_time()) 
