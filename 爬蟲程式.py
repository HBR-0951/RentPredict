#%%
import pandas as pd
import requests as req
import json    
import time
import random 
#%%
headers={
		'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER'
		}    
#%%
def web_scraping_one_page(url):
    try:
        ret_data = []
        resp = req.get(url, headers=headers)
        resp.raise_for_status()
        data = json.loads(resp.text)
        for w in data['webRentCaseGroupingList']:
            data_dict = {'addr': w['simpAddress'],
                       'district': w['district'],
                       'road': w['road'],
                       'description': w['caseName'],
                       'area': w['buildPin'],
                       'room': w['rm'],
                       'bath': w['bathRm'],
                       'dinning': w['livingRm'],
                       'obj_type': w['rentPurPoseName'],
                       'which_floor': w['fromFloor'],
                       'total_floor': w['upFloor'],
                       'mgnt_fee': w['managementFee'],
                       'rent_fee': w['rentPrice']
                       }
            ret_data.append(data_dict.copy()) 
        scraping_success = True
    except Exception as err:
        print(err)
        pass
        scraping_success = False

    return scraping_success, ret_data

#%%
lst_rent_fee = []
    
print('Requesting data from web...')
print('---> Requesting data in the page 1...', end = '', flush = True)


url='https://rent.houseprice.tw/ws/list/%E5%8F%B0%E4%B8%AD%E5%B8%82_city/%E8%A5%BF%E5%B1%AF%E5%8D%80-%E5%8D%97%E5%B1%AF%E5%8D%80-%E5%8C%97%E5%B1%AF%E5%8D%80-%E5%8C%97%E5%8D%80-%E8%A5%BF%E5%8D%80-%E4%B8%AD%E5%8D%80-%E6%9D%B1%E5%8D%80-%E5%8D%97%E5%8D%80_zip/'

OK, ret_data = web_scraping_one_page(url)  
if OK:
   lst_rent_fee += ret_data 
else:
   print('Error to get data in the 1st page')    

cnt = 1

total_cnt = 702

for p in range(2,total_cnt):  
    print('\r---> Requesting data in the page {0}...({1:.1f}%)'.format(p, cnt/(total_cnt-1)*100), end = '', flush = True)
    url_n = '{0}?p={1}'.format(url, p)
    OK, ret_data = web_scraping_one_page(url_n)
    if OK:
        lst_rent_fee += ret_data
    else:
       print('Error to get data in page {0}'.format(p))    
    time.sleep(random.randint(1, 3))
    cnt += 1
    
print('\nComplete! Total {0} records scrapped.'.format(len(lst_rent_fee))) 
#%%    
cols=['addr','district','road','description','area','room','bath','dinning','obj_type','which_floor','total_floor','mgnt_fee','rent_fee']
df = pd.DataFrame (lst_rent_fee, columns = cols)

df.to_csv('rent_price_raw.csv',index=False) 
print('File saved !!')

