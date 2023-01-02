#%%
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
#%%
df  = pd.read_csv('rent_price_raw.csv')
df['obj_type'].value_counts()
#%%---分類資料
df_rm = df[(df.obj_type == '獨立套房') + (df.obj_type == '整層住家') + (df.obj_type == '住宅') + (df.obj_type == '分租套房')]
df_sf = df[(df.obj_type == '店面') + (df.obj_type == '辦公')]
#%%---處理缺失值
df_rm['room'] = df_rm['room'].fillna(1)
df_rm['bath'] = df_rm['bath'].fillna(1)
df_rm['dinning'] = df_rm['dinning'].fillna(0)
df_rm['mgnt_fee'] = df_rm['mgnt_fee'].fillna(0)
df_rm = df_rm.dropna() 
df_rm = df_rm.drop_duplicates()
df_rm.reset_index(inplace=True, drop=True)

df_sf['room'] = df_sf['room'].fillna(0)
df_sf['bath'] = df_sf['bath'].fillna(0)
df_sf['dinning'] = df_sf['dinning'].fillna(0)
df_sf['mgnt_fee'] = df_sf['mgnt_fee'].fillna(0)
df_sf = df_sf.dropna()
df_sf = df_sf.drop_duplicates()
df_sf.reset_index(inplace=True, drop=True)
#%%---處理雜項-----管理費
l = ['元/月','/月','$','月繳 ',' 元','約',',']
for i in l:
    df_rm['mgnt_fee'] = df_rm.mgnt_fee.apply(lambda x: str(x).replace(i,''))
l2 = ['0元(已含租金內)','0元(不含租金內)','無','租金內含','租金已含','租金含管','含','內含','內含在租金','已含租金內','已0租金內','內0','內0在租金']
for j in l2:
    df_rm['mgnt_fee'] = df_rm.mgnt_fee.apply(lambda x: str(x).replace(j,'0'))

z1, z2 = [], []
for z in range(0,len(df_rm.index)):
    y = df_rm.mgnt_fee[z].isdigit()
    if y == False:
        z1.append(z)
df_rm = df_rm.drop(index = z1) 

df_rm['total_fee'] = df_rm.mgnt_fee.astype(float) + df_rm.rent_fee.astype(float)
df_rm['mgnt_fee'] = df_rm['mgnt_fee'].astype('float64')

#-----
l = ['元/月','/月','$','月繳 ',' 元','約',',']
for i in l:
    df_sf['mgnt_fee'] = df_sf.mgnt_fee.apply(lambda x: str(x).replace(i,''))
l2 = ['0元(已含租金內)','0元(不含租金內)','無','租金內含','租金已含','租金含管','含','內含','內含在租金','已含租金內','已0租金內','內0','內0在租金']
for j in l2:
    df_sf['mgnt_fee'] = df_sf.mgnt_fee.apply(lambda x: str(x).replace(j,'0'))

z2 = []
for z in range(0,len(df_sf.index)):
    y = df_sf.mgnt_fee[z].isdigit()
    if y == False:
        z2.append(z)
df_sf = df_sf.drop(index = z2) 

df_sf['total_fee'] = df_sf.mgnt_fee.astype(float) + df_sf.rent_fee.astype(float)
df_sf['mgnt_fee'] = df_sf['mgnt_fee'].astype('float64')
#%%---處理雜項-----樓層
l3 = ['--','B1']
for i in l3:
    df_rm['which_floor'] = df_rm.which_floor.apply(lambda x: x.replace(i,'0'))
df_rm['which_floor'] = df_rm['which_floor'].astype('int32')
#df_rm.boxplot(column = 'which_floor')
df_rm = df_rm[(df_rm.which_floor > 0) & (df_rm.which_floor < 18)]

#-----
df_sf['which_floor'] = df_sf.which_floor.apply(lambda x: x.replace('--','0'))
df_sf['which_floor'] = df_sf.which_floor.apply(lambda x: x.replace('B1','-1'))
df_sf['which_floor'] = df_sf['which_floor'].astype('int32')
#df_sf.boxplot(column = 'which_floor')
df_sf = df_sf[(df_sf.which_floor != 0) & (df_sf.which_floor < 50)]
#%%---處理離群值---其他ㄉ
dt_rm = df_rm.describe().round(1)

#df_rm.boxplot(column = 'area')
df_rm = df_rm[(df_rm.area > 4) & (df_rm.area < 35)]

#df_rm.boxplot(column = 'room')
df_rm = df_rm[(df_rm.room < 6) & (df_rm.room > 0)]

#df_rm.boxplot(column = 'bath')
df_rm = df_rm[(df_rm.bath < 5) & (df_rm.bath > 0)]

#df_rm.boxplot(column = 'dinning')
df_rm = df_rm[(df_rm.dinning < 5) & (df_rm.dinning >= 0)]

#df_rm.boxplot(column = 'total_floor')
df_rm = df_rm[(df_rm.total_floor > 0) & (df_rm.total_floor < 28)]

#df_rm.boxplot(column = 'total_fee')
df_rm = df_rm[df_rm.total_fee < 150000]

#-----
dt_sf = df_sf.describe().round(1)

#df_sf.boxplot(column = 'area')
df_sf = df_sf[(df_sf.area > 1) & (df_sf.area < 142)]

#df_sf.boxplot(column = 'room')
df_sf = df_sf[(df_sf.room >= 0) & (df_sf.room < 6)]

#df_sf.boxplot(column = 'bath')
df_sf = df_sf[(df_sf.bath >= 0) & (df_sf.bath < 5)]

#df_sf.boxplot(column = 'dinning')
df_sf = df_sf[(df_sf.dinning >= 0) & (df_sf.dinning < 5)]

#df_sf.boxplot(column = 'total_floor')
df_sf = df_sf[(df_sf.total_floor <= 30) & (df_sf.total_floor > 0)]

#df_sf.boxplot(column = 'total_fee')
df_sf = df_sf[df_sf.total_fee < 100000]
#%%---查看型態---處理index
#df_rm.dtypes 
#df_sf.dtypes 
df_rm.reset_index(inplace=True, drop=True)
df_sf.reset_index(inplace=True, drop=True)
#%%---區域編碼1
df_rm['district_num'] = df_rm['district']
df_rm['district_num'] = df_rm.district_num.replace('中區','1').replace('北區','2').replace('南區','3').replace('東區','4').replace('西區','5').replace('北屯區','6').replace('南屯區','7').replace('西屯區','8')
#df_rm.isna().sum() 

df_sf['district_num'] = df_sf['district']
df_sf['district_num'] = df_sf.district_num.replace('中區','1').replace('北區','2').replace('南區','3').replace('東區','4').replace('西區','5').replace('北屯區','6').replace('南屯區','7').replace('西屯區','8')
#df_sf.isna().sum() 
#%%---類型編碼2
df_rm['type_num'] = df_rm['obj_type']
df_rm['type_num'] = df_rm.type_num.replace('獨立套房','1').replace('分租套房','2').replace('整層住家','3').replace('住宅','4')
#df_rm.isna().sum() 

df_sf['type_num'] = df_sf['obj_type']
df_sf['type_num'] = df_sf.type_num.replace('店面','1').replace('辦公','2')
#df_sf.isna().sum() 
#%%---整理儲存
ncols_rm = ['description', 'addr', 'district', 'road', 'obj_type','district_num', 'type_num', 'area', 'room',
            'dinning', 'bath', 'which_floor', 'total_floor', 'total_fee']
df_rm_tidy = df_rm[ncols_rm]
df_rm_tidy.to_csv('rent_price_clean_rm.csv', index = False)

ncols_sf = ['description', 'addr', 'district', 'road', 'obj_type','district_num', 'type_num', 'area', 'room', 
            'dinning', 'bath', 'which_floor', 'total_floor', 'total_fee'] 
df_sf_tidy = df_sf[ncols_sf]
df_sf_tidy.to_csv('rent_price_clean_sf.csv', index = False)
