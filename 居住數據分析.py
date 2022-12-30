#%%
import warnings
warnings.filterwarnings('ignore')
#%%
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from matplotlib.font_manager import FontProperties
#%%---讀取居住用的資料
data_name_rm = 'rent_price_clean_rm.csv'
myFont = FontProperties(fname = 'msj.ttf', size = 20)

data_df_rm = pd.read_csv(data_name_rm)
#data_df_rm.head()
#%%---讀取辦公用的資料
data_name_sf = 'rent_price_clean_sf.csv'
myFont = FontProperties(fname = 'msj.ttf', size = 20)

data_df_sf = pd.read_csv(data_name_sf)
#data_df_sf.head()
#%%---設定之後看哪個特徵是重點
scores_rm = defaultdict(list)
scores_sf = defaultdict(list)
rs = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
#%%---居住資料預測
df_rm = data_df_rm.copy()

X = df_rm.iloc[:, 5:13].values
y = df_rm.iloc[:,13].values
feat_labels_rm = df_rm.columns[5:13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#---LinearRegression
lr_rm = LinearRegression(n_jobs = -1)
df_rm_lr = lr_rm.fit(X_train, y_train)
acc_rm_lr = r2_score(y_test, lr_rm.predict(X_test))
#acc_rm_lr

#---RandomForestRegressor----這個機率最高**採用這個
forest_rm = RandomForestRegressor(n_estimators = 1200, max_features = 2, random_state = 0, n_jobs = -1)
df_rm_forest = forest_rm.fit(X_train, y_train)
acc_rm_forest = r2_score(y_test, forest_rm.predict(X_test))
#acc_rm_forest

#---DecisionTreeRegressor
dtr_rm = DecisionTreeRegressor(max_depth = 4, random_state = 0)
df_rm_forest = dtr_rm.fit(X_train, y_train)
acc_rm_dtr = r2_score(y_test, dtr_rm.predict(X_test))
#acc_rm_dtr
#%%---查看居住特徵重點
for train_ind, test_ind in rs.split(X):
    X_Train, X_Test = X[train_ind], X[test_ind]
    y_Train, y_Test = y[train_ind], y[test_ind]
    _forest = forest_rm.fit(X_Train, y_Train)
    forest_acc = r2_score(y_Test, forest_rm.predict(X_Test))

    for i in range(X.shape[1]):
        X_t = X_Test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(y_Test, forest_rm.predict(X_t))
        scores_rm[feat_labels_rm[i]].append(((forest_acc - shuff_acc)/ forest_acc))
        
a = sorted([(round(np.mean(score), 6), feat) for feat, score in scores_rm.items()], reverse = True)

#for f in range(len(a)):
#    print('{:>2d}) {:<20s} \t {:>.6f}' .format(f + 1, a[f][1], a[f][0]))
#%%---採用比較有用的居住資料預測
df_rm = data_df_rm.copy()

X = df_rm[['district_num', 'area', 'room', 'dinning', 'bath', 'which_floor', 'total_floor']]
y = df_rm[['total_fee']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

forest_rm_op = RandomForestRegressor(n_estimators = 1200, max_features = 2, random_state = 0, n_jobs = -1)
df_rm_forest = forest_rm_op.fit(X_train, y_train)
acc_rm = r2_score(y_test, forest_rm_op.predict(X_test))
#acc_rm
#%%---商業資料預測
df_sf = data_df_sf.copy()

X = df_sf.iloc[:, 5:13].values
y = df_sf.iloc[:,13].values
feat_labels_sf = df_sf.columns[5:13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#---LinearRegression
lr_sf = LinearRegression(n_jobs = -1)
df_sf_lr = lr_sf.fit(X_train, y_train)
acc_sf_lr = r2_score(y_test, lr_sf.predict(X_test))
#acc_sf_lr

#---RandomForestRegressor
forest_sf = RandomForestRegressor(n_estimators = 1200, max_features = 2, random_state = 0, n_jobs = -1)
df_sf_forest = forest_sf.fit(X_train, y_train)
acc_sf_forest = r2_score(y_test, forest_sf.predict(X_test))
#acc_sf_forest

#---DecisionTreeRegressor
dtr_sf = DecisionTreeRegressor(max_depth = 4, random_state = 0)
df_sf_forest = dtr_sf.fit(X_train, y_train)
acc_sf_dtr = r2_score(y_test, dtr_sf.predict(X_test))
#acc_sf_dtr
#%%---查看商業特徵重點
for train_ind, test_ind in rs.split(X):
    X_Train, X_Test = X[train_ind], X[test_ind]
    y_Train, y_Test = y[train_ind], y[test_ind]
    _forest = forest_sf.fit(X_Train, y_Train)
    forest_acc = r2_score(y_Test, forest_sf.predict(X_Test))
#    print('---------------')
#    print(forest_acc)
    
    for i in range(X.shape[1]):
        X_t = X_Test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(y_Test, forest_sf.predict(X_t))
        scores_sf[feat_labels_sf[i]].append(((forest_acc - shuff_acc)/ forest_acc))
#        print(shuff_acc)
        
a = sorted([(round(np.mean(score), 6), feat) for feat, score in scores_sf.items()], reverse = True)

#for f in range(len(a)):
#    print('{:>2d}) {:<20s} \t {:>.6f}' .format(f + 1, a[f][1], a[f][0]))
#%%---採用比較有用的商業資料預測---但發現沒有比較好
df_sf = data_df_sf.copy()

X = df_sf[['district_num', 'type_num', 'area', 'room', 'which_floor', 'total_floor']]
y = df_sf[['total_fee']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

forest_sf_op = RandomForestRegressor(n_estimators = 1200, max_features = 2, random_state = 0, n_jobs = -1)
df_sf_forest = forest_sf_op.fit(X_train, y_train)
acc_sf_forest = r2_score(y_test, forest_sf_op.predict(X_test))
acc_sf_forest

#---有趣的測試1---只看面積
'''
df_sf = data_df_sf.copy()

X = df_sf[['area', 'type_num']]
y = df_sf[['total_fee']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

forest_sf = RandomForestRegressor(n_estimators = 1200, max_features = 2, random_state = 0, n_jobs = -1)
df_sf_forest_1 = forest_sf.fit(X_train, y_train)
acc_sf_forest_1 = r2_score(y_test, forest_sf.predict(X_test))
#acc_sf_forest_1
'''
#---有趣的測試2---不看面積
'''
df_sf = data_df_sf.copy()

X = df_sf[['district_num', 'type_num', 'room', 'dinning', 'bath', 'which_floor', 'total_floor']]
y = df_sf[['total_fee']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

df_sf_forest_2 = forest_sf.fit(X_train, y_train)
acc_sf_forest_2 = r2_score(y_test, forest_sf.predict(X_test))
#acc_sf_forest_2
'''
#%%---方便分辨而創立的字典
dic_district = {'中區': '1', '北區': '2', '南區': '3', '東區': '4', '西區': '5', '北屯區': '6', '南屯區': '7', '西屯區': '8'}
dic_type_1 = {'獨立套房': '1', '分租套房': '2', '整層住家': '3', '住宅': '4'}
dic_type_2 = {'店面': '1', '辦公': '2'}
#%%---預測估價
while True:
    a = input('請輸入您要估計的類型用途(居住/商業): ')
    if a == '居住' or '商業':
        pass
    else:
        print('估計類型輸入錯誤！請再重新輸入一次！')
        break
    
    df_area = input('面積(坪) = ')
    df_room = input('房間數 = ')
    df_dinning = input('廳數 = ')
    df_bath = input('衛浴數 = ')
    df_which_floor = input('樓層數 = ')
    df_total_floor = input('總樓層數 = ')
    
    if int(df_total_floor) < int(df_which_floor):
        print('樓層輸入錯誤！請再重新輸入一次！')
        break
        
    else:
        if a == '居住':
            typee = input('類型(獨立套房/分租套房/整層住家/住宅) = ')
            if typee == '獨立套房' or  '分租套房' or '整層住家' or '住宅':
                df_type_num = dic_type_1[typee]
            else:
                print('類型輸入錯誤！請再重新輸入一次！')
                break
                
        elif a == '商業':
            typee = input('類型(店面/辦公) = ')
            if typee == '店面' or  '辦公':
                df_type_num = dic_type_2[typee]
            else:
                print('類型輸入錯誤！請再重新輸入一次！')
                break
    
        districtt = input('區域(中區/北區/南區/東區/西區/北屯區/南屯區/西屯區) = ')
        if districtt == '中區' or '北區' or '南區' or '東區' or '西區' or '北屯區' or '南屯區' or '西屯區':
            df_district_num = dic_district[districtt]
        else:
            print('區域輸入錯誤！請再重新輸入一次！')
            break
            
        if a == '居住':
            n = np.array([df_district_num, df_area, df_room, df_dinning, df_bath, df_which_floor, df_total_floor]).reshape(1, -1)
            n_pre = forest_rm_op.predict(n.reshape(1, -1))
            total_fee_pre = int(n_pre)
            print(f'估計月租金(含管理費)建議定價為: {total_fee_pre} 元。')
            
        elif a == '商業':
            n = np.array([df_district_num, df_type_num, df_area, df_room, df_dinning, df_total_floor]).reshape(1, -1)
            n_pre = forest_sf_op.predict(n.reshape(1, -1))
            total_fee_pre = int(n_pre)
            print(f'估計月租金(含管理費)建議定價為: {total_fee_pre} 元。')
        
        b = input('是否繼續估計(Y/N) = ')
        
        if b == 'Y':
            pass
        else:
            print('謝謝您的使用！')
            break
