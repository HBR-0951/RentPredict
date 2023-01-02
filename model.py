# %%
from matplotlib.font_manager import FontProperties
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
import pandas as pd
import warnings
import requests as req
import json
import time
import random
warnings.filterwarnings('ignore')
# %%


class model:
    def __init__(self):
        self.forest_sf_op = None
        self.forest_rm_op = None
        self.OnInitialized()

    def OnInitialized(self):
        print("Initialized model!")
        # self.Crawler()
        # self.PreProcessing()
        self.DataAnalysis()
        print("Initialized success!")

    def Crawler(self):
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
        # %%
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER'
        }
        # %%

        # %%
        lst_rent_fee = []

        print('Requesting data from web...')
        print('---> Requesting data in the page 1...', end='', flush=True)

        url = 'https://rent.houseprice.tw/ws/list/%E5%8F%B0%E4%B8%AD%E5%B8%82_city/%E8%A5%BF%E5%B1%AF%E5%8D%80-%E5%8D%97%E5%B1%AF%E5%8D%80-%E5%8C%97%E5%B1%AF%E5%8D%80-%E5%8C%97%E5%8D%80-%E8%A5%BF%E5%8D%80-%E4%B8%AD%E5%8D%80-%E6%9D%B1%E5%8D%80-%E5%8D%97%E5%8D%80_zip/'

        OK, ret_data = web_scraping_one_page(url)
        if OK:
            lst_rent_fee += ret_data
        else:
            print('Error to get data in the 1st page')

        cnt = 1

        total_cnt = 702

        for p in range(2, total_cnt):
            print('\r---> Requesting data in the page {0}...({1:.1f}%)'.format(
                p, cnt/(total_cnt-1)*100), end='', flush=True)
            url_n = '{0}?p={1}'.format(url, p)
            OK, ret_data = web_scraping_one_page(url_n)
            if OK:
                lst_rent_fee += ret_data
            else:
                print('Error to get data in page {0}'.format(p))
            time.sleep(random.randint(1, 3))
            cnt += 1

        print('\nComplete! Total {0} records scrapped.'.format(
            len(lst_rent_fee)))
        # %%
        cols = ['addr', 'district', 'road', 'description', 'area', 'room', 'bath',
                'dinning', 'obj_type', 'which_floor', 'total_floor', 'mgnt_fee', 'rent_fee']
        df = pd.DataFrame(lst_rent_fee, columns=cols)

        df.to_csv('rent_price_raw.csv', index=False)
        print('File saved !!')

    def PreProcessing(self):
        # %%
        df = pd.read_csv('rent_price_raw.csv')
        df['obj_type'].value_counts()
        # %%---分類資料
        df_rm = df[(df.obj_type == '獨立套房') + (df.obj_type == '整層住家') +
                   (df.obj_type == '住宅') + (df.obj_type == '分租套房')]
        df_sf = df[(df.obj_type == '店面') + (df.obj_type == '辦公')]
        # %%---處理缺失值
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
        # %%---處理雜項-----管理費
        l = ['元/月', '/月', '$', '月繳 ', ' 元', '約', ',']
        for i in l:
            df_rm['mgnt_fee'] = df_rm.mgnt_fee.apply(
                lambda x: str(x).replace(i, ''))
        l2 = ['0元(已含租金內)', '0元(不含租金內)', '無', '租金內含', '租金已含', '租金含管',
              '含', '內含', '內含在租金', '已含租金內', '已0租金內', '內0', '內0在租金']
        for j in l2:
            df_rm['mgnt_fee'] = df_rm.mgnt_fee.apply(
                lambda x: str(x).replace(j, '0'))

        z1, z2 = [], []
        for z in range(0, len(df_rm.index)):
            y = df_rm.mgnt_fee[z].isdigit()
            if y == False:
                z1.append(z)
        df_rm = df_rm.drop(index=z1)

        df_rm['total_fee'] = df_rm.mgnt_fee.astype(
            float) + df_rm.rent_fee.astype(float)
        df_rm['mgnt_fee'] = df_rm['mgnt_fee'].astype('float64')

        # -----
        l = ['元/月', '/月', '$', '月繳 ', ' 元', '約', ',']
        for i in l:
            df_sf['mgnt_fee'] = df_sf.mgnt_fee.apply(
                lambda x: str(x).replace(i, ''))
        l2 = ['0元(已含租金內)', '0元(不含租金內)', '無', '租金內含', '租金已含', '租金含管',
              '含', '內含', '內含在租金', '已含租金內', '已0租金內', '內0', '內0在租金']
        for j in l2:
            df_sf['mgnt_fee'] = df_sf.mgnt_fee.apply(
                lambda x: str(x).replace(j, '0'))

        z2 = []
        for z in range(0, len(df_sf.index)):
            y = df_sf.mgnt_fee[z].isdigit()
            if y == False:
                z2.append(z)
        df_sf = df_sf.drop(index=z2)

        df_sf['total_fee'] = df_sf.mgnt_fee.astype(
            float) + df_sf.rent_fee.astype(float)
        df_sf['mgnt_fee'] = df_sf['mgnt_fee'].astype('float64')
        # %%---處理雜項-----樓層
        l3 = ['--', 'B1']
        for i in l3:
            df_rm['which_floor'] = df_rm.which_floor.apply(
                lambda x: x.replace(i, '0'))
        df_rm['which_floor'] = df_rm['which_floor'].astype('int32')
        #df_rm.boxplot(column = 'which_floor')
        df_rm = df_rm[(df_rm.which_floor > 0) & (df_rm.which_floor < 18)]

        # -----
        df_sf['which_floor'] = df_sf.which_floor.apply(
            lambda x: x.replace('--', '0'))
        df_sf['which_floor'] = df_sf.which_floor.apply(
            lambda x: x.replace('B1', '-1'))
        df_sf['which_floor'] = df_sf['which_floor'].astype('int32')
        #df_sf.boxplot(column = 'which_floor')
        df_sf = df_sf[(df_sf.which_floor != 0) & (df_sf.which_floor < 50)]
        # %%---處理離群值---其他ㄉ
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

        # -----
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
        # %%---查看型態---處理index
        # df_rm.dtypes
        # df_sf.dtypes
        df_rm.reset_index(inplace=True, drop=True)
        df_sf.reset_index(inplace=True, drop=True)
        # %%---區域編碼1
        df_rm['district_num'] = df_rm['district']
        df_rm['district_num'] = df_rm.district_num.replace('中區', '1').replace('北區', '2').replace('南區', '3').replace(
            '東區', '4').replace('西區', '5').replace('北屯區', '6').replace('南屯區', '7').replace('西屯區', '8')
        # df_rm.isna().sum()

        df_sf['district_num'] = df_sf['district']
        df_sf['district_num'] = df_sf.district_num.replace('中區', '1').replace('北區', '2').replace('南區', '3').replace(
            '東區', '4').replace('西區', '5').replace('北屯區', '6').replace('南屯區', '7').replace('西屯區', '8')
        # df_sf.isna().sum()
        # %%---類型編碼2
        df_rm['type_num'] = df_rm['obj_type']
        df_rm['type_num'] = df_rm.type_num.replace('獨立套房', '1').replace(
            '分租套房', '2').replace('整層住家', '3').replace('住宅', '4')
        # df_rm.isna().sum()

        df_sf['type_num'] = df_sf['obj_type']
        df_sf['type_num'] = df_sf.type_num.replace(
            '店面', '1').replace('辦公', '2')
        # df_sf.isna().sum()
        # %%---整理儲存
        ncols_rm = ['description', 'addr', 'district', 'road', 'obj_type', 'district_num', 'type_num', 'area', 'room',
                    'dinning', 'bath', 'which_floor', 'total_floor', 'total_fee']
        df_rm_tidy = df_rm[ncols_rm]
        df_rm_tidy.to_csv('rent_price_clean_rm.csv', index=False)

        ncols_sf = ['description', 'addr', 'district', 'road', 'obj_type', 'district_num', 'type_num', 'area', 'room',
                    'dinning', 'bath', 'which_floor', 'total_floor', 'total_fee']
        df_sf_tidy = df_sf[ncols_sf]
        df_sf_tidy.to_csv('rent_price_clean_sf.csv', index=False)

    def DataAnalysis(self):

        # %%---讀取居住用的資料
        data_name_rm = 'rent_price_clean_rm.csv'
        myFont = FontProperties(fname='msj.ttf', size=20)

        data_df_rm = pd.read_csv(data_name_rm)
        # data_df_rm.head()
        # %%---讀取辦公用的資料
        data_name_sf = 'rent_price_clean_sf.csv'
        myFont = FontProperties(fname='msj.ttf', size=20)

        data_df_sf = pd.read_csv(data_name_sf)
        # data_df_sf.head()
        # %%---設定之後看哪個特徵是重點
        scores_rm = defaultdict(list)
        scores_sf = defaultdict(list)
        rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        # %%---居住資料預測
        df_rm = data_df_rm.copy()

        X = df_rm.iloc[:, 5:13].values
        y = df_rm.iloc[:, 13].values
        feat_labels_rm = df_rm.columns[5:13]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        # ---LinearRegression
        lr_rm = LinearRegression(n_jobs=-1)
        df_rm_lr = lr_rm.fit(X_train, y_train)
        acc_rm_lr = r2_score(y_test, lr_rm.predict(X_test))
        # acc_rm_lr

        # ---RandomForestRegressor----這個機率最高**採用這個
        forest_rm = RandomForestRegressor(
            n_estimators=1200, max_features=2, random_state=0, n_jobs=-1)
        df_rm_forest = forest_rm.fit(X_train, y_train)
        acc_rm_forest = r2_score(y_test, forest_rm.predict(X_test))
        # acc_rm_forest

        # ---DecisionTreeRegressor
        dtr_rm = DecisionTreeRegressor(max_depth=4, random_state=0)
        df_rm_forest = dtr_rm.fit(X_train, y_train)
        acc_rm_dtr = r2_score(y_test, dtr_rm.predict(X_test))
        # acc_rm_dtr
        # %%---查看居住特徵重點
        for train_ind, test_ind in rs.split(X):
            X_Train, X_Test = X[train_ind], X[test_ind]
            y_Train, y_Test = y[train_ind], y[test_ind]
            _forest = forest_rm.fit(X_Train, y_Train)
            forest_acc = r2_score(y_Test, forest_rm.predict(X_Test))

            for i in range(X.shape[1]):
                X_t = X_Test.copy()
                np.random.shuffle(X_t[:, i])
                shuff_acc = r2_score(y_Test, forest_rm.predict(X_t))
                scores_rm[feat_labels_rm[i]].append(
                    ((forest_acc - shuff_acc) / forest_acc))

        a = sorted([(round(np.mean(score), 6), feat)
                    for feat, score in scores_rm.items()], reverse=True)

        # for f in range(len(a)):
        #    print('{:>2d}) {:<20s} \t {:>.6f}' .format(f + 1, a[f][1], a[f][0]))
        # %%---採用比較有用的居住資料預測
        df_rm = data_df_rm.copy()

        X = df_rm[['district_num', 'area', 'room',
                   'dinning', 'bath', 'which_floor', 'total_floor']]
        y = df_rm[['total_fee']]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        self.forest_rm_op = RandomForestRegressor(
            n_estimators=1200, max_features=2, random_state=0, n_jobs=-1)
        df_rm_forest = self.forest_rm_op.fit(X_train, y_train)
        acc_rm = r2_score(y_test, self.forest_rm_op.predict(X_test))
        # acc_rm
        # %%---商業資料預測
        df_sf = data_df_sf.copy()

        X = df_sf.iloc[:, 5:13].values
        y = df_sf.iloc[:, 13].values
        feat_labels_sf = df_sf.columns[5:13]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        # ---LinearRegression
        lr_sf = LinearRegression(n_jobs=-1)
        df_sf_lr = lr_sf.fit(X_train, y_train)
        acc_sf_lr = r2_score(y_test, lr_sf.predict(X_test))
        # acc_sf_lr

        # ---RandomForestRegressor
        forest_sf = RandomForestRegressor(
            n_estimators=1200, max_features=2, random_state=0, n_jobs=-1)
        df_sf_forest = forest_sf.fit(X_train, y_train)
        acc_sf_forest = r2_score(y_test, forest_sf.predict(X_test))
        # acc_sf_forest

        # ---DecisionTreeRegressor
        dtr_sf = DecisionTreeRegressor(max_depth=4, random_state=0)
        df_sf_forest = dtr_sf.fit(X_train, y_train)
        acc_sf_dtr = r2_score(y_test, dtr_sf.predict(X_test))
        # acc_sf_dtr
        # %%---查看商業特徵重點
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
                scores_sf[feat_labels_sf[i]].append(
                    ((forest_acc - shuff_acc) / forest_acc))
        #        print(shuff_acc)

        a = sorted([(round(np.mean(score), 6), feat)
                    for feat, score in scores_sf.items()], reverse=True)

        # for f in range(len(a)):
        #    print('{:>2d}) {:<20s} \t {:>.6f}' .format(f + 1, a[f][1], a[f][0]))
        # %%---採用比較有用的商業資料預測---但發現沒有比較好
        df_sf = data_df_sf.copy()

        X = df_sf[['district_num', 'type_num', 'area',
                   'room', 'which_floor', 'total_floor']]
        y = df_sf[['total_fee']]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

        self.forest_sf_op = RandomForestRegressor(
            n_estimators=1200, max_features=2, random_state=0, n_jobs=-1)
        df_sf_forest = self.forest_sf_op.fit(X_train, y_train)
        acc_sf_forest = r2_score(y_test, self.forest_sf_op.predict(X_test))
        acc_sf_forest

    def PredictRent(self, use, area, room, dinning, bath, floor, totalFloor, types, district):

        # %%---方便分辨而創立的字典
        dic_district = {'中區': '1', '北區': '2', '南區': '3', '東區': '4',
                        '西區': '5', '北屯區': '6', '南屯區': '7', '西屯區': '8'}
        dic_type_1 = {'獨立套房': '1', '分租套房': '2', '整層住家': '3', '住宅': '4'}
        dic_type_2 = {'店面': '1', '辦公': '2'}
        # %%---預測估價
        
        # print(use, area, room, dinning, bath, floor, totalFloor, types, district)

        df_district_num = dic_district[district]

        if use == '居住':
            df_type_num = dic_type_1[types]

            n = np.array([df_district_num, area, room, dinning,
                        bath, floor, totalFloor]).reshape(1, -1)

            n_pre = self.forest_rm_op.predict(n.reshape(1, -1))
            total_fee_pre = str(int(n_pre))
            # print(total_fee_pre)
            return total_fee_pre

        elif use == '商業':
            df_type_num = dic_type_2[types]

            n = np.array([df_district_num, df_type_num, area,
                        room, dinning, totalFloor]).reshape(1, -1)

            n_pre = self.forest_sf_op.predict(n.reshape(1, -1))
            total_fee_pre = str(int(n_pre))
            # print(total_fee_pre)
            return total_fee_pre

        # while True:
        #     a = input('用途(居住/商業): ')
        #     if a != "居住" and a != "商業":
        #         print('估計類型輸入錯誤！請再重新輸入一次！')
        #         continue

        #     df_area = input('面積(坪) = ')
        #     df_room = input('房間數 = ')
        #     df_dinning = input('廳數 = ')
        #     df_bath = input('衛浴數 = ')
        #     df_which_floor = input('樓層數 = ')
        #     df_total_floor = input('總樓層數 = ')

        #     if int(df_total_floor) < int(df_which_floor):
        #         print('樓層輸入錯誤！請再重新輸入一次！')
        #         continue

        #     else:
        #         if a == '居住':
        #             typee = input('類型(獨立套房/分租套房/整層住家/住宅) = ')
        #             if typee == '獨立套房' or '分租套房' or '整層住家' or '住宅':
        #                 df_type_num = dic_type_1[typee]
        #             else:
        #                 print('類型輸入錯誤！請再重新輸入一次！')
        #                 continue

        #         elif a == '商業':
        #             typee = input('類型(店面/辦公) = ')
        #             if typee == '店面' or '辦公':
        #                 df_type_num = dic_type_2[typee]
        #             else:
        #                 print('類型輸入錯誤！請再重新輸入一次！')
        #                 continue

        #         districtt = input('區域(中區/北區/南區/東區/西區/北屯區/南屯區/西屯區) = ')
        #         if districtt == '中區' or '北區' or '南區' or '東區' or '西區' or '北屯區' or '南屯區' or '西屯區':
        #             df_district_num = dic_district[districtt]
        #         else:
        #             print('區域輸入錯誤！請再重新輸入一次！')
        #             continue

        #         if a == '居住':
        #             n = np.array([df_district_num, df_area, df_room, df_dinning,
        #                           df_bath, df_which_floor, df_total_floor]).reshape(1, -1)
        #             n_pre = self.forest_rm_op.predict(n.reshape(1, -1))
        #             total_fee_pre = int(n_pre)
        #             print(f'估計月租金(含管理費)建議定價為: {total_fee_pre} 元。')

        #         elif a == '商業':
        #             n = np.array([df_district_num, df_type_num, df_area,
        #                           df_room, df_dinning, df_total_floor]).reshape(1, -1)
        #             n_pre = self.forest_sf_op.predict(n.reshape(1, -1))
        #             total_fee_pre = int(n_pre)
        #             print(f'估計月租金(含管理費)建議定價為: {total_fee_pre} 元。')
        #     break
