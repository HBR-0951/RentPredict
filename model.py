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
        # %%---????????????
        df_rm = df[(df.obj_type == '????????????') + (df.obj_type == '????????????') +
                   (df.obj_type == '??????') + (df.obj_type == '????????????')]
        df_sf = df[(df.obj_type == '??????') + (df.obj_type == '??????')]
        # %%---???????????????
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
        # %%---????????????-----?????????
        l = ['???/???', '/???', '$', '?????? ', ' ???', '???', ',']
        for i in l:
            df_rm['mgnt_fee'] = df_rm.mgnt_fee.apply(
                lambda x: str(x).replace(i, ''))
        l2 = ['0???(???????????????)', '0???(???????????????)', '???', '????????????', '????????????', '????????????',
              '???', '??????', '???????????????', '???????????????', '???0?????????', '???0', '???0?????????']
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
        l = ['???/???', '/???', '$', '?????? ', ' ???', '???', ',']
        for i in l:
            df_sf['mgnt_fee'] = df_sf.mgnt_fee.apply(
                lambda x: str(x).replace(i, ''))
        l2 = ['0???(???????????????)', '0???(???????????????)', '???', '????????????', '????????????', '????????????',
              '???', '??????', '???????????????', '???????????????', '???0?????????', '???0', '???0?????????']
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
        # %%---????????????-----??????
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
        # %%---???????????????---?????????
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
        # %%---????????????---??????index
        # df_rm.dtypes
        # df_sf.dtypes
        df_rm.reset_index(inplace=True, drop=True)
        df_sf.reset_index(inplace=True, drop=True)
        # %%---????????????1
        df_rm['district_num'] = df_rm['district']
        df_rm['district_num'] = df_rm.district_num.replace('??????', '1').replace('??????', '2').replace('??????', '3').replace(
            '??????', '4').replace('??????', '5').replace('?????????', '6').replace('?????????', '7').replace('?????????', '8')
        # df_rm.isna().sum()

        df_sf['district_num'] = df_sf['district']
        df_sf['district_num'] = df_sf.district_num.replace('??????', '1').replace('??????', '2').replace('??????', '3').replace(
            '??????', '4').replace('??????', '5').replace('?????????', '6').replace('?????????', '7').replace('?????????', '8')
        # df_sf.isna().sum()
        # %%---????????????2
        df_rm['type_num'] = df_rm['obj_type']
        df_rm['type_num'] = df_rm.type_num.replace('????????????', '1').replace(
            '????????????', '2').replace('????????????', '3').replace('??????', '4')
        # df_rm.isna().sum()

        df_sf['type_num'] = df_sf['obj_type']
        df_sf['type_num'] = df_sf.type_num.replace(
            '??????', '1').replace('??????', '2')
        # df_sf.isna().sum()
        # %%---????????????
        ncols_rm = ['description', 'addr', 'district', 'road', 'obj_type', 'district_num', 'type_num', 'area', 'room',
                    'dinning', 'bath', 'which_floor', 'total_floor', 'total_fee']
        df_rm_tidy = df_rm[ncols_rm]
        df_rm_tidy.to_csv('rent_price_clean_rm.csv', index=False)

        ncols_sf = ['description', 'addr', 'district', 'road', 'obj_type', 'district_num', 'type_num', 'area', 'room',
                    'dinning', 'bath', 'which_floor', 'total_floor', 'total_fee']
        df_sf_tidy = df_sf[ncols_sf]
        df_sf_tidy.to_csv('rent_price_clean_sf.csv', index=False)

    def DataAnalysis(self):

        # %%---????????????????????????
        data_name_rm = 'rent_price_clean_rm.csv'
        myFont = FontProperties(fname='msj.ttf', size=20)

        data_df_rm = pd.read_csv(data_name_rm)
        # data_df_rm.head()
        # %%---????????????????????????
        data_name_sf = 'rent_price_clean_sf.csv'
        myFont = FontProperties(fname='msj.ttf', size=20)

        data_df_sf = pd.read_csv(data_name_sf)
        # data_df_sf.head()
        # %%---????????????????????????????????????
        scores_rm = defaultdict(list)
        scores_sf = defaultdict(list)
        rs = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        # %%---??????????????????
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

        # ---RandomForestRegressor----??????????????????**????????????
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
        # %%---????????????????????????
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
        # %%---???????????????????????????????????????
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
        # %%---??????????????????
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
        # %%---????????????????????????
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
        # %%---???????????????????????????????????????---????????????????????????
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

    def PredictRent(self):

        # %%---??????????????????????????????
        dic_district = {'??????': '1', '??????': '2', '??????': '3', '??????': '4',
                        '??????': '5', '?????????': '6', '?????????': '7', '?????????': '8'}
        dic_type_1 = {'????????????': '1', '????????????': '2', '????????????': '3', '??????': '4'}
        dic_type_2 = {'??????': '1', '??????': '2'}
        # %%---????????????
        while True:
            a = input('????????????????????????????????????(??????/??????): ')
            if a != "??????" and a != "??????":
                print('??????????????????????????????????????????????????????')
                continue

            df_area = input('??????(???) = ')
            df_room = input('????????? = ')
            df_dinning = input('?????? = ')
            df_bath = input('????????? = ')
            df_which_floor = input('????????? = ')
            df_total_floor = input('???????????? = ')

            if int(df_total_floor) < int(df_which_floor):
                print('????????????????????????????????????????????????')
                continue

            else:
                if a == '??????':
                    typee = input('??????(????????????/????????????/????????????/??????) = ')
                    if typee == '????????????' or '????????????' or '????????????' or '??????':
                        df_type_num = dic_type_1[typee]
                    else:
                        print('????????????????????????????????????????????????')
                        continue

                elif a == '??????':
                    typee = input('??????(??????/??????) = ')
                    if typee == '??????' or '??????':
                        df_type_num = dic_type_2[typee]
                    else:
                        print('????????????????????????????????????????????????')
                        continue

                districtt = input('??????(??????/??????/??????/??????/??????/?????????/?????????/?????????) = ')
                if districtt == '??????' or '??????' or '??????' or '??????' or '??????' or '?????????' or '?????????' or '?????????':
                    df_district_num = dic_district[districtt]
                else:
                    print('????????????????????????????????????????????????')
                    continue

                if a == '??????':
                    n = np.array([df_district_num, df_area, df_room, df_dinning,
                                  df_bath, df_which_floor, df_total_floor]).reshape(1, -1)
                    n_pre = self.forest_rm_op.predict(n.reshape(1, -1))
                    total_fee_pre = int(n_pre)
                    print(f'???????????????(????????????)???????????????: {total_fee_pre} ??????')

                elif a == '??????':
                    n = np.array([df_district_num, df_type_num, df_area,
                                  df_room, df_dinning, df_total_floor]).reshape(1, -1)
                    n_pre = self.forest_sf_op.predict(n.reshape(1, -1))
                    total_fee_pre = int(n_pre)
                    print(f'???????????????(????????????)???????????????: {total_fee_pre} ??????')
            break
