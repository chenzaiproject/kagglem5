import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import preprocessing, metrics
from sklearn.preprocessing import LabelEncoder
import gc
import os
from datetime import datetime,timedelta
from tqdm.notebook import tqdm
from scipy.sparse import csr_matrix
from google.colab import drive
import joblib
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
import math
from math import ceil
from datetime import datetime

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }

PRICE_DTYPES = {"item_id": "category", "store_id": "category", "wm_yr_wk": "int16", "sell_price": "float32" }

h = 28
tr_last = 1941
def create_dt(nrows = None, first_day = 1200):
    prices = pd.read_csv("./sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
    cal = pd.read_csv("./calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    start_day = max(1, first_day)
    numcols = [f"d_{day}" for day in range(start_day, tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols}
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("./sales_train_evaluation.csv", nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    for day in range(tr_last+1, tr_last+28+1):
        dt[f"d_{day}"] = np.nan
    dt = pd.melt(dt,id_vars = catcols,value_vars = [col for col in dt.columns if col.startswith("d_")],var_name = "d",value_name = "sales")
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    return dt
  
%%time
FIRST_DAY = 1
data = create_dt(first_day = FIRST_DAY)


import math, decimal
dec = decimal.Decimal
def get_moon_phase(d):
    diff = d - datetime(2001, 1, 1)
    days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
    lunations = dec("0.20439731") + (days * dec("0.03386319269"))
    phase_index = math.floor((lunations % dec(1) * dec(8)) + dec('0.5'))
    return int(phase_index) & 7

def create_fea(df):
    df.rename(columns={'sales':'demand'}, inplace=True)
    print('类别编码')
    icols = [['item_id', 'store_id']]
    for col in icols:
        col_name = '_'+'_'.join(col)+'_'
        df['enc'+col_name+'mean'] = df.groupby(col)['demand'].transform('mean').astype(np.float16)
        df['enc'+col_name+'std'] = df.groupby(col)['demand'].transform('std').astype(np.float16)
    print('lag特征')
    df['lag_t7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(7)).astype(np.float16)
    df['lag_t14'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(14)).astype(np.float16)
    df['lag_t21'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(21)).astype(np.float16)
    df['lag_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28)).astype(np.float16)
    df['lag_t35'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(35)).astype(np.float16)
    print('rolling特征')
    for i in [7,14,21,28]:
        for j in [7,14,30,45,60]:
            df['rolling_mean'+str(i)+'_t'+str(j)] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(i).rolling(j).mean()).astype(np.float16)
            if i == 28:
                df['rolling_std'+str(i)+'_t'+str(j)] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(i).rolling(j).std()).astype(np.float16)
                
    df = df[(df.date >= '2016-04-25') | (pd.notna(df.rolling_mean28_t60))]
    print('处理时间')
    df['wday'] = getattr(df["date"].dt, 'dayofweek').astype("int8") 
    df['mday'] = getattr(df["date"].dt, 'day').astype("int8") 
    df['week'] = getattr(df["date"].dt, 'weekofyear').astype("int8") 
    df['month'] = getattr(df["date"].dt, 'month').astype("int8") 
    df['quarter'] = getattr(df["date"].dt, 'quarter').astype("int8") 
    df['year'] = getattr(df["date"].dt, 'year').astype("int16") 
    df['year'] = (df['year'] - df['year'].min()).astype("int8")
    df['week_month'] = df['mday'].apply(lambda x: ceil(x/7)).astype("int8") 
    df['week_end'] = (df['wday'] >= 5).astype("int8")  
    df['moon'] = df.date.apply(get_moon_phase) 
    
    print('处理价格')
    df['price_max'] = df.groupby(['store_id','item_id'])['sell_price'].transform('max') 
    df['price_min'] = df.groupby(['store_id','item_id'])['sell_price'].transform('min') 
    df['price_std'] = df.groupby(['store_id','item_id'])['sell_price'].transform('std') 
    df['price_mean'] = df.groupby(['store_id','item_id'])['sell_price'].transform('mean') 
    df['price_median'] = df.groupby(['store_id', 'item_id'])['sell_price'].transform('median') 
    df['price_dif'] = df['price_median'] - df['price_mean'] 
    df['price_norm'] = df['sell_price'] / df['price_max'] 
    df['price_nunique'] = df.groupby(['store_id', 'item_id'])['sell_price'].transform('nunique')
    df['item_nunique'] = df.groupby(['store_id', 'sell_price'])['item_id'].transform('nunique') 
    df['price_momentum'] = df['sell_price'] / df.groupby(['store_id', 'item_id'])['sell_price'].transform(lambda x: x.shift(1))
    df['price_momentum_m'] = df['sell_price'] / df.groupby(['store_id', 'item_id', 'month'])['sell_price'].transform('mean')
    df['price_momentum_y'] = df['sell_price'] / df.groupby(['store_id', 'item_id', 'year'])['sell_price'].transform('mean')
    return df
  
  
  
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
data = create_fea(data)
data = reduce_mem_usage(data)


ex_f = ['id', 'demand', 'sales', 'date', 'd', 'weekday', 'wm_yr_wk', 'state_id', 'store_id']
features = [i for i in data.columns if i not in ex_f]
cat_feats = ['item_id', 'dept_id', 'cat_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"] + ['snap_CA', 'snap_WI', 'snap_TX']
all_features = features.copy()

### xgboost
import xgboost as xgb
def run_model(store_id, num_iter, data):
    
    df = data[data['store_id'] == store_id].copy()
    x_train = df[df['date'] <= '2016-04-24']
    y_train = x_train['demand']
    x_val = df[(df['date'] > '2016-04-24') & (df['date'] <= '2016-05-22')]
    y_val = x_val['demand']
    train_set = xgb.DMatrix(x_train[features],label=y_train)
    val_set = xgb.DMatrix(x_val[features],label=y_val)
    watch_list = [(val_set, 'eval'), (train_set, 'train')]
    param = {'max_depth': 5, 'eta': 0.01, 'silent': 1, 'objective': 'reg:linear','num_boost_round':950,'subsample':0.8,'colsample_bytree':0.2319,'min_child_weight':11,'verbose_eval':20}
    model = xgb.train(param, train_set, num_boost_round=num_iter,early_stopping_rounds=100, evals=watch_list)
    return model
  
### lightgbm
# def run_model(store_id, num_iter, data):
#     df = data[data['store_id'] == store_id].copy()
#     x_train = df[df['date'] <= '2016-04-24']
#     y_train = x_train['demand']
#     x_val = df[(df['date'] > '2016-04-24') & (df['date'] <= '2016-05-22')]
#     y_val = x_val['demand']
#     train_set = lgb.Dataset(x_train[features],y_train,categorical_feature=cat_feats,free_raw_data=False,weight=x_train['sample_weight'])
#     val_set = lgb.Dataset(x_val[features],y_val,categorical_feature=cat_feats,free_raw_data=False,weight=x_val['sample_weight'])
#     params = {
#             'boosting_type': 'gbdt',
#             'objective': 'tweedie',
#             'tweedie_variance_power': 1.1,
#             'metric': 'rmse',
#             'subsample': 0.85,
#             'subsample_freq': 1,
#             'learning_rate': 0.03,
#             'num_leaves': 2**11-1,
#             'min_data_in_leaf': 2**12-1,
#             'feature_fraction': 0.6,
#             'max_bin': 100,
#             'boost_from_average': False,
#             'verbose': -1,
#             'seed': 42
#           }
#     model = lgb.train(params, train_set, num_boost_round=num_iter, early_stopping_rounds=100, valid_sets=[train_set, val_set], verbose_eval=100)
#     return model



for i in range(10):
    print('Model', i+1)
    model_name = 'store_' + str(i) + '.sav'
    model = run_model(i, 2000, data)
    joblib.dump(model, model_name)
    
    
    
    
def simple_fe(df):
    df['lag_t7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(7)).astype(np.float16)
    df['lag_t14'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(14)).astype(np.float16)
    df['lag_t21'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(21)).astype(np.float16)
    df['lag_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28)).astype(np.float16)
    for i in [7,14,21]:
        for j in [7,14,30,45,60]:
            df['rolling_mean'+str(i)+'_t'+str(j)] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(i).rolling(j).mean()).astype(np.float16)
    return df
  
  
def get_test7(df, coe, model, store_id):
    data0 = df[df['store_id'] == store_id].copy()
    print('First 7')
    test = data0[(data0['date'] > '2016-04-24') & (data0['date'] <= '2016-05-01')]
    data0.loc[(data0['date'] > '2016-04-24') & (data0['date'] <= '2016-05-01'), 'demand'] = model.predict(test[features])*coe
    print('Second 7')
    data0 = simple_fe(data0)
    test = data0[(data0['date'] > '2016-05-01') & (data0['date'] <= '2016-05-08')]
    data0.loc[(data0['date'] > '2016-05-01') & (data0['date'] <= '2016-05-08'), 'demand'] = model.predict(test[features])*coe
    print('Third 7')
    data0 = simple_fe(data0)
    test = data0[(data0['date'] > '2016-05-08') & (data0['date'] <= '2016-05-15')]
    data0.loc[(data0['date'] > '2016-05-08') & (data0['date'] <= '2016-05-15'), 'demand'] = model.predict(test[features])*coe
    print('Forth 7')
    data0 = simple_fe(data0)
    test = data0[(data0['date'] > '2016-05-15') & (data0['date'] <= '2016-05-22')]
    data0.loc[(data0['date'] > '2016-05-15') & (data0['date'] <= '2016-05-22'), 'demand'] = model.predict(test[features])*coe
    return data0[(data0['date'] > '2016-04-24') & (data0['date'] <= '2016-05-22')]
  
  
  
%%time
coe = [1.025, 1.025, 1.025, 1.025, 1.025, 1.025, 1.025, 1.025, 1.025, 1.025]
for i in range(10):
    model_name = 'store_' + str(i) + '.sav'
    model = joblib.load(model_name)
    print('Test on store', i+1)
    if i == 0:
        test = get_test7(data1, 1, model, i)
    else:
        test = pd.concat([test, get_test7(data1, 1, model, i)], axis = 0)
        
        
sample_submission = pd.read_csv('./sample_submission.csv')
predictions = test[['id', 'date', 'demand']]
predictions = pd.pivot(predictions, index='id', columns='date', values='demand').reset_index()
predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
evaluation_rows = [row for row in sample_submission['id'] if 'evaluation' in row]
evaluation = sample_submission[sample_submission['id'].isin(evaluation_rows)]

validation = sample_submission[['id']].merge(predictions, on='id')
final = pd.concat([validation, evaluation])
df = final.copy()
columns_list = df.columns[:]
submission = df[columns_list]
final.to_csv('./submission_acc2.csv', index=False)
