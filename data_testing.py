import copy

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import metrics
from collections import defaultdict
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import os
from tensorflow import keras

# print()
print("Preparing to read data for testing...")
data = pd.read_csv('data/test_car_auction_price_data.csv', low_memory=False)
# print(data.isnull().sum())
print(data.dtypes)
# print(data['odo'])
# print(data['auction_venue'])
print("Preparing to rename and drop data columns...")
data.rename({'site_name': 'site', 'produced_on': 'year_car', 'displacement': 'engine',
             'auction_venue': 'auction_site'}, axis=1,
            inplace=True)

data.drop(columns=['Unnamed: 0', 'id', 'car_name_raw', 'produced_on_raw', 'brand_raw', 'model_raw',
                   'grade_raw', 'auction_venue_raw', 'open_on_raw', 'open_on', "exterior_score_raw", "exterior_score",
                   "interior_score_raw", "interior_score", "lot_no_raw", "odo_raw", "color_raw",
                   "displacement_raw", "gear_raw", "seats_raw", "seats", "chassis_number_raw",
                   "chassis_number", "fuel_raw", "fuel", "start_price_raw", "current_price_raw",
                   "current_price", "final_price_raw", "equipment_raw", "equipment", "url",
                   "created_at", "updated_at", "score_raw", "predicted_price", "market_price", 'gear'],
          axis=1, inplace=True)

print("Preparing to rearrange column index...")
# Rearrange column index
data = data.loc[:,
       ['site', 'car_name', 'brand', 'grade', 'model', 'color', 'year_car', 'engine', 'odo', 'score', 'start_price',
        'auction_site', 'final_price']]

# print(data)
# print(data.dtypes)
print("Preparing to encode data...")
df_test = copy.deepcopy(data)

for i in df_test.columns:
    if i not in df_test:
        df_test[i] = df_test[i].map(str)
# df_train.drop(columns=cols, inplace=True)

cols = np.array(df_test.columns[df_test.dtypes != object])
d = defaultdict(LabelEncoder)

df_test = df_test.apply(lambda x: d[x.name].fit_transform(x))

df_test[np.delete(cols, len(cols) - 1)] = data[np.delete(cols, len(cols) - 1)]
# print(df_test)
# for column in df_test.columns:
#     # Check if the column contains string values
#     if df_test[column].dtype == 'object':
#         print("Preparing to label encoder...")
#         # Create a LabelEncoder object
#         label_encoder = LabelEncoder()
#         print(f"label_encoder: {label_encoder}")
#         # Fit the LabelEncoder to the column and transform the column
#         print("Preparing to fit the label encoder to the column and transform the column...")
#         df_test[column] = label_encoder.fit_transform(df_test[column])
#         # data = data.apply(lambda x: d[x.name].fit_transform(x))
#         print("Done fit the label encoder.")
#         print(f"data[column]: {df_test[column]}")

feature1 = ['site', 'car_name', 'brand', 'grade', 'model', 'color', 'year_car', 'engine', 'odo', 'score', 'start_price',
            'auction_site']
df_test = df_test.dropna()
X0 = df_test[feature1]

Acc = pd.DataFrame(index=None,
                   columns=['model', 'RMSE', 'MSE', 'MAE', 'MAPE', 'R^2 score', 'Adjust R2 score'])
# print(data)
y0 = df_test['final_price'].values
lab_enc = preprocessing.LabelEncoder()
y_true = lab_enc.fit_transform(y0)
# print(y_true)
print("Preparing to load model...")
model = joblib.load("./model/BayesianRidge_v4.pkl")
# i = 0
# model = keras.models.load_model('./model-667-0.000-145473.797.h5')
y_pred = model.predict(X0)
score = model.score(X0, y_true)
MAE = metrics.mean_absolute_error(y_true, y_pred)
MAPE = mean_absolute_percentage_error(y_true, y_pred)
MSE = mean_squared_error(y_true, y_pred)
RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"R^2: {score}")
# print(f"Loss: {score[0]}")
print(f"MAE: {MAE}")
print(f"MAPE: {MAPE}")
print(f"MSE:", MSE)
print(f"RMSE:", RMSE)
# for root, dirs, files in os.walk('E:/Paracel/CarAuctionPrediction/'):
#     for name in files:
#         if name.endswith(".pkl"):
#             print(f"Preparing to load model {files[i]}...")
#             try:
#                 model = joblib.load(filename=f"model/{name}")
#                 pred = model.predict(X0)
#                 #print(pred)
#                 print(f"Preparing to calculate R2 score of model {files[i]}...")
#                 score = model.score(X0, y_true)
#                 print(f"R^2 of {files[i]}: {score}")
#                 print(f"Preparing to calculate MAE score of model {files[i]}...")
#                 MAE = metrics.mean_absolute_error(y_true, pred)
#                 print(f"MAE of {files[i]}: {MAE}")
#                 print(f"Preparing to calculate MAPE score of model {files[i]}...")
#                 MAPE = mean_absolute_percentage_error(y_true, pred)
#                 print(f"MAPE of {files[i]}: {MAPE}")
#                 print(f"Preparing to calculate MSE score of model {files[i]}...")
#                 MSE = mean_squared_error(y_true, pred)
#                 print(f"MSE of {files[i]}:", MSE)
#                 print(f"Preparing to calculate RMSE score of model {files[i]}...")
#                 RMSE = np.sqrt(mean_squared_error(y_true, pred))
#                 print(f"RMSE of {files[i]}:", RMSE)
#                 n = 100000
#                 k = 10
#                 print(f"Preparing to calculate Adjust R2 score of model {files[i]}...")
#                 adj_r2_score = 1 - ((1 - score) * (n - 1) / (n - k - 1))
#                 #print("RMSE:", np.sqrt(mean_squared_error(y_true, pred)))
#                 val_acc = pd.Series({'model': name, 'RMSE': RMSE, 'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE, 'R^2 score': score,
#                                      'Adjust R2 score': adj_r2_score})
#                 Acc = pd.concat([Acc, val_acc.to_frame().T], ignore_index=True)
#                 i += 1
#             except Exception as e:
#                 print("ERROR : "+str(e))
#
# print(Acc)

# for root, dirs, files in os.walk(r'E:\Paracel\CarAuctionPrediction\model'):
#     for name in files:
#         model = joblib.load(name)
#         n = 100000
#         k = 10
#         adj_r2_score = 1 - ((1-score)*(n-1)/(n-k-1))
#         adj_r2 = pd.Series({'Adjust R2 score': adj_r2_score})
#         Acc = pd.concat([Acc, adj_r2.to_frame().T], ignore_index=True)
# print(f"Adjust R2 Score: {adj_r2_score}")
# # def Definedata():
# #     # define dataset
# #     X = data[feature1]
# #     y0 = data['final_price'].values
# #     lab_enc = preprocessing.LabelEncoder()
# #     y = lab_enc.fit_transform(y0)
# #     return X, y
# # model = model.get_booster().feature_names
# # print(model)
# # X0 = data[feature1].values
# # y_predicted = model.predict(X0)
# # print(f"y_predicted: {y_predicted}")
# # X, y = Definedata()
#
# # y_predicted = model.predict(X0)
# # print(y_predicted)
# # y0 = data['final_price'].values
# # lab_enc = preprocessing.LabelEncoder()
# # y = lab_enc.fit_transform(y0)
# # print(y)
# # MSE = metrics.mean_squared_error(y_true, y_predicted)
# # print(MSE)
# # scores = cross_val_score(estimator=model, X=X0, y=y_true, scoring='neg_mean_squared_error')
# # print(scores.mean())
# # # print(RMSE)
# # score = model.score(X0, y)
# # print(score)
# # new_data = pd.DataFrame(
# #     {'site': data['site'], 'brand': data['brand'], 'car_name': data['car_name'], 'color': data['color_clean'],
# #      'score': data['score_clean'], 'year': data['year_car_clean'],
# #      'engine': data['engine_clean'], 'final_price': y_true, 'predicted price': pred, 'difference': y_true - pred})
# # print(new_data.head())
# # new_data.to_csv('results/res_random_forest_v2.xlsx', index=True)
