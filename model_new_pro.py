import copy
import logging
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib import style
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from lightgbm import CVBooster, DaskLGBMRegressor, Sequence
import joblib
from sklearn.neighbors import RadiusNeighborsRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

logging.basicConfig(filename="new_log.txt",
                    format='%(asctime)s %(message)s',
                    filemode='w')
#
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

iterations = 10
print("Preparing to read, split and analyze data...")
# logger.debug("Preparing to read, split and analyze data...")

dataset = 'data/training_auction_price_clean_data.csv'

df = pd.read_csv(dataset, low_memory=False)
df = df.drop_duplicates()
df = df.dropna()
print(df.duplicated().sum())
# df = df.drop(columns=['Unnamed: 0', 'index'])
train, test = train_test_split(df, test_size=0.3, random_state=42)

df_train = copy.deepcopy(train)
df_test = copy.deepcopy(test)
# print(df_test['final_price'])
cols = np.array(train.columns[train.dtypes != object])

for i in df_train.columns:
    if i not in cols:
        df_train[i] = df_train[i].map(str)
        df_test[i] = df_test[i].map(str)
df_train.drop(columns=cols, inplace=True)

# build dictionary function
cols = np.array(train.columns[train.dtypes != object])
d = defaultdict(LabelEncoder)

# only for categorical columns apply dictionary by calling fit_transform
df_train = df_train.apply(lambda x: d[x.name].fit_transform(x))

df_train[cols] = train[cols]
# print(df_train)

df_test = df_test.apply(lambda x: d[x.name].fit_transform(x))

df_test[np.delete(cols, len(cols) - 1)] = test[np.delete(cols, len(cols) - 1)]
print(df_train)

ftrain = ['site', 'car_name', 'brand', 'grade', 'model', 'color', 'year_car', 'engine', 'odo', 'score', 'start_price',
          'auction_site']

print("Split and analyze data successfully")


# logger.debug("Split and analyze data successfully")


def Definedata():
    # define dataset
    X = df_train[ftrain]

    y0 = df_train['final_price'].values
    lab_enc = preprocessing.LabelEncoder()
    y = lab_enc.fit_transform(y0)
    return X, y


print(df_train)
print("Preparing to show graph about data...")
# logger.debug("Preparing to show graph about data...")
style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize=(12, 7))

# Plotting heatmap.
# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(df_train.corr(), dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df_train.corr(), cmap=sns.diverging_palette(20, 220, n=200), annot=True, mask=mask, center=0, )
plt.title("Heatmap of all the Features of Train data set", fontsize=25)
plt.show()
print("Showing data graph successfully")
# logger.debug("Showing data graph successfully")

print("logger done")
# logger.debug("logger done")

Acc = pd.DataFrame(index=None,
                   columns=['model', 'RMSE', 'MSE', 'MAE', 'MAPE', 'Accuracy on Traing set', 'Accuracy on Testing set',
                            'R2',
                            ])
# logger.info("This is just an information for you")
# logger.debug("Acc initialized done")
print("Acc initialized done")
X, y = Definedata()
# X = X.reshape(X.shape[1:])
# X = X.transpose()
# logger.debug("Definedata() done")
print("Definedata() done")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"y_train:{y_train}")
print(f"y_test:{y_test}")
# logger.debug("train_test_split with X_train, X_test, y_train, y_test done")
print("train_test_split with X_train, X_test, y_train, y_test done")

regressors = [
    ['RadiusNeighborsRegressor', RadiusNeighborsRegressor()],
    ['KNeighborsTransformer', KNeighborsTransformer()],
    ['GradientBoostingRegressor', GradientBoostingRegressor()],
    ['RandomForestRegressor', RandomForestRegressor()],
    ['BaggingRegressor', BaggingRegressor()],
    ['ExtraTreesRegressor', ExtraTreesRegressor()],
    ['LinearRegression', LinearRegression()],
    ['LogisticRegression', LogisticRegression()],
    ['LogisticRegressionCV', LogisticRegressionCV()],
    ['DaskLGBMRegressor', DaskLGBMRegressor()],
    ['XGBoost', XGBRegressor()],
['Catboost', CatBoostRegressor()],
    ['DecisionTreeRegressor', DecisionTreeRegressor()],
    ['MLPRegressor', MLPRegressor()],
    ['LGBMRegressor', LGBMRegressor()],
    ['Ridge', Ridge()],
    ['AdaBoostRegressor', AdaBoostRegressor()]

]

# logger.debug("regressors initialized done")
print("regressors initialized done")
i = 4
# model training
for mod in regressors:
    name = mod[0]
    print(name)
    # logger.debug(name)
    model = mod[1]
    print(model)
    model.fit(X_train, y_train)
    print("Model fits successfully")

    joblib.dump(model, f'model/{name}_v{i}.pkl')
    print(f"Model {name} saved successfully")

    y_pred = model.predict(X_test)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print("RMSE calculated successfully")
    print(f"RMSE of {name} is {RMSE}")

    ATrS = model.score(X_train, y_train)
    # ATrS = r2_score(y_train, y_pred)
    print("ATrS calculated successfully")
    print(f"ATrS of {name} is {ATrS}")

    ATeS = model.score(X_test, y_test)
    print("ATeS calculated successfully")
    print(f"ATeS of {name} is {ATeS}")

    R2 = r2_score(y_test, y_pred)
    print("R2 calculated successfully")
    print(f"R2 of {name} is {R2}")

    MSE = metrics.mean_squared_error(y_test, y_pred)
    print("MSE calculated successfully")
    print(f"MSE of {name} is {MSE}")

    MAE = metrics.mean_absolute_error(y_test, y_pred)
    print("MAE calculated successfully")
    print(f"MAE of {name} is {MAE}")

    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    print("MAPE calculated successfully")
    print(f"MAPE of {name} is {MAPE}")

    val_acc = pd.Series(
        {'model': name, 'RMSE': RMSE, 'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE, 'Accuracy on Training set': ATrS,
         'Accuracy on Testing set': ATeS, 'R2': R2})
    Acc = pd.concat([Acc, val_acc.to_frame().T], ignore_index=True)
print(Acc)

# check loss and accuracy of regression models in each iteration
for i in range(1, iterations):
    model.fit(X_train[:i], y_train[:i])
    logger.debug(f"model: {model}")
    print(f"model: {model}")
    y_pred = model.predict(X_test[:i])
    print(f"y_pred: {y_pred}")
    logger.debug(f"y_pred: {y_pred}")

    RMSE = np.sqrt(metrics.mean_squared_error(y_test[:i], y_pred))
    ATrS = model.score(X_train, y_train)
    ATeS = model.score(X_test, y_test)
    print(f"RMSE at iteration {i} in {name} is: {RMSE}")
    logger.info(f"RMSE at iteration {i} is: {RMSE}")
    print(f"Accuracy of Training at iteration {i} in {name} in RMSE is: {ATrS}")
    logger.info(f"Accuracy of Training at iteration {i} in {name} in RMSE is: {ATrS}")
    print(f"Accuracy of Testing at iteration {i} in {name} in RMSE is: {ATeS}")
    logger.info(f"Accuracy of Testing at iteration {i} in {name} in RMSE is: {ATeS}")

    MSE = metrics.mean_squared_error(y_test[:i], y_pred)
    print(f"MSE at iteration {i} in {name} is: {MSE}")
    logger.info(f"MSE at iteration {i} is: {MSE}")
    print(f"Accuracy of Training at iteration {i} in {name} in MSE is: {ATrS}")
    logger.info(f"Accuracy of Training at iteration {i} in {name} in MSE is: {ATrS}")
    print(f"Accuracy of Testing at iteration {i} in {name} in MSE is: {ATeS}")
    logger.info(f"Accuracy of Testing at iteration {i} in {name} in MSE is: {ATeS}")

    MAE = metrics.mean_absolute_error(y_test[:i], y_pred)
    print(f"MAE at iteration {i} in {name} is: {MAE}")
    logger.info(f"MAE at iteration {i} is: {MAE}")
    ATrS = model.score(X_train, y_train)
    ATeS = model.score(X_test, y_test)
    print(f"Accuracy of Training at iteration {i} in {name} in MAE is: {ATrS}")
    logger.info(f"Accuracy of Training at iteration {i} in {name} in MAE is: {ATrS}")
    print(f"Accuracy of Testing at iteration {i} in {name} in MAE is: {ATeS}")
    logger.info(f"Accuracy of Testing at iteration {i} in {name} in MAE is: {ATeS}")

logger.info("RMSE calculated successfully")
print("RMSE calculated successfully")
logger.info("MSE calculated successfully")
print("MSE calculated successfully")
logger.info("MAE calculated successfully")
print("MAE calculated successfully")

logger.info("ATrS calculated successfully")
print("ATrS calculated successfully")
logger.info("ATeS calculated successfully")
print("ATeS calculated successfully")
joblib.dump(model, f"{name}.pkl")
print("Model saved successfully")
logger.info("Model saved successfully")


logger.info("Acc calculated successfully")
print("Acc calculated successfully")

# logger.debug("Exit from loop successfully")
print("Exit from loop successfully")

print(f"\t\tError Table in {name}")
print(f'MAE      : {MAE}')
print(f'MSE      : {MSE}')
print(f'RMSE : {RMSE}')
print(f'Accuracy on Traing set   : {ATrS}')
print(f'Accuracy on Testing set  : {ATeS}')

logger.info(f"\t\tError Table in {name}")
logger.info(f'MAE      : {MAE}')
logger.info(f'MSE      : {MSE}')
logger.info(f'RMSE : {RMSE}')
logger.info(f'Accuracy on Traing set   : {ATrS}')
logger.info(f'Accuracy on Testing set  : {ATeS}')

print(Acc)
logger.debug(Acc)

print("OK1")
model = RandomForestRegressor(n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                              max_depth=25)
print("OK2")
feature1 = ['site', 'brand', 'car_name', 'color_clean', 'raw_record', 'odo_clean',
            'score_clean', 'year_car_clean', 'engine_clean']
print("OK3")

X0 = df_test[feature1]
# print(X0['site'])
print("OK4")
X, y = Definedata()
print("OK5")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("OK6")
model.fit(X_train, y_train)
print("OK7")
y_predicted = model.predict(X0)
# print(f"y_predicted: {y_predicted}")
print("OK8")

submission = pd.DataFrame({'id': test.index, 'site': df['site'], 'brand': df['brand'], 'car_name': df['car_name'],
                           'color_clean': df['color_clean'], 'raw_record': df['raw_record'],
                           'odo_clean': df['odo_clean'], 'score_clean': df['score_clean'],
                           'year_car_clean': df['year_car_clean'], 'engine_clean': df['engine_clean'],
                           'predicted_price': y_predicted, 'real_final_price': df['final_price']
                           })
print("OK9")
print(submission.head(10))

# Convert DataFrame to a csv file that can be uploaded
# This is saved in the same directory as your notebook
filename = 'random_forest_prediction_report_v3.xlsx'
print("OK10")
submission.to_excel(filename, index=True)
print('Saved file: ' + filename)
