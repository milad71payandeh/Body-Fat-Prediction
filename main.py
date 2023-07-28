import numpy as np
import pandas as pd
import seaborn as sb
import scipy.stats as stt
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.decomposition as dec
import sklearn.linear_model as li
import sklearn.neural_network as nn

#------Section 1: Check the Statistical Information of the Dataset-------
# df = pd.read_csv('bodyfat.csv', sep=',', header=0, encoding='utf-8')
#
# with pd.option_context('display.max_rows', None,'display.max_columns', None):
#     print(df.describe())


#-----------Section 2: Check Correlation between Features---------
# df = pd.read_csv('bodyfat.csv', sep=',', header=0, encoding='utf-8')
#
# correlation = df.corr()
#
# sb.heatmap(correlation, vmin=-1, vmax=+1)
# plt.show()


#-------Section 3: Dimention Reduction with PCA-------
# df = pd.read_csv('bodyfat.csv', sep=',', header=0, encoding='utf-8')
#
# data = df.to_numpy()
#
# Y = df['BodyFat'].to_numpy().reshape((-1, 1))
# df.drop(['BodyFat'], inplace=True, axis=1)
# X = df.to_numpy()
#
# trX, teX, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=0)
#
# scaler = pp.MinMaxScaler()
# trX2 = scaler.fit_transform(trX)
# teX2 = scaler.transform(teX)
#
# pca= dec.PCA(n_components=0.95)
# trX3 = pca.fit_transform(trX2)
# teX3 = pca.transform(teX2)
#
# print(f'{trX3.shape = }')
# print(f'{teX3.shape = }')


#--------Section 4: Check Correlation after PCA------
# df = pd.read_csv('bodyfat.csv', sep=',', header=0, encoding='utf-8')
#
# data = df.to_numpy()
#
# Y = df['BodyFat'].to_numpy().reshape((-1, 1))
# df.drop(['BodyFat'], inplace=True, axis=1)
# X = df.to_numpy()
#
# trX, teX, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=0)
#
# scaler = pp.MinMaxScaler()
# trX2 = scaler.fit_transform(trX)
# teX2 = scaler.transform(teX)
#
# pca= dec.PCA(n_components=0.95)
# trX3 = pca.fit_transform(trX2)
# teX3 = pca.transform(teX2)
#
# C = np.zeros((8, 8))
#
# for i in range(8):
#     for j in range(8):
#         C[i, j] = stt.pearsonr(trX3[:, i], trX3[:, j])[0]
#
#
# sb.heatmap(C, vmin=-1, vmax=+1)
# plt.show()


#-------Section 5: Build Prediction Model with Linear Regression and Evaluate------
# def result(model, trX, teX, trY, teY):
#
#     r2train = model.score(trX, trY)
#     r2test = model.score(teX, teY)
#
#     print(f'Train R2 Score: {round(r2train, 6)}')
#     print(f'Test R2 Score: {round(r2test, 6)}')
#
#     trpred = model.predict(trX)
#     tepred = model.predict(teX)
#
#     a = min([np.min(trpred), np.min(tepred), 0])
#     b = max([np.max(trpred), np.max(tepred), 1])
#     plt.subplot(1, 2, 1)
#     plt.scatter(trY, trpred, s=12, c= 'teal')
#     plt.plot([a, b], [a, b], c='crimson', label='y = x')
#     plt.title(f'Train [R2 = {round(r2train, 4)}]')
#     plt.xlabel('Target Values')
#     plt.ylabel('Predicted Values')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.scatter(teY2, tepred, s=12, c= 'teal')
#     plt.plot([a, b], [a, b], c='crimson', label='y = x')
#     plt.title(f'Test [R2 = {round(r2test, 4)}]')
#     plt.xlabel('Target Values')
#     plt.ylabel('Predicted Values')
#     plt.legend()
#
#     plt.show()
#
#
# df = pd.read_csv('bodyfat.csv', sep=',', header=0, encoding='utf-8')
#
# data = df.to_numpy()
#
# Y = df['BodyFat'].to_numpy().reshape((-1, 1))
# df.drop(['BodyFat'], inplace=True, axis=1)
# X = df.to_numpy()
#
# trX, teX, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=0)
#
# scalerX = pp.MinMaxScaler()
# trX2 = scalerX.fit_transform(trX)
# teX2 = scalerX.transform(teX)
#
# scalerY = pp.MinMaxScaler()
# trY2 = scalerY.fit_transform(trY)
# teY2 = scalerY.transform(teY)
#
# pca = dec.PCA(n_components=0.95)
# trX3 = pca.fit_transform(trX2)
# teX3 = pca.transform(teX2)
#
# Lr = li.LinearRegression()
# Lr.fit(trX3, trY2)
#
# result(Lr, trX3, teX3, trY2, teY2)


#----- Section 6: Build NN and Compare with Regression-----
def result(model, trX, teX, trY, teY):

    r2train = model.score(trX, trY)
    r2test = model.score(teX, teY)

    print(f'Train R2 Score: {round(r2train, 6)}')
    print(f'Test R2 Score: {round(r2test, 6)}')

    trpred = model.predict(trX)
    tepred = model.predict(teX)

    a = min([np.min(trpred), np.min(tepred), 0])
    b = max([np.max(trpred), np.max(tepred), 1])
    plt.subplot(1, 2, 1)
    plt.scatter(trY, trpred, s=12, c= 'teal')
    plt.plot([a, b], [a, b], c='crimson', label='y = x')
    plt.title(f'Train [R2 = {round(r2train, 4)}]')
    plt.xlabel('Target Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(teY2, tepred, s=12, c= 'teal')
    plt.plot([a, b], [a, b], c='crimson', label='y = x')
    plt.title(f'Test [R2 = {round(r2test, 4)}]')
    plt.xlabel('Target Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    plt.show()


df = pd.read_csv('bodyfat.csv', sep=',', header=0, encoding='utf-8')

data = df.to_numpy()

Y = df['BodyFat'].to_numpy().reshape((-1, 1))
df.drop(['BodyFat'], inplace=True, axis=1)
X = df.to_numpy()

trX, teX, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=0)

scalerX = pp.MinMaxScaler()
trX2 = scalerX.fit_transform(trX)
teX2 = scalerX.transform(teX)

scalerY = pp.MinMaxScaler()
trY2 = scalerY.fit_transform(trY)
teY2 = scalerY.transform(teY)

pca = dec.PCA(n_components=0.95)
trX3 = pca.fit_transform(trX2)
teX3 = pca.transform(teX2)

Lr = li.LinearRegression()
Lr.fit(trX3, trY2)

mlp = nn.MLPRegressor(hidden_layer_sizes=(30, 40), activation='relu', random_state=0)
mlp.fit(trX3, trY2)

result(Lr, trX3, teX3, trY2, teY2)
result(mlp, trX3, teX3, trY2, teY2)

