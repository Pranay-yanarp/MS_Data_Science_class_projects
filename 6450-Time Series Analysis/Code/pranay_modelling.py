import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import statsmodels.api as sm
import math
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
import TS_Helper as helper
import statsmodels.tsa.holtwinters as ets
from sklearn.model_selection import train_test_split
from numpy import linalg as LA

np.random.seed(100)

df = pd.read_csv('city_hour.csv')

city1 = df['City'].unique()

for i in city1:
    z=df[df['City']==i]
    print(f'{i} = {len(z)}')

df = df[df['City']=='Delhi']

#==================================== Data Cleaning =================================

print(df.isnull().sum())

df.drop(['Xylene'],axis=1,inplace=True)

cols = df.columns[2:14]
for i in cols:
    df.loc[:,i]=df[i].fillna(np.mean(df[i]))

df.loc[:,'AQI_Bucket'] = df['AQI_Bucket'].fillna(df['AQI_Bucket'].mode()[0])


print(df.isnull().sum())

df = df.reset_index()

start_date = '2015-01-01 01:00:00'
date = pd.date_range(start=start_date, periods=len(df), freq='H')
df['time_new'] = date

y = df['AQI']
time= df['time_new']

# ==================================== AQI vs time plot =================================

plt.plot(time,y)
plt.title('Plot of AQI vs time')
plt.xlabel('Time')
# plt.xticks(rotation=90)
plt.ylabel('AQI')
plt.show()

#================================ Auto Correlation plot =================================

lags=50

# acf3 = helper.auto_correlation_cal(y, lags)

acf3 = acf(y, nlags=lags)
plt.stem(np.arange(-lags, lags+1 ), np.hstack(((acf3[::-1])[:-1],acf3)), linefmt='grey', markerfmt='o')
m = 1.96 / np.sqrt(100)
plt.axhspan(-m, m, alpha=.2, color='blue')
plt.title("ACF Plot of 'AQI'")
plt.xlabel("Lags")
plt.ylabel("ACF of 'AQI'")
plt.grid()
plt.legend(["ACF"], loc='upper right')
plt.show()

#================================ ACF/PACF plot =================================

helper.ACF_PACF_Plot(y,50,'AQI')

#================================ Correlation plot =================================

plt.figure(figsize=(10,10))
sns.heatmap((df.iloc[:,2:15]).corr(),annot=True)
plt.tight_layout()
plt.show()

#================================ Stationarity Check =================================

# ADF plot
helper.ADF_Cal(y)

# KPSS plot
helper.kpss_test(y)

# rolling average
rol_m = helper.rolling_mean(y)

plt.plot(rol_m, label='rolling mean')
plt.title("rolling mean vs time")
plt.xlabel("time")
plt.ylabel("rolling mean")
plt.legend()
plt.grid()
plt.show()

# rolling variance
rol_v = helper.rolling_variance(y)

plt.plot(rol_v, label='rolling variance')
plt.title("rolling variance vs time")
plt.xlabel("time")
plt.ylabel("rolling variance")
plt.legend()
plt.grid()
plt.show()

# ================== 1st order differencing ======================
# y1 = helper.diff1(y1)
y1 = y.diff()
lags=50

# acf3 = helper.auto_correlation_cal(y1, lags)

acf3 = acf(y1[1:], nlags=lags)
plt.stem(np.arange(-lags, lags+1 ), np.hstack(((acf3[::-1])[:-1],acf3)), linefmt='grey', markerfmt='o')
m = 1.96 / np.sqrt(100)
plt.axhspan(-m, m, alpha=.2, color='blue')
plt.title("ACF Plot of 'AQI 1st order'")
plt.xlabel("Lags")
plt.ylabel("ACF of 'AQI' 1st order")
plt.grid()
plt.legend(["ACF"], loc='upper right')
plt.show()


# rolling average of 1st order differencing
rol_m1 = helper.rolling_mean(y1[1:])

plt.plot(rol_m1, label='rolling mean')
plt.title("rolling mean vs time - 1st order differencing")
plt.xlabel("time")
plt.ylabel("rolling mean")
plt.legend()
plt.grid()
plt.show()

# rolling variance of 1st order differencing
rol_v1 = helper.rolling_variance(y1[1:])

plt.plot(rol_v1, label='rolling variance')
plt.title("rolling variance vs time - 1st order differencing")
plt.xlabel("time")
plt.ylabel("rolling variance")
plt.legend()
plt.grid()
plt.show()

# ADF test of 1st order differencing
helper.ADF_Cal(y1[1:])

# KPSS test of 1st order differencing
helper.kpss_test(y1[1:])

# ============================== Time Series Decomposition ======================
from statsmodels.tsa.seasonal import STL
STL=STL(y, period=12)
res = STL.fit()

st = res.seasonal
t = res.trend
rt = res.resid

Ft=1-(np.var(rt)/np.var(t+rt))
print(f'The strength of trend of the raw data is {Ft}')

Fs=1-(np.var(rt)/np.var(st+rt))
print(f'The strength of seasonality of the raw data is {Fs}')

#============================== Train Test split ============================

yt,yf = train_test_split(y,shuffle=False,test_size=0.2)
yt1,yf1 = train_test_split(y1[1:],shuffle=False,test_size=0.2)
dtrain,dtest = train_test_split(time,shuffle=False,test_size=0.2)


#============================== Holt Winter Method ============================

holtt=ets.ExponentialSmoothing(yt,trend='mul',damped_trend=True,seasonal='mul',seasonal_periods=12).fit()
holtf=holtt.forecast(steps=len(yf))
holtf=pd.DataFrame(holtf).set_index(yf.index)

plt.plot(dtrain,yt, label='Train Data')
plt.plot(dtest,yf, label='Test Data')
plt.plot(dtest,holtf.values.flatten(), label='Holt Winter Forecast')
plt.title('AQI vs time (Holt winter)- prediction plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.show()

# ============================= Feature Selection ================================

x = df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO',
        'SO2', 'O3', 'Benzene', 'Toluene',]]

x = x.iloc[1:,:]
y_f = y1[1:]

X_train, X_test, y_train, y_test = train_test_split(x, y_f, test_size=0.2, random_state=100, shuffle=False)

svd_X = df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO',
            'SO2', 'O3', 'Benzene', 'Toluene', 'AQI',]]

H = np.matmul(svd_X.T, svd_X)
print(f'Shape of H is {H.shape}')
s, d, v = np.linalg.svd(H)
print(f'Singular Values {d}')
print(f'The condition number is {LA.cond(svd_X)}')
print('unknown coefficients :',helper.LSE(X_train, y_train))

print("===================================OLS MODEL=================================")

# X = sm.add_constant(X_train)
model_1 = sm.OLS(y_train, X_train)
results_1 = model_1.fit()

print(results_1.summary())

print("===================================OLS MODEL=================================")
print("=========================== SO2 dropped ==================================")

X_train_new=X_train.drop(['NOx'],axis=1)
results_1=sm.OLS(y_train,X_train_new).fit()
print(results_1.summary())

X_test_new = X_test.drop(['NOx'],axis=1)
y_pred_ols = results_1.predict(X_test_new)

y_pred_ols_inv = helper.inverse_diff(y[len(yt1)+1:].values,np.array(y_pred_ols),1)

# def inverse_diff_forecast(y_last,z_hat,interval=1):
#     y_new = np.zeros(len(z_hat))
#     y_new[0] = y_last
#     for i in range(1,len(z_hat)):
#         y_new[i] = z_hat[i-interval] + y_new[i-interval]
#     # y_new = y_new[1:]
#     return y_new

# y_pred_ols_inv = inverse_diff_forecast(yt.values[-1],np.array(y_pred_ols),1)

plt.plot(dtest, y[len(yt1)+1:].values.flatten(), label='Test Data')
plt.plot(dtest[:-1], y_pred_ols_inv, label='OLS Method Forecast')
plt.title('AQI vs time (OLS) - prediction plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.show()
print("\n")


plt.plot(dtest, y_test.values.flatten(), label='Test Data')
plt.plot(dtest, y_pred_ols.values.flatten(), label='OLS Method Forecast')
plt.title('AQI vs time (OLS) differenced values- prediction plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.show()
print("\n")


res_ols_err = y_test-y_pred_ols

lags=100

acf_ols_err = helper.auto_correlation_cal(res_ols_err, lags)

plt.stem(np.arange(-lags+1, lags ), np.hstack(((acf_ols_err[::-1])[:-1],acf_ols_err)), linefmt='grey', markerfmt='o')
m = 1.96 / np.sqrt(100)
plt.axhspan(-m, m, alpha=.2, color='blue')
plt.title("ACF Plot of residual error (OLS)")
plt.xlabel("Lags")
plt.ylabel("ACF values")
plt.grid()
plt.legend(["ACF"], loc='upper right')
plt.show()


# diagnostic testing

print(f'mean of residual error (OLS) method : {np.mean(res_ols_err)}')

print(f'variance of residual error (OLS) method : {np.var(res_ols_err)}')

# ================================== Base-Models ================================

# ================================= average method ==============================

df1 = pd.DataFrame()
y_hat=[]
for i in range(len(yt)):
    if i==0:
        y_hat.append(np.nan)
    else:
        y_hat.append(round((pd.Series(yt)).head(i).mean(),3))

df1['yt']=yt
df1['average']=y_hat

y_avg_pred = []
for j in yf.index:
    y_avg_pred.append(round(df1['yt'].mean(),3))

y_avg_pred = pd.Series(y_avg_pred,index=yf.index)

# plt.plot(yt, label='Train Data')
# plt.plot(yf, label='Test Data')
# plt.plot(y_avg_pred, label='Average Method Forecast')
plt.plot(dtrain,yt, label='Train Data')
plt.plot(dtest,yf, label='Test Data')
plt.plot(dtest,y_avg_pred, label='Average Method Forecast')
plt.title('AQI vs time(Average) - prediction plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.show()


# ================================= naive method =================================

y_hat1=[]
for i in range(len(df1)):
    if i==0:
        y_hat1.append(np.nan)
    else:
        y_hat1.append(df1.loc[i-1,'yt'])

df1['naive']=y_hat1

y_naive_pred = []
for j in yf.index:
    y_naive_pred.append(round(df1.loc[len(df1)-1,'yt']))

y_naive_pred = pd.Series(y_naive_pred,index=yf.index)

# plt.plot(yt, label='Train Data')
# plt.plot(yf, label='Test Data')
# plt.plot(y_naive_pred, label='Naive Method Forecast')
plt.plot(dtrain,yt, label='Train Data')
plt.plot(dtest,yf, label='Test Data')
plt.plot(dtest,y_naive_pred, label='Naive Method Forecast')
plt.title('AQI vs time(Naive) - prediction plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.show()

#  ================================= drift method  =================================

y_hat2=[]
for i in range(len(df1)):
    if i<=1:
        y_hat2.append(np.nan)
    else:
        y_hat2.append(df1.loc[i-1,'yt']+1*((df1.loc[i-1,'yt']-df1.loc[0,'yt'])/((df1.index[i-1]+1)-1) ))

df1['drift']=y_hat2

y_drift_pred = []
h=1
for i in range(len(yf)):
    y_drift_pred.append(df1.loc[len(df1)-1,'yt']+h*((df1.loc[len(df1)-1,'yt']-df1.loc[0,'yt'])/(len(df1)-1) ))
    h+=1

y_drift_pred = pd.Series(y_drift_pred,index=yf.index)

# plt.plot(yt, label='Train Data')
# plt.plot(yf, label='Test Data')
# plt.plot(y_drift_pred, label='Drift Method Forecast')
plt.plot(dtrain,yt, label='Train Data')
plt.plot(dtest,yf, label='Test Data')
plt.plot(dtest,y_drift_pred, label='Drift Method Forecast')
plt.title('AQI vs time(Drift) - prediction plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.show()

#  ================================= SES method  =================================

y_hat3=[]
alpha=0.5
for i in range(len(df1)):
    if i==0:
        y_hat3.append(np.nan)
    elif i==1:
        y_hat3.append(alpha*df1.loc[i-1,'yt']+(1-alpha)*df1.loc[0,'yt']) # since initial condition is 1st sample
    else:
        y_hat3.append(alpha*df1.loc[i-1,'yt']+(1-alpha)*y_hat3[i-1])

df1['ses']=y_hat3

y_ses_pred = []

for i in range(len(yf)):
    y_ses_pred.append(alpha*df1.loc[len(df1)-1,'yt']+(1-alpha)*y_hat3[-1])

y_ses_pred = pd.Series(y_ses_pred,index=yf.index)

# plt.plot(yt, label='Train Data')
# plt.plot(yf, label='Test Data')
# plt.plot(y_ses_pred, label='SES Method Forecast')
plt.plot(dtrain,yt, label='Train Data')
plt.plot(dtest,yf, label='Test Data')
plt.plot(dtest,y_ses_pred, label='SES Method Forecast')
plt.title('AQI vs time(SES) - prediction plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.show()

#  ================================= ARMA Process  =================================

print('========================= ARMA (1,1) model =================================')

acf_y2 = acf(y1[1:],100)
helper.Cal_GPAC(acf_y2,7,7)

na = 1
nb = 1

model = sm.tsa.ARMA(yt1,(na,nb)).fit(trend='nc',disp=0)
print(model.summary())

for i in range(na):
    print(f'The AR Coefficient : a{i+1} is:, {model.params[i]}')

for i in range(nb):
    print(f'The MA Coefficient : b{i+1} is:, {model.params[i+na]}')

# =========== 1 step prediction ================
def inverse_diff(y20,z_hat,interval=1):
    y_new = np.zeros(len(y20))
    for i in range(1,len(z_hat)):
        y_new[i] = z_hat[i-interval] + y20[i-interval]
    y_new = y_new[1:]
    return y_new

model_hat = model.predict(start=0,end=len(yt1)-1)

# res_arma_error = np.array(yt) - np.array(model_hat)

y_hat = helper.inverse_diff(y[:len(yt1)].values,np.array(model_hat),1)

res_arma_error = y[1:len(yt1)] - y_hat

lags=100

helper.ACF_PACF_Plot(res_arma_error,lags,'residual error (ARMA(1,1))')
# acf_res = helper.auto_correlation_cal(res_arma_error, lags)
acf_res = acf(res_arma_error, nlags= lags)
plt.stem(np.arange(-lags, lags+1 ), np.hstack(((acf_res[::-1])[:-1],acf_res)), linefmt='grey', markerfmt='o')
m = 1.96 / np.sqrt(100)
plt.axhspan(-m, m, alpha=.2, color='blue')
plt.title("ACF Plot of residual error (ARMA(1,1))")
plt.xlabel("Lags")
plt.ylabel("ACF values")
plt.grid()
plt.legend(["ACF"], loc='upper right')
plt.show()


plt.plot(time[:99],y[:99], label = 'train set')
plt.plot(time[:99],y_hat[:99], label = '1-step prediction')
plt.title('AQI vs time(ARMA(1,1)) - prediction plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.tight_layout()
plt.show()

# diagnostic testing
from scipy.stats import chi2

print('confidence intervals of estimated parameters:',model.conf_int())

poles = []
for i in range(na):
    poles.append(-(model.params[i]))

print('zero/cancellation:')
zeros = []
for i in range(nb):
    zeros.append(-(model.params[i+na]))

print(f'zeros : {zeros}')
print(f'poles : {poles}')

Q = len(yt)*np.sum(np.square(acf_res[lags:]))

DOF = lags-na-nb

alfa = 0.01

chi_critical = chi2.ppf(1-alfa, DOF)

print('Chi Squared test results')

if Q<chi_critical:
    print(f'The residuals is white, chi squared value :{Q}')
else:
    print(f'The residual is NOT white, chi squared value :{Q}')

# =========== h step prediction ================
forecast = model.forecast(steps=len(yf1))

y_arma_pred = pd.Series(forecast[0], index=yf1.index)

y_hat_fore = inverse_diff(y[len(yt1):].values,np.array(y_arma_pred),1)

# h step prediction

plt.plot(dtest,y[len(yt1)+1:].values.flatten(), label='Test Data')
plt.plot(dtest,y_hat_fore, label='ARMA Method Forecast')
plt.title('AQI vs time(ARMA(1,1)) - prediction plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.show()

res_arma_forecast = y[len(yt1)+1:] - y_hat_fore

print(f'variance of residual error : {np.var(res_arma_error)}')

print(f'variance of forecast error : {np.var(res_arma_forecast)}')

# ========================== 2nd ARMA model na=4, nb=4 ===========================
print('========================= ARMA (4,4) model =================================')

na = 4
nb = 4

model1 = sm.tsa.ARMA(yt1,(na,nb)).fit(trend='nc',disp=0)
print(model1.summary())

for i in range(na):
    print(f'The AR Coefficient : a{i+1} is:, {model1.params[i]}')

for i in range(nb):
    print(f'The MA Coefficient : b{i+1} is:, {model1.params[i+na]}')


#================= 1 step prediction =====================
model_hat1 = model1.predict(start=0,end=len(yt1)-1)

# res_arma_error = np.array(yt) - np.array(model_hat)

y_hat1 = helper.inverse_diff(y[:len(yt1)].values,np.array(model_hat1),1)

res_arma_error1 = y[1:len(yt1)] - y_hat1

lags=100


helper.ACF_PACF_Plot(res_arma_error1,lags,'ARMA(4,4) residual error')
# acf_res = helper.auto_correlation_cal(res_arma_error, lags)
acf_res = acf(res_arma_error1, nlags= lags)
plt.stem(np.arange(-lags, lags+1 ), np.hstack(((acf_res[::-1])[:-1],acf_res)), linefmt='grey', markerfmt='o')
m = 1.96 / np.sqrt(100)
plt.axhspan(-m, m, alpha=.2, color='blue')
plt.title("ACF Plot of residual error (ARMA(4,4))")
plt.xlabel("Lags")
plt.ylabel("ACF values")
plt.grid()
plt.legend(["ACF"], loc='upper right')
plt.show()


plt.plot(time[:99],y[:99], label = 'train set')
plt.plot(time[:99],y_hat1[:99], label = '1-step prediction')
plt.title('AQI vs time(ARMA(4,4)) - prediction plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.tight_layout()
plt.show()

# diagnostic testing
from scipy.stats import chi2

print('confidence intervals of estimated parameters:',model1.conf_int())

poles = []
for i in range(na):
    poles.append(-(model1.params[i]))

print('zero/cancellation:')
zeros = []
for i in range(nb):
    zeros.append(-(model1.params[i+na]))

print(f'zeros : {zeros}')
print(f'poles : {poles}')

Q = len(yt1)*np.sum(np.square(acf_res[lags:]))

DOF = lags-na-nb

alfa = 0.01

chi_critical = chi2.ppf(1-alfa, DOF)

print('Chi Squared test results')

if Q<chi_critical:
    print(f'The residuals is white, chi squared value :{Q}')
else:
    print(f'The residual is NOT white, chi squared value :{Q}')

# =========== h step prediction ================

forecast1 = model1.forecast(steps=len(yf1))

y_arma_pred1 = pd.Series(forecast1[0], index=yf1.index)

y_hat_fore1 = inverse_diff(y[len(yt1):].values,np.array(y_arma_pred1),1)

# h step prediction

plt.plot(dtest,y[len(yt1)+1:].values.flatten(), label='Test Data')
plt.plot(dtest,y_hat_fore1, label='ARMA Method Forecast')
plt.title('AQI vs time(ARMA(4,4)) - prediction plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.show()

res_arma_forecast1 = y[len(yt1)+1:] - y_hat_fore1

print(f'variance of residual error : {np.var(res_arma_error1)}')

print(f'variance of forecast error : {np.var(res_arma_forecast1)}')
#  ================================= ARIMA Process  =================================

print('========================= ARIMA (1,1,1) model =================================')

na = 1
d=1
nb = 1

# yt,yf=train_test_split(y, shuffle=False, test_size=0.2)

model2 = sm.tsa.ARIMA(endog=yt,order=(na,d,nb)).fit()
print(model2.summary())
for i in range(na):
    print(f'The AR Coefficient : a{i+1} is:, {model2.params[i]}')

for i in range(nb):
    print(f'The MA Coefficient : b{i+1} is:, {model2.params[i+na]}')

print('confidence intervals of estimated parameters:',model2.conf_int())


# ================== 1 step prediction ========================
model_hat2 = model2.predict(start=1,end=len(yt)-1)

y_hat3 = inverse_diff(y[:len(yt)].values,np.array(model_hat2),1)

res_arima_error = y[1:len(yt)] - y_hat3

lags=100

acf_res = acf(res_arima_error, nlags=lags)

plt.stem(np.arange(-lags, lags+1), np.hstack(((acf_res[::-1])[:-1],acf_res)), linefmt='grey', markerfmt='o')
m = 1.96 / np.sqrt(100)
plt.axhspan(-m, m, alpha=.2, color='blue')
plt.title("ACF Plot of residual error(ARIMA)")
plt.xlabel("Lags")
plt.ylabel("ACF values")
plt.grid()
plt.legend(["ACF"], loc='upper right')
plt.show()

plt.plot(time[:99],y[:99], label = 'train set')
plt.plot(time[:99],y_hat3[:99], label = '1-step prediction')
plt.title('AQI vs time(ARIMA) - 1 step prediction plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.tight_layout()
plt.show()

# diagnostic testing
from scipy.stats import chi2

print('confidence intervals of estimated parameters:',model2.conf_int())

Q = len(yt)*np.sum(np.square(acf_res[lags:]))

DOF = lags-na-nb

alfa = 0.01

chi_critical = chi2.ppf(1-alfa, DOF)

print('Chi Squared test results')

if Q<chi_critical:
    print(f'The residuals is white, chi squared value :{Q}')
else:
    print(f'The residual is NOT white, chi squared value :{Q}')

# ================== h step prediction ========================
forecast2 = model2.forecast(steps=len(yf))

y_arima_pred = pd.Series(forecast2[0], index=yf.index)

# plt.plot(dtrain,yt, label='Train Data')
plt.plot(dtest,yf.values.flatten(), label='Test Data')
plt.plot(dtest,y_arima_pred.values.flatten(), label='ARIMA Method Forecast')
plt.title('AQI vs time(ARIMA) - Forecast plot')
plt.xlabel('time')
plt.ylabel('AQI')
plt.legend()
plt.show()

res_arima_forecast = np.array(yf) - forecast[0]

print(f'variance of residual error : {np.var(res_arima_error)}')

print(f'variance of forecast error : {np.var(res_arima_forecast)}')

