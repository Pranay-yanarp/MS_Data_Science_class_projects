import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.stattools import adfuller
# import sys
# sys.path.append('F:\pycharm progs\DS_tools')
#
# import TS_Helper

##=================================================================================
# def rolling_cal(df1):
#     df1cols=df1.columns[1:]
#     n=df1.iloc[:,0].size
#     meanrolls=pd.DataFrame(columns=df1cols)
#     varrolls=pd.DataFrame(columns=df1cols)
#     dummy_mean=[]
#     dummy_var=[]
#     for i in df1cols:
#         for j in range(n):
#             dummy_mean.append(np.mean( df1[i].head(j) ))
#             dummy_var.append(np.var( df1[i].head(j) ))
#         meanrolls[i]=dummy_mean
#         varrolls[i]=dummy_var
#         dummy_mean=[]
#         dummy_var=[]
#
#         plt.figure(figsize=(15,8))
#         plt.plot(df1.iloc[:,0],meanrolls[i],label=f'{i}')
#         plt.legend()
#         plt.grid()
#         plt.xlabel('time')
#         plt.xticks(rotation=90)
#         plt.ylabel(f'{i}')
#         plt.title(f'{i} vs time rolling mean')
#         plt.show()
#
#         plt.figure(figsize=(15,8))
#         plt.plot(df1.iloc[:,0],varrolls[i],label=f'{i}')
#         plt.legend()
#         plt.grid()
#         plt.xlabel('time')
#         plt.xticks(rotation=90)
#         plt.ylabel(f'{i}')
#         plt.title(f'{i} vs time rolling variance')
#         plt.show()
##=================================================================================
# '''
# use below function to plot rolling mean and rolling variances
# arguments needed to be given : time, column, columnname
# '''
def rolling_cal1(time,in1,yax):
    col1=pd.Series(in1)
    dummy_mean=[]
    dummy_var=[]
    for i in range(col1.size):
        dummy_mean.append(np.mean( col1.head(i) ))
        dummy_var.append(np.var( col1.head(i) ))

    plt.figure(figsize=(15,8))
    plt.plot(time,dummy_mean,label=yax)
    plt.legend()
    plt.grid()
    plt.xlabel('time')
    plt.xticks(rotation=90)
    plt.ylabel(f'{yax}')
    plt.title(f'{yax} vs time rolling mean')
    plt.show()

    plt.figure(figsize=(15,8))
    plt.plot(time,dummy_var,label=yax)
    plt.legend()
    plt.grid()
    plt.xlabel('time')
    plt.xticks(rotation=90)
    plt.ylabel(f'{yax}')
    plt.title(f'{yax} vs time rolling variance')
    plt.show()
#=================================================================================
'''
calculates ADF test and displays results given series as input
'''
from statsmodels.tsa.stattools import adfuller
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    return result
#=================================================================================
'''
calculates KPSS test and displays results given series as input
'''
from statsmodels.tsa.stattools import kpss
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
    return kpss_output


#=================================================================================
'''
calculates pearsons correlation coefficient given x,y
'''
def correlation_coefficent_cal(x,y):
    x=np.array(x)
    y=np.array(y)
    num=(np.sum((x-np.mean(x))*(y-np.mean(y))))
    den= np.sqrt(np.sum((x-np.mean(x))**2)) * np.sqrt(np.sum((y-np.mean(y))**2))
    r=num/den
    return r


#=================================================================================
'''
calculates auto correlation given series and number of lags
'''
def auto_correlation_cal(series, lags):
    y = np.array(series).copy()
    y_mean = np.mean(series)
    cor = []
    for lag in np.arange(1, lags + 1):
        num1 = y[lag:] - y_mean
        num2 = y[:-lag] - y_mean
        num = sum(num1 * num2)
        den = sum((y - y_mean) ** 2)
        cor.append(num / den)
    return pd.Series(cor)

#=================================================================================
'''
input: original series, predicted series
output: Mean Square Error 
'''
def MSE(org,pred):
    error=org-pred
    error2=error**2
    return np.mean(error2)

#=================================================================================
'''
input: original series, predicted series
output: Variance of Error 
'''

def var_error(org,pred):
    error=org-pred
    return np.var(error)

#=================================================================================
'''
input: series
output: rolling/moving average 
'''

def rolling_mean(series):
    dummy_mean=[]
    s1 = pd.Series(series)
    for i in range(len(s1)):
        dummy_mean.append(np.mean( s1.head(i) ))
    return dummy_mean

#=================================================================================
'''
input: series
output: rolling/moving variance 
'''

def rolling_variance(series):
    dummy_var=[]
    s1 = pd.Series(series)
    for i in range(len(s1)):
        dummy_var.append(np.var( s1.head(i) ))
    return dummy_var

#=================================================================================
'''
input: series
output: naive method output 
'''

def naive_method(series):
    naive=[]
    for i in range(len(series)):
        if i==0:
            naive.append(np.nan)
        else:
            naive.append(series[i-1])
    return naive

#=================================================================================
'''
input: series
output: drift method output 
'''

def drift_onestep(series):
    drift=[]
    h=1
    for i in range(len(series)):
        if i<=1:
            drift.append(np.nan)
        else:
            drift.append(series[i-1]+h*((series[i-1]-series[0])/(i-1)))
    return drift

#=================================================================================
'''
input: series, number of lags
output: Qvalue 
'''

def Qvalue(series,lags):
    r=auto_correlation_cal(series,lags)
    rk=r**2
    return len(series)*(np.sum(rk))


#=================================================================================
'''
input: x = independent variables, y = dependent variable
output: beta = result of noraml equation
'''
def LSE(x,y):
    a1=np.linalg.inv(np.dot(x.T,x))
    return np.dot(a1,np.dot(x.T,y))

#=================================================================================
'''
input: e = error,T = number of observations,k=number of features
output: sigma-e = estimated variance 
'''
def estimated_variance(e,T,k):
    import math
    e2=e**2
    e2_sum=np.sum(e2)
    return (math.sqrt(e2_sum/(T-k-1)))

#=================================================================================
'''
input: series, m, MA(is m is even)
output: moving average of m MA(if m is odd) or MAxm MA (if m is even) 
'''
def trend_cycle(arr1,m=-1,MA=-1):
    if m==-1:
        m,MA=take_input()
    if m in [1,2]:
        print('m=1,2 will not be accepted')
        m=int(input('Enter new order of moving average:'))

    k=int((m-1)/2)
    mean1=[]
    mean2=[]
    if m%2==0: # if m is even
        if MA%2!=0: # if MA is not even
            print('MA not even, will not be accepted')
            MA=int(input('Enter the folding order:'))

        for j in range(k):
            mean1.append(np.nan)

        for i in range(k,len(arr1)-k-1):
            mean1.append( np.mean(arr1[i-k:i+k+1]) )

        for j in range(k+1):
            mean1.append(np.nan)

        mean1=np.array(mean1)

        mean2.append(np.nan)
        for i in range(len(mean1)-1):
            mean2.append(np.mean(mean1[i:i+MA]))

        return np.array(mean2)

    else: # if m is odd
        for j in range(k):
            mean1.append(np.nan)
        for i in range(k,len(arr1)-k):
            mean1.append( np.mean(arr1[i-k:i+k]) )
        for j in range(k):
            mean1.append(np.nan)
        return np.array(mean1)


def take_input():
    m=int(input('Enter order of moving average:'))
    if m in [1,2]:
        print('m=1,2 will not be accepted')
        m=int(input('Enter new order of moving average:'))

    if m%2==0:
        b=int(input('Enter the folding order:'))
        if b%2==0:
            MA=b
        else:
            print('MA not even, will not be accepted')
            MA=int(input('Enter the new folding order:'))
        return m,MA
    else:
        return m,1

#=================================================================================
'''
input: y, lags, title
output: plots of ACF and PACF of y 
'''
def ACF_PACF_Plot(y,lags,ti):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title(f'ACF/PACF of the {ti}')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()

#=================================================================================
'''
input: series, order
output: difference series of given order
'''

def diff1(d1,int1):
    d1=np.array(d1)
    diff = []
    for i in range(int1,len(d1)):
        diff.append((d1[i] - d1[i-int1]))
    return diff

#=================================================================================
'''
input: acf series, j-length, k-length
output: GPAC table
'''
def Cal_GPAC(acf, len_j, len_k):

    len_k = len_k + 1
    gpac = np.empty(shape=(len_j, len_k))

    for k in range(1, len_k):
        num = np.empty(shape=(k, k))
        denom = np.empty(shape=(k, k))
        for j in range(0, len_j):
            for row in range(0, k):
                for column in range(0, k):
                    if column < k - 1:
                        num[row][column] = acf[np.abs(j+(row-column))]
                        denom[row][column] = acf[np.abs(j+(row-column))]
                    else:
                        num[row][column] = acf[np.abs(j+row+1)]
                        denom[row][column] = acf[np.abs(j+(row-column))]

            num_determinant = round(np.linalg.det(num), 6)
            denom_determinant = round(np.linalg.det(denom), 6)

            if denom_determinant == 0.0:
                gpac[j][k] = np.inf
            else:
                gpac[j][k] = round((num_determinant/denom_determinant), 3)

    gpac = pd.DataFrame(gpac[:, 1:])
    gpac.columns = [i for i in range(1, len_k)]

    sns.heatmap(gpac, annot=True)
    plt.title('GPAC Table')
    plt.show()

#=================================================================================
'''
input: acf series, j-length, k-length
output: GPAC table
'''
def Cal_GPAC_dash(acf, len_j, len_k):

    len_k = len_k + 1
    gpac = np.empty(shape=(len_j, len_k))

    for k in range(1, len_k):
        num = np.empty(shape=(k, k))
        denom = np.empty(shape=(k, k))
        for j in range(0, len_j):
            for row in range(0, k):
                for column in range(0, k):
                    if column < k - 1:
                        num[row][column] = acf[np.abs(j+(row-column))]
                        denom[row][column] = acf[np.abs(j+(row-column))]
                    else:
                        num[row][column] = acf[np.abs(j+row+1)]
                        denom[row][column] = acf[np.abs(j+(row-column))]

            num_determinant = round(np.linalg.det(num), 6)
            denom_determinant = round(np.linalg.det(denom), 6)

            if denom_determinant == 0.0:
                gpac[j][k] = np.inf
            else:
                gpac[j][k] = round((num_determinant/denom_determinant), 3)

    gpac = pd.DataFrame(gpac[:, 1:])
    gpac.columns = [i for i in range(1, len_k)]

    return gpac

#=================================================================================
'''
input: y, z_hat, interval
output: h_hat
'''
def inverse_diff(y20,z_hat,interval=1):
    y_new = np.zeros(len(y20))
    for i in range(1,len(z_hat)):
        y_new[i] = z_hat[i-interval] + y20[i-interval]
    y_new = y_new[1:]
    return y_new

#=================================================================================
#=================================================================================
'''
input: y, z_hat, interval
output: h_hat
'''
def inverse_diff_forecast(y_last,z_hat,interval=1):
    y_new = np.zeros(len(z_hat))
    y_new[0] = y_last
    for i in range(1,len(z_hat)):
        y_new[i] = z_hat[i-interval] + y_new[i-interval]
    # y_new = y_new[1:]
    return y_new

#=================================================================================