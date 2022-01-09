import dash
import dash_bootstrap_components as dbc
import plotly.figure_factory as ff
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import math
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from statsmodels.graphics.gofplots import qqplot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.stattools import adfuller
import plotly
import plotly.figure_factory as ff
import TS_Helper as helper
from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
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


y1 = y.diff()

#============================== Train Test split ============================

yt,yf = train_test_split(y,shuffle=False,test_size=0.2)
yt1,yf1 = train_test_split(y1[1:],shuffle=False,test_size=0.2)
dtrain,dtest = train_test_split(time,shuffle=False,test_size=0.2)

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
singular1 = d
cond_number = LA.cond(svd_X)
unknown_coef = helper.LSE(X_train, y_train)

# print("===================================OLS MODEL=================================")

# X = sm.add_constant(X_train)
model_1 = sm.OLS(y_train, X_train)
results_1 = model_1.fit()

print(results_1.summary())

print("=========================== SO2 dropped ==================================")

X_train_new=X_train.drop(['NOx'],axis=1)
results_1=sm.OLS(y_train,X_train_new).fit()
print(results_1.summary())

X_test_new = X_test.drop(['NOx'],axis=1)
y_pred_ols = results_1.predict(X_test_new)


#======================= Dash Board ================================================


from io import BytesIO
import base64
def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Selection Menu", className="display-4"),
        html.Hr(),
        html.P(
            "Time series analysis", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Time Series Plot", href="/line1", active="exact"),
                dbc.NavLink("Histogram Plot", href="/hist1", active="exact"),
                dbc.NavLink("ACF/PACF Plot", href="/acf1", active="exact"),
                dbc.NavLink("GPAC Table", href="/gpac1", active="exact"),
                dbc.NavLink("ADF/KPSS test", href="/adf1", active="exact"),
                dbc.NavLink("OLS Regression", href="/ols1", active="exact"),
                dbc.NavLink("Base Models Plot", href="/base1", active="exact"),
                dbc.NavLink("ARMA/ARIMA models", href="/main1", active="exact"),

            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

#===================== Time series data ================================
line_layout = html.Div([
    html.H1('Time Series Plot', style={'textAlign':'center'}),
    html.Br(),
    html.Br(),
    dcc.Graph(id = 'my-graph1'),
    html.P('Time Range Slider'),
    dcc.Slider(id='slider1', min=0, max=48192, value=500,
               marks= {0:'0',500:'500',1000:'1000',2000:'2000',5000:'5000',10000:'10000',20000:'20000',30000:'30000',48192:'48192'}),
    html.Br(),
    html.Br(),

])

@app.callback(
    Output(component_id='my-graph1', component_property='figure'),
    [Input(component_id='slider1',component_property='value'),]
)

def display_color(value_x):
    rang1 = np.arange(0,int(value_x))
    fig1 = px.line(df,x=rang1,y=y[:int(value_x)])
    return fig1

#===================== Histogram Plot ================================
import io
import base64

histogram_layout = html.Div([
    html.H1('Histogram Plot', style={'textAlign':'center'}),
    html.Br(),
    dcc.Graph(id = 'my-graph2'),
    html.Br(),
    html.P('Number of bins'),
    dcc.Slider(id='bins1', min=10, max=100, value=10,
               marks= {10:'10',15:'15',20:'20',30:'30',40:'40',50:'50',60:'60',70:'70',80:'80',90:'90',100:'100'}),
    html.Br(),
    html.P('Please select the data'),
    dcc.RadioItems(
        id='data11',
        options=[
            {'label':'Raw data  ','value':'0'},
            {'label':'1st order differencing  ','value':'1'},
        ],value='0'
    ),
])

@app.callback(
    Output(component_id='my-graph2', component_property='figure'),
    [Input(component_id='bins1',component_property='value'),
     Input(component_id='data11',component_property='value')]
)

def display_color1(bin1,dats):
    if int(dats)==0:
        give=y
    elif int(dats)==1:
        give=y1[1:]
    fig1 = px.histogram(x=give, nbins=int(bin1))
    return fig1


#===================== ACF/PACF Plot ================================

acf_pacf_layout = html.Div([
    html.H1('ACF Plot', style={'textAlign':'center'}),
    html.Br(),
    dcc.Graph(id = 'my-graph3'),
    html.Br(),
    html.H1('PACF Plot', style={'textAlign':'center'}),
    html.Br(),
    dcc.Graph(id = 'my-graph4'),
    html.Br(),
    html.P('Please select the data'),
    dcc.Dropdown(
        id='data12',
        options=[
            {'label':'Raw data','value':'0'},
            {'label':'1st order differencing','value':'1'},
        ],value='0'
    ),
    html.P('Please enter the number of lags'),
    html.Div([
        'Input:',
        dcc.Input(id='InputA',value=50,type='float')
    ]),
    html.Br(),
])

@app.callback(
    [Output(component_id='my-graph3', component_property='figure'),
     Output(component_id='my-graph4', component_property='figure')],
    [Input(component_id='data12',component_property='value'),
     Input(component_id='InputA',component_property='value')]
)


def display_color2(dats,lags1):
    if int(dats)==0:
        give=y
    elif int(dats)==1:
        give=y1[1:]

    acf3 = sm.tsa.stattools.acf(give, nlags=int(lags1))
    y_acf = np.hstack(((acf3[::-1])[:-1],acf3))

    trace = plotly.graph_objs.Scatter(x=np.arange(-int(lags1), int(lags1)+1 ),
                                      y=np.hstack(((acf3[::-1])[:-1],acf3)),
                                      mode='markers')
    z=np.arange(-int(lags1), int(lags1)+1 )
    shapes = list()
    count1 = 0
    for i in z:
        shapes.append({'type': 'line',
                       'xref': 'x',
                       'yref': 'y',
                       'x0': i,
                       'y0': 0,
                       'x1': i,
                       'y1': y_acf[count1]})
        count1 += 1

    layout = plotly.graph_objs.Layout(shapes=shapes)
    fig = plotly.graph_objs.Figure(data=[trace],
                                   layout=layout)


    pacf3 = sm.tsa.stattools.pacf(give, nlags=int(lags1))
    y_pacf = pacf3
    trace1 = plotly.graph_objs.Scatter(x=np.arange(int(lags1)+1),
                                      y=pacf3,
                                      mode='markers')
    z1 = np.arange(int(lags1)+1)
    shapes1 = list()
    count1 = 0
    for i in z1:
        shapes1.append({'type': 'line',
                       'xref': 'x',
                       'yref': 'y',
                       'x0': i,
                       'y0': 0,
                       'x1': i,
                       'y1': y_pacf[count1]})
        count1 += 1

    layout1 = plotly.graph_objs.Layout(shapes=shapes1)
    fig1 = plotly.graph_objs.Figure(data=[trace1],
                                   layout=layout1)

    return fig, fig1

#===================== GPAC Table ================================

gpac_layout = html.Div([
    html.H1('GPAC Table', style={'textAlign':'center'}),
    html.Br(),
    dcc.Graph(id = 'my-graph5'),
    html.Br(),
    html.P('Please select the data'),
    dcc.Dropdown(
        id='data13',
        options=[
            {'label':'Raw data','value':'0'},
            {'label':'1st order differencing','value':'1'},
        ],value='0'
    ),
    html.P('Please enter the j-length value'),
    html.Div([
        'Input:',
        dcc.Input(id='Input1',value=7,type='float')
    ]),
    html.Br(),
    html.P('Please enter the k-length value'),
    html.Div([
        'Input:',
        dcc.Input(id='Input2',value=7,type='float')
    ]),
    html.Br(),
])

@app.callback(
    Output(component_id='my-graph5', component_property='figure'),
    [Input(component_id='data13',component_property='value'),
     Input(component_id='Input1',component_property='value'),
     Input(component_id='Input2',component_property='value')]
)

def display_color1(dats,j1,k1):
    if int(dats)==0:
        give=y
    elif int(dats)==1:
        give=y1[1:]

    acf3 = sm.tsa.stattools.acf(give, nlags=100)
    gpac_df1 = helper.Cal_GPAC_dash(acf3,j1,k1)
    gpac_df2 = pd.DataFrame()
    for i in gpac_df1.columns:
        gpac_df2[i] = gpac_df1[i][len(gpac_df1)::-1]
    # fig1 = px.imshow(gpac_df1)
    # fig1 = plotly.graph_objects.Heatmap(gpac_df1)
    fig1 = ff.create_annotated_heatmap(np.array(gpac_df2))
    return fig1

#===================== ADF/KPSS Test ================================

adf_layout = html.Div([
    html.H1('ADF/KPSS test results', style={'textAlign':'center'}),
    html.Br(),
    html.P('Please select the data'),
    dcc.Dropdown(
        id='data14',
        options=[
            {'label':'Raw data','value':'0'},
            {'label':'1st order differencing','value':'1'},
        ],value='0'
    ),
    html.Br(),
    html.Br(),
    html.H2('ADF results:', style={'textAlign':'center'}),
    html.H3(id='output1a'),
    html.H3(id='output2a'),
    html.H3('Critical Values:'),
    html.H3(id='output3a'),
    html.H3(id='output4a'),
    html.H3(id='output5a'),
    html.H2(id='output6a'),
    html.Br(),
    html.Br(),
    html.H2('KPSS results:', style={'textAlign':'center'}),
    html.H3(id='output7a'),
    html.H3(id='output8a'),
    html.H3(id='output9a'),
    html.H3(id='output10a'),
    html.H3(id='output11a'),
    html.H3(id='output12a'),
    html.H3(id='output13a'),
    html.H2(id='output14a'),
])

@app.callback(
    [Output(component_id='output1a', component_property='children'),
     Output(component_id='output2a', component_property='children'),
     Output(component_id='output3a', component_property='children'),
     Output(component_id='output4a', component_property='children'),
     Output(component_id='output5a', component_property='children'),
     Output(component_id='output6a', component_property='children'),
     Output(component_id='output7a', component_property='children'),
     Output(component_id='output8a', component_property='children'),
     Output(component_id='output9a', component_property='children'),
     Output(component_id='output10a', component_property='children'),
     Output(component_id='output11a', component_property='children'),
     Output(component_id='output12a', component_property='children'),
     Output(component_id='output13a', component_property='children'),
     Output(component_id='output14a', component_property='children'),],
    [Input(component_id='data14',component_property='value'),]
)

def display_color1(dats):
    if int(dats)==0:
        give=y
    elif int(dats)==1:
        give=y1[1:]
    adf_result = helper.ADF_Cal(give)
    adf_keys = list(adf_result[4].keys())
    adf_vals = list(adf_result[4].values())

    # adf_pre = f'ADF results:' \
    #           f' ADF Statistic:{adf_result[0]} | p-value:{adf_result[1]} | Critical Values: | {adf_keys[0]}:{adf_vals[0]} | {adf_keys[1]}:{adf_vals[1]} | {adf_keys[2]}:{adf_vals[2]}'
    o1 =f' ADF Statistic:{adf_result[0]}'
    o2 =f' p-value:{adf_result[1]}'
    o3 =f'{adf_keys[0]}:{adf_vals[0]} '
    o4 =f'{adf_keys[1]}:{adf_vals[1]}'
    o5 =f'{adf_keys[2]}:{adf_vals[2]}'

    if adf_result[1]<0.05:
        adf_ver = 'Data is stationary'
    else:
        adf_ver = 'Data is non-stationary'

    kpss_result = helper.kpss_test(give)

    kpss_pre = f'KPSS results: | {kpss_result.index[0]}:{kpss_result[0]} | {kpss_result.index[1]}:{kpss_result[1]} | {kpss_result.index[2]}:{kpss_result[2]}  | {kpss_result.index[3]}:{kpss_result[3]} | {kpss_result.index[4]}:{kpss_result[4]} | {kpss_result.index[5]}:{kpss_result[5]} | {kpss_result.index[6]}:{kpss_result[6]}'
    o6 = f'{kpss_result.index[0]}:{kpss_result[0]}'
    o7 = f'{kpss_result.index[1]}:{kpss_result[1]}'
    o8 = f'{kpss_result.index[2]}:{kpss_result[2]}'
    o9 = f'{kpss_result.index[3]}:{kpss_result[3]}'
    o10 = f'{kpss_result.index[4]}:{kpss_result[4]}'
    o11 = f'{kpss_result.index[5]}:{kpss_result[5]}'
    o12 = f'{kpss_result.index[6]}:{kpss_result[6]}'

    if kpss_result[1]>0.05:
        kpss_ver = 'Data is stationary'
    else:
        kpss_ver = 'Data is non-stationary'

    return o1,o2,o3,o4,o5,adf_ver,o6,o7,o8,o9,o10,o11,o12,kpss_ver

#===================== OLS Regression ================================

ols_layout = html.Div([
    html.H1('OLS Regression results', style={'textAlign':'center'}),
    html.Br(),
    dcc.RadioItems(
        id='data11',
        options=[
            {'label':'Run model  ','value':'0'},
        ],value='0'
    ),
    html.Br(),
    html.Br(),
    html.P('Results of OLS Regression :'),
    html.H2(id='output10'),
    html.P('y-test plot :'),
    dcc.Graph(id = 'output11'),
    html.Br(),
    html.P('y-prediction plot :'),
    dcc.Graph(id = 'output12'),
    html.Br(),

])

@app.callback(
    [Output(component_id='output10', component_property='children'),
     Output(component_id='output11', component_property='figure'),
     Output(component_id='output12', component_property='figure'),],
    [Input(component_id='data11',component_property='value'),]
)
def display_color1(dats):
    if int(dats)==0:
        pass

    ols_results = f'ADF results:' \
                  f' AIC :{results_1.aic}       |       BIC :{results_1.aic}        |       p-value : {results_1.pvalues[0]} | RMSE: {np.sqrt(results_1.mse_total)} ' \
                      f'| R-squared:{results_1.rsquared} | Adjusted R-squared:{results_1.rsquared_adj} | condition number : {cond_number} ' \
                      f'| Singular values : {singular1}'

    fig1 = px.line(x=np.arange(len(y_test)), y=y_test)
    fig2 = px.line(x=np.arange(len(y_test)), y=y_pred_ols)

    return ols_results,fig1, fig2


#===================== Base Models ================================

basemodel_layout = html.Div([
    html.H1('Base models', style={'textAlign':'center'}),
    html.Br(),
    html.P('test data plot'),
    dcc.Graph(id = 'my-graph100'),
    html.Br(),

    # html.P('prediction plot'),
    # dcc.Graph(id = 'my-graph101'),
    # html.Br(),

    html.P('Please select the base model'),
    dcc.Dropdown(
        id='data100',
        options=[
            {'label':'Average Method','value':'0'},
            {'label':'Naive Method','value':'1'},
            {'label':'Drift Method','value':'2'},
            {'label':'SES Method','value':'3'},
            {'label':'Holt Winter Method','value':'4'},
        ],value='0'
    ),
    html.Br(),
])

@app.callback(
    Output(component_id='my-graph100', component_property='figure'),
    [Input(component_id='data100',component_property='value'),]
)
# @app.callback(
#     [Output(component_id='my-graph100', component_property='figure'),
#      Output(component_id='my-graph101', component_property='figure'),],
#     [Input(component_id='data100',component_property='value'),]
# )

def display_color2(dats):
    df1 = pd.DataFrame()
    df1['yt']=yt

    if int(dats)==0:

        y_hat=[]
        for i in range(len(yt)):
            if i==0:
                y_hat.append(np.nan)
            else:
                y_hat.append(round((pd.Series(yt)).head(i).mean(),3))

        df1['average']=y_hat

        y_avg_pred = []
        for j in yf.index:
            y_avg_pred.append(round(df1['average'].mean(),3))

        y_avg_pred = pd.Series(y_avg_pred,index=yf.index)

        fig1 = px.line(x=dtest,y=yf)
        fig1.add_scatter(x=dtest,y=y_avg_pred,mode='lines')
        # fig2 = px.line(y=y_avg_pred)

    elif int(dats)==1:
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

        fig1 = px.line(x=dtest,y=yf)
        fig1.add_scatter(x=dtest,y=y_naive_pred,mode='lines')
        # fig2 = px.line(y=y_naive_pred)

    elif int(dats)==2:
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

        fig1 = px.line(x=dtest,y=yf)
        fig1.add_scatter(x=dtest,y=y_drift_pred,mode='lines')
        # fig2 = px.line(y=y_drift_pred)

    elif int(dats)==3:
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

        fig1 = px.line(x=dtest,y=yf)
        fig1.add_scatter(x=dtest,y=y_ses_pred,mode='lines')
        # fig2 = px.line(y=y_ses_pred)

    elif int(dats)==4:
        holtt=ets.ExponentialSmoothing(yt,trend='mul',damped_trend=True,seasonal='mul',seasonal_periods=12).fit()
        holtf=holtt.forecast(steps=len(yf))
        # holtf=pd.DataFrame(holtf).set_index(yf.index)

        fig1 = px.line(x=dtest,y=yf)
        fig1.add_scatter(x=dtest,y=holtf,mode='lines')
        # fig2 = px.line(y=holtf)


    return fig1


#===================== ARMA/ARIMA Models ================================

mainmodels_layout = html.Div([
    html.H1('Time series models', style={'textAlign':'center'}),
    html.Br(),
    html.P('Please select the model'),
    dcc.RadioItems(
        id='data1000',
        options=[
            {'label':'ARMA (1,1)','value':'0'},
            {'label':'ARMA (4,4)','value':'1'},
            {'label':'ARIMA (1,1,1)','value':'2'},
        ],value='0'
    ),
    html.Br(),
    html.H2('Model Coefficients:'),
    html.H2('AR Coefficients:'),
    html.H3(id='put1'),
    html.H2('MA Coefficients:'),
    html.H3(id='put2'),
    html.Br(),
    html.H2('Residual Analysis:'),
    html.H2('ACF plot of residual errors:'),
    dcc.Graph(id = 'my-graph1000a'),
    html.Br(),
    html.H3(id='put3'),
    html.H2('1 Step ahead prediction:'),
    dcc.Graph(id = 'my-graph1001a'),
    html.P('Time Range Slider'),
    dcc.Slider(id='slider1001a', min=0, max=38552, value=38552,
               marks= {100:'100',500:'500',1000:'1000',5000:'5000',10000:'10000',20000:'20000',30000:'30000',38552:'38552'}),
    html.Br(),
    html.Br(),
    html.H2('h step ahead prediction:'),
    dcc.Graph(id = 'my-graph1002a'),
    html.P('Time Range Slider'),
    dcc.Slider(id='slider1002a', min=0, max=9639, value=9639,
               marks= {100:'100',500:'500',1000:'1000',2000:'2000',4000:'4000',5000:'5000',8000:'8000',9639:'9639'}),
    html.Br(),
    html.Br(),


])

@app.callback(
    [Output(component_id='put1', component_property='children'),
     Output(component_id='put2', component_property='children'),
     Output(component_id='my-graph1000a', component_property='figure'),
     Output(component_id='put3', component_property='children'),
     Output(component_id='my-graph1001a', component_property='figure'),
     Output(component_id='my-graph1002a', component_property='figure'),],
    [Input(component_id='data1000',component_property='value'),
     Input(component_id='slider1001a',component_property='value'),
     Input(component_id='slider1002a',component_property='value'),],
)


def display_color4(dats, s1, s2):
    if int(dats)==0:
        na = 1
        nb = 1

        model = sm.tsa.ARMA(yt1,(na,nb)).fit(trend='nc',disp=0)
        print(model.summary())

        b1_ar = model.params[0]
        b1 = f'ar1 : {b1_ar}'

        b2_ma = model.params[1]
        b2 = f'ma1 : {b2_ma}'


        # =========== 1 step prediction ================

        model_hat = model.predict(start=0,end=len(yt1)-1)

        y_hat = helper.inverse_diff(y[:len(yt1)].values,np.array(model_hat),1)

        res_arma_error = y[1:len(yt1)] - y_hat

        lags1=50

        acf3 = sm.tsa.stattools.acf(res_arma_error, nlags=int(lags1))
        y_acf = np.hstack(((acf3[::-1])[:-1],acf3))

        trace = plotly.graph_objs.Scatter(x=np.arange(-int(lags1), int(lags1)+1 ),
                                      y=np.hstack(((acf3[::-1])[:-1],acf3)),
                                      mode='markers')
        z=np.arange(-int(lags1), int(lags1)+1 )
        shapes = list()
        count1 = 0
        for i in z:
            shapes.append({'type': 'line', 'xref': 'x', 'yref': 'y', 'x0': i, 'y0': 0, 'x1': i, 'y1': y_acf[count1]})
            count1 += 1

        layout = plotly.graph_objs.Layout(shapes=shapes)
        fig1 = plotly.graph_objs.Figure(data=[trace],layout=layout)

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

        Q = len(yt)*np.sum(np.square(acf3[lags1:]))

        DOF = lags1-na-nb

        alfa = 0.01

        chi_critical = chi2.ppf(1-alfa, DOF)

        print('Chi Squared test results')

        if Q<chi_critical:
            b3 = f'The residuals is white, chi squared value :{Q}'
        else:
            b3 = f'The residual is NOT white, chi squared value :{Q}'

        # time1 = time[:int(s1)]

        fig2 = px.line(x=time[:int(s1)],y=y[:int(s1)])
        fig2.add_scatter(x=time[:int(s1)],y=y_hat[:int(s1)])

        # =========== h step prediction ================
        forecast = model.forecast(steps=len(yf1))

        y_arma_pred = pd.Series(forecast[0], index=yf1.index)

        y_hat_fore = helper.inverse_diff(y[len(yt1):].values,np.array(y_arma_pred),1)

        # h step prediction

        need1 = y[len(yt1)+1:].values.flatten()
        fig3 = px.line(x=dtest[:int(s1)],y=need1[:int(s2)])
        fig3.add_scatter(x=dtest[:int(s1)],y=y_hat_fore[:int(s2)])

        res_arma_forecast = y[len(yt1)+1:] - y_hat_fore

        print(f'variance of residual error : {np.var(res_arma_error)}')

        print(f'variance of forecast error : {np.var(res_arma_forecast)}')

    elif int(dats)==1:

        na = 4
        nb = 4

        model1 = sm.tsa.ARMA(yt1,(na,nb)).fit(trend='nc',disp=0)
        print(model1.summary())

        b1_ar1=[]

        for i in range(na):
            b1_ar1.append(model1.params[i])

        b1 = f'ar1 : {b1_ar1[0]} | ar2 : {b1_ar1[1]} | ar3 : {b1_ar1[2]} | ar4 : {b1_ar1[3]} |'

        b1_ma1=[]
        for i in range(nb):
            b1_ma1.append(model1.params[i+na])

        b2 = f'ma1 : {b1_ma1[0]} | ma2 : {b1_ma1[1]} | ma3 : {b1_ma1[2]} | ma4 : {b1_ma1[3]} |'

        #================= 1 step prediction =====================
        model_hat1 = model1.predict(start=0,end=len(yt1)-1)

        y_hat1 = helper.inverse_diff(y[:len(yt1)].values,np.array(model_hat1),1)

        res_arma_error1 = y[1:len(yt1)] - y_hat1

        # lags=100

        lags1=50

        acf3 = sm.tsa.stattools.acf(res_arma_error1, nlags=int(lags1))
        y_acf = np.hstack(((acf3[::-1])[:-1],acf3))

        trace = plotly.graph_objs.Scatter(x=np.arange(-int(lags1), int(lags1)+1 ),
                                      y=np.hstack(((acf3[::-1])[:-1],acf3)),
                                      mode='markers')
        z=np.arange(-int(lags1), int(lags1)+1 )
        shapes = list()
        count1 = 0
        for i in z:
            shapes.append({'type': 'line', 'xref': 'x', 'yref': 'y', 'x0': i, 'y0': 0, 'x1': i, 'y1': y_acf[count1]})
            count1 += 1

        layout = plotly.graph_objs.Layout(shapes=shapes)
        fig1 = plotly.graph_objs.Figure(data=[trace],layout=layout)

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

        Q = len(yt1)*np.sum(np.square(acf3[lags1:]))

        DOF = lags1-na-nb

        alfa = 0.01

        chi_critical = chi2.ppf(1-alfa, DOF)

        print('Chi Squared test results')

        if Q<chi_critical:
            b3 = f'The residuals is white, chi squared value :{Q}'
        else:
            b3 = f'The residual is NOT white, chi squared value :{Q}'


        fig2 = px.line(x=time[:int(s1)],y=y[:int(s1)])
        fig2.add_scatter(x=time[:int(s1)],y=y_hat1[:int(s1)])

        # =========== h step prediction ================

        forecast1 = model1.forecast(steps=len(yf1))

        y_arma_pred1 = pd.Series(forecast1[0], index=yf1.index)

        y_hat_fore1 = helper.inverse_diff(y[len(yt1):].values,np.array(y_arma_pred1),1)

        # h step prediction

        need1 = y[len(yt1)+1:].values.flatten()
        fig3 = px.line(x=dtest[:int(s2)],y=need1[:int(s2)])
        fig3.add_scatter(x=dtest[:int(s2)],y=y_hat_fore1[:int(s2)])

        res_arma_forecast = y[len(yt1)+1:] - y_hat_fore1

        print(f'variance of residual error : {np.var(res_arma_error1)}')

        print(f'variance of forecast error : {np.var(res_arma_forecast)}')

    elif int(dats)==2:

        na = 1
        d=1
        nb = 1

        # yt,yf=train_test_split(y, shuffle=False, test_size=0.2)

        model2 = sm.tsa.ARIMA(endog=yt,order=(na,d,nb)).fit()
        print(model2.summary())

        b1_ari = model2.params[1]

        b1 = f'ar1 : {b1_ari}'

        b2_ari = model2.params[2]

        b2 = f'ma1: {b2_ari}'

        print('confidence intervals of estimated parameters:',model2.conf_int())


        # ================== 1 step prediction ========================
        model_hat2 = model2.predict(start=1,end=len(yt)-1)

        y_hat3 = helper.inverse_diff(y[:len(yt)].values,np.array(model_hat2),1)

        res_arima_error = y[1:len(yt)] - y_hat3

        lags1=50

        acf3 = sm.tsa.stattools.acf(res_arima_error, nlags=int(lags1))
        y_acf = np.hstack(((acf3[::-1])[:-1],acf3))

        trace = plotly.graph_objs.Scatter(x=np.arange(-int(lags1), int(lags1)+1 ),
                                      y=np.hstack(((acf3[::-1])[:-1],acf3)),
                                      mode='markers')
        z=np.arange(-int(lags1), int(lags1)+1 )
        shapes = list()
        count1 = 0
        for i in z:
            shapes.append({'type': 'line', 'xref': 'x', 'yref': 'y', 'x0': i, 'y0': 0, 'x1': i, 'y1': y_acf[count1]})
            count1 += 1

        layout = plotly.graph_objs.Layout(shapes=shapes)
        fig1 = plotly.graph_objs.Figure(data=[trace],layout=layout)

        # diagnostic testing
        from scipy.stats import chi2

        print('confidence intervals of estimated parameters:',model2.conf_int())

        Q = len(yt)*np.sum(np.square(acf3[lags1:]))

        DOF = lags1-na-nb

        alfa = 0.01

        chi_critical = chi2.ppf(1-alfa, DOF)

        print('Chi Squared test results')

        if Q<chi_critical:
            b3 = f'The residuals is white, chi squared value :{Q}'
        else:
            b3 = f'The residual is NOT white, chi squared value :{Q}'

        fig2 = px.line(x=time[:int(s1)],y=y[:int(s1)])
        fig2.add_scatter(x=time[:int(s1)],y=y_hat3[:int(s1)])

        # ================== h step prediction ========================
        forecast2 = model2.forecast(steps=len(yf))

        y_arima_pred = pd.Series(forecast2[0], index=yf.index)

        need1 = yf.values.flatten()
        need2 = y_arima_pred.values.flatten(),
        fig3 = px.line(x=dtest[:int(s2)],y=need1[:int(s2)])
        fig3.add_scatter(x=dtest[:int(s2)],y=need2[:int(s2)])

        res_arima_forecast = np.array(yf) - forecast2[0]

        print(f'variance of residual error : {np.var(res_arima_error)}')

        print(f'variance of forecast error : {np.var(res_arima_forecast)}')

    return b1,b2,fig1,b3,fig2,fig3

#=========================== call back to Sidebar ================================================
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/line1":
        return line_layout
    elif pathname == "/hist1":
        return histogram_layout
    elif pathname == "/acf1":
        return acf_pacf_layout
    elif pathname == "/gpac1":
        return gpac_layout
    elif pathname == "/adf1":
        return adf_layout
    elif pathname == "/ols1":
        return ols_layout
    elif pathname == "/base1":
        return basemodel_layout
    elif pathname == "/main1":
        return mainmodels_layout


    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__=='__main__':
    app.run_server(port=8054)











