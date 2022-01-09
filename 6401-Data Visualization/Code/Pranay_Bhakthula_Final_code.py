import dash
import dash_bootstrap_components as dbc
import plotly.figure_factory as ff
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from dash import dcc as dcc1
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# import dataset
df = pd.read_csv('GamingStudy_data.csv', encoding='cp1252')

print(df.describe())

print(df.info())

# number of nulls in each column
print(df.isnull().sum())

#
df.drop(df.iloc[:,:15].columns, axis=1, inplace=True)

df.drop(df.iloc[:,8:25].columns, axis=1, inplace=True)

df.drop(['highestleague','Residence_ISO3','Birthplace_ISO3','Reference','accept','League'],axis=1,inplace=True) # Redundant columns

df.drop(df[(df['Playstyle']!='Singleplayer') & (df['Playstyle']!='Multiplayer - online - with strangers') & (df['Playstyle']!='Multiplayer - online - with online acquaintances or teammates') & (df['Playstyle']!='Multiplayer - online - with real life friends') & (df['Playstyle']!='Multiplayer - offline (people in the same room)') & (df['Playstyle']!='all of the above')].index,axis=0,inplace=True)

df.drop(df[df['Hours']>5000].index,axis=0,inplace=True)

print(df.isnull().sum())

# df1=df.copy()
df['SPIN_T'].fillna(df['SPIN_T'].mode()[0],inplace=True)

df.dropna(inplace=True)

print(df.isnull().sum())


North_america = ['USA','Canada','St Vincent','El Salvador','Honduras','Guatemala','Belize','Grenada','Guadeloupe','Panama',
                 'Puerto Rico',  'Costa Rica','Mexico','Dominican Republic','Nicaragua']

asia = ['South Korea', 'Bahrain','Mongolia','Kuwait', 'Japan','Malaysia','UAE','Vietnam','Thailand', 'India', 'Hong Kong',
        'Kazakhstan','Taiwan','Indonesia','Pakistan','Singapore','Bangladesh', 'China', 'Turkey','Russia',
        'Saudi Arabia','Jordan','Brunei','Israel', 'Qatar','Philippines','Palestine','Syria','Lebanon' ]

europe = ['Germany','Liechtenstein','Republic of Kosovo','Montenegro',  'Georgia', 'Faroe Islands',  'Moldova','Albania', 'Belarus','Slovenia','Finland','UK','Ireland','Sweden', 'Greece','Belgium','Iceland','Switzerland', 'Netherlands', 'Austria','Croatia',
          'Denmark', 'Portugal', 'France','Malta','Bosnia and Herzegovina','Romania','Latvia','Estonia','Czech Republic', 'Norway', 'Poland',
          'Serbia','Spain','Slovakia','Gibraltar ','Bulgaria', 'Italy','Ukraine','Hungary','Macedonia',  'Luxembourg', 'Lithuania', 'Cyprus']

africa = ['South Africa','Jamaica', 'Egypt','Morocco','Tunisia','Algeria','Namibia',]

australia = ['Australia','New Zealand ', 'Fiji',]

south_america = [ 'Argentina','Ecuador','Brazil','Bolivia','Colombia', 'Trinidad & Tobago', 'Venezuela','Chile', 'Peru','Uruguay', ]

unknown1 = ['Unknown']

conti = []
for i in df['Residence']:
    if i in North_america:
        conti.append('North America')
    elif i in asia:
        conti.append('Asia')
    elif i in europe:
        conti.append('Europe')
    elif i in south_america:
        conti.append('South America')
    elif i in australia:
        conti.append('Australia')
    elif i in africa:
        conti.append('Africa')
    elif i in unknown1:
        conti.append('Unknown')

df['Res_continent'] = conti


gad_new = []
for i in df['GAD_T']:
    if i<=4:
        gad_new.append('mild')
    elif ((i>=5)&(i<=9)):
        gad_new.append('moderate')
    elif ((i>=10)&(i<=14)):
        gad_new.append('moderately severe')
    elif i>=15:
        gad_new.append('severe')

swl_new = []
for i in df['SWL_T']:
    if i<=9:
        swl_new.append('Extremely Dissatisfied')
    elif ((i>=10)&(i<=14)):
        swl_new.append('Dissatisfied')
    elif ((i>=15)&(i<=19)):
        swl_new.append('Slightly Dissatisfied')
    elif i==20:
        swl_new.append('Neutral')
    elif ((i>=21)&(i<=25)):
        swl_new.append('Slightly Satisfied')
    elif ((i>=26)&(i<=30)):
        swl_new.append('Satisfied')
    elif ((i>=31)&(i<=35)):
        swl_new.append('Extremely Satisfied')

spin_new = []
for i in df['SPIN_T']:
    if i<=20:
        spin_new.append('No phobia')
    elif ((i>=21)&(i<=30)):
        spin_new.append('Mild')
    elif ((i>=31)&(i<=40)):
        spin_new.append('Moderate')
    elif ((i>=41)&(i<=50)):
        spin_new.append('Severe')
    elif i>=51:
        spin_new.append('Very Severe')

df['GAD_New'] = gad_new
df['SWL_New'] = swl_new
df['SPIN_New'] = spin_new

# ================================== Outlier Detection =====================================

df1=df.copy()
cols_out = ['Hours','Age','streams']

for i in cols_out:
    q1_h, q2_h, q3_h = df1[i].quantile([0.25,0.5,0.75])

    IQR_h = q3_h - q1_h
    lower1 = q1_h - 1.5*IQR_h
    upper1 = q3_h + 1.5*IQR_h

    print(f'Q1 and Q3 of the {i} is {q1_h:.2f}  & {q3_h:.2f} ')
    print(f'IQR for the {i} is {IQR_h:.2f} ')
    print(f'Any {i} < {lower1:.2f}  and {i} > {upper1:.2f}  is an outlier')

    sns.boxplot(y=df1[i])
    plt.title(f'Boxplot of {i} before removing outliers')
    plt.show()


    df1 = df1[(df1[i] > lower1) & (df1[i] < upper1)]

    sns.boxplot(y=df1[i])
    plt.title(f'Boxplot of {i} after removing outliers')
    plt.show()

# # perform IQR method 2nd time to reduce the outliers
#
# cols_out = ['Hours','streams']
#
# for i in cols_out:
#     q1_h, q2_h, q3_h = df1[i].quantile([0.25,0.5,0.75])
#
#     IQR_h = q3_h - q1_h
#     lower1 = q1_h - 1.5*IQR_h
#     upper1 = q3_h + 1.5*IQR_h
#
#     print(f'Q1 and Q3 of the {i} is {q1_h:.2f}  & {q3_h:.2f} ')
#     print(f'IQR for the {i} is {IQR_h:.2f} ')
#     print(f'Any {i} < {lower1:.2f}  and {i} > {upper1:.2f}  is an outlier')
#
#     sns.boxplot(y=df1[i])
#     plt.title(f'Boxplot of {i} before removing outliers')
#     plt.show()
#
#
#     df1 = df1[(df1[i] > lower1) & (df1[i] < upper1)]
#
#     sns.boxplot(y=df1[i])
#     plt.title(f'Boxplot of {i} after removing outliers')
#     plt.show()



#================================= PCA analysis =========================================



features = ['Hours','Age','streams','Narcissism','SWL_T','SPIN_T','GAD_T']

X = df[features].values

X = StandardScaler().fit_transform(X)

#b
H = np.matmul(X.T,X)
_,d,_=np.linalg.svd(X)
print(f'Original X singular values {d}')
print(f'Original X condition number {np.linalg.cond(X)}')

plt.figure()
sns.heatmap(df[features].corr(), annot=True)
plt.tight_layout()
plt.show()

#d
pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)

X_PCA=pca.transform(X)

print(f'explained variance ratio of original vs reduced feature space :{pca.explained_variance_ratio_}')

plt.plot(np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1,1),np.cumsum(pca.explained_variance_ratio_))
plt.xticks(np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1,1))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


import pandas as pd
plt.figure()
sns.heatmap(pd.DataFrame(X_PCA).corr(), annot=True)
plt.title('correlation plot of PCA features')
plt.show()

#================================= Normality test =========================================


features = ['Hours','Age','streams','Narcissism','SWL_T','SPIN_T','GAD_T']
for i in features:
    print(f'Normality test of {i} column :')

    kstest_x = st.kstest(df[i],'norm')

    print(f"K-S test: statistics={kstest_x[0]:.4f} p-value={kstest_x[1]:.4f}")
    if kstest_x[1] > 0.01:
        print(f"{i} dataset looks Normal")
    else:
        print(f"{i} dataset looks Non-Normal")

    shapiro_x = st.shapiro(df[i])

    print(f"Shapiro test: statistics={shapiro_x[0]:.4f} p-value={shapiro_x[1]:.4f}")
    if shapiro_x[1] > 0.01:
        print(f"{i} dataset looks Normal")
    else:
        print(f"{i} dataset looks Non-Normal")

    dak_x = st.normaltest(df[i])

    print(f"da_k_squared test: statistics={dak_x[0]:.4f} p-value={dak_x[1]:.4f}")
    if dak_x[1] > 0.01:
        print(f"{i} dataset looks Normal")
    else:
        print(f"{i} dataset looks Non-Normal")

    print('\n\n')




#======================= Dash Board ================================================

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
            "Online Gamers Anxiety Analysis", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Statistics", href="/stats1", active="exact"),
                dbc.NavLink("Line Plot", href="/line1", active="exact"),
                dbc.NavLink("Bar Plot", href="/bar1", active="exact"),
                # dbc.NavLink("Bar Group Plot", href="/barg1", active="exact"),
                dbc.NavLink("Pie Chart", href="/pie1", active="exact"),
                dbc.NavLink("Count Plot", href="/count1", active="exact"),
                dbc.NavLink("Histogram Plot", href="/hist1", active="exact"),
                dbc.NavLink("Scatter Plot", href="/scatter1", active="exact"),
                dbc.NavLink("Box Plot", href="/box1", active="exact"),
                dbc.NavLink("Heatmap", href="/heat1", active="exact"),
                dbc.NavLink("Pair Plot", href="/pair1", active="exact"),
                dbc.NavLink("Violin Plot", href="/violin1", active="exact"),
                dbc.NavLink("KDE Plot", href="/kde1", active="exact"),
                dbc.NavLink("Area Plot", href="/area1", active="exact"),
                dbc.NavLink("Download csv file", href="/download1", active="exact"),

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

#===================== Statistics ================================

stats_layout = html.Div([
    html.H1('Basic statistics', style={'textAlign':'center'}),
    html.Br(),
    html.Br(),
    html.P('Radiobox to select Variable'),
    html.Br(),
    dcc.RadioItems(
        id='select121',
        options=[
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
            {'label':'Gender','value':'Gender'},
            {'label':'Platform','value':'Platform'},
            {'label':'Game','value':'Game'},
            {'label':'Work','value':'Work'},
            {'label':'Degree','value':'Degree'},
            {'label':'Playstyle','value':'Playstyle'},
            {'label':'Residence','value':'Residence'},
            {'label':'Res_continent','value':'Res_continent'},
        ],value=['Hours'],
    ),
    html.Br(),
    html.Br(),
    html.H2(id='output111'),
    html.Br(),
    html.H2(id='output112'),
    html.Br(),
    html.H2(id='output113'),
    html.Br(),
    html.H2(id='output114'),
    html.Br(),
    html.H2(id='output115')
])

@app.callback(
    [Output(component_id='output111', component_property='children'),
     Output(component_id='output112', component_property='children'),
     Output(component_id='output113', component_property='children'),
     Output(component_id='output114', component_property='children'),
     Output(component_id='output115', component_property='children'),],
    [Input(component_id='select121',component_property='value'),]
)

def display_color(sel1):
    if type(df1.loc[0,sel1])==type('hi'):
        out_stat1 = f'Variable is Categorical'
        out_stat2 = f'mode : {df1[sel1].mode()[0]}'
        out_stat3 = f'count : {len(df1)}'
        out_stat4 = f'number of  unique values : {len(df1[sel1].unique())}'
        out_stat5 = f''
    else:
        out_stat1 = f'Variable is Numeric'
        out_stat2 = f'mean : {np.mean(df1[sel1])}'
        out_stat3 = f'median : {np.median(df1[sel1])}'
        out_stat4 = f'standard deviation : {np.std(df1[sel1])}'
        out_stat5 = f'Variance : {np.var(df1[sel1])}'

    return out_stat1, out_stat2, out_stat3, out_stat4, out_stat5


#===================== Line Plot ================================

line_layout = html.Div([
    html.H1('Line Plot', style={'textAlign':'center'}),
    html.Br(),
    html.P('Radiobox to select Clean Dataset'),
    dcc.RadioItems(
        id='select1a',
        options=[
            {'label':' Clean Dataset','value':'ON'},
        ],value=['OFF'],
    ),

    html.H2('Single Selection Line Plot', style={'textAlign':'center'}),
    html.Br(),
    dcc.Graph(id = 'my-graph1'),
    html.P('select y-axis'),
    dcc.Dropdown(
        id='select1',
        options=[
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
        ],value='Hours'
    ),
    html.Br(),
    html.Br(),
    html.Br(),

    html.H2('Multiple Selection Line Plot', style={'textAlign':'center'}),
    html.Br(),
    dcc.Graph(id = 'my-graph2'),
    html.P('select x-axis'),
    dcc.Dropdown(
        id='select2',
        options=[
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
        ],value='Hours'
    ),

    html.Br(),
    html.P('select y-axis'),
    dcc.Dropdown(
        id='select3',
        options=[
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
        ],value='Hours'
    ),
    html.Br(),


])

@app.callback(
    [Output(component_id='my-graph1', component_property='figure'),
     Output(component_id='my-graph2', component_property='figure')],
    [Input(component_id='select1a',component_property='value'),
     Input(component_id='select1',component_property='value'),
     Input(component_id='select2',component_property='value'),
     Input(component_id='select3',component_property='value')]
)

def display_color(sel1,value_y1,value_x2,value_y2):
    if 'ON' in sel1:
        fig1 = px.line(x=np.arange(len(df1)),y=df1[value_y1])
        fig2 = px.line(df1,x=value_x2,y=value_y2)
    else:
        fig1 = px.line(x=np.arange(len(df)),y=df[value_y1])
        fig2 = px.line(df,x=value_x2,y=value_y2)
    return fig1, fig2

#===================== Bar Plot ================================

bar_layout = html.Div([
    html.H1('Bar Plot', style={'textAlign':'center'}),
    html.Br(),
    html.P('Radiobox to select Clean Dataset'),
    dcc.RadioItems(
        id='select4a',
        options=[
            {'label':' Clean Dataset','value':'ON'},
        ],value=['OFF'],
    ),
    html.Br(),
    dcc.Graph(id = 'my-graph3'),
    html.Br(),
    html.H1('Bar Group Plot ', style={'textAlign':'center'}),
    dcc.Graph(id = 'my-graph300'),
    html.P('select x-axis'),
    dcc.Dropdown(
        id='select4',
        options=[
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
            {'label':'GAD_New','value':'GAD_New'},
            {'label':'SWL_New','value':'SWL_New'},
            {'label':'SPIN_New','value':'SPIN_New'},
        ],value='Hours'
    ),

    html.Br(),
    html.P('select y-axis'),
    dcc.Dropdown(
        id='select5',
        options=[
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
            {'label':'GAD_New','value':'GAD_New'},
            {'label':'SWL_New','value':'SWL_New'},
            {'label':'SPIN_New','value':'SPIN_New'},
        ],value='Age'
    ),
    html.Br(),

    html.P('Select legend'),
    dcc.Dropdown(
        id='select5a',
        options=[
            {'label':'GAD_New','value':'GAD_New'},
            {'label':'SWL_New','value':'SWL_New'},
            {'label':'SPIN_New','value':'SPIN_New'},
        ],value='GAD_New'
    ),
    html.Br(),

])

@app.callback(
    [Output(component_id='my-graph3', component_property='figure'),
     Output(component_id='my-graph300', component_property='figure'),],
    [Input(component_id='select4a',component_property='value'),
     Input(component_id='select4',component_property='value'),
     Input(component_id='select5',component_property='value'),
     Input(component_id='select5a',component_property='value'),]
)

def display_color1(sel1,value_x, value_y, val_leg):
    if 'ON' in sel1:
        fig1 = px.bar(df1,x=value_x,y=value_y,color=val_leg)
        fig2 = px.bar(df1,x=value_x,y=value_y,color=val_leg,barmode='group')
        # fig1 = sns.barplot(data=df1,x=value_x,y=value_y,hue=val_leg)
    else:
        fig1 = px.bar(df,x=value_x,y=value_y,color=val_leg)
        fig2 = px.bar(df,x=value_x,y=value_y,color=val_leg,barmode='group')
        # fig1 = sns.barplot(data=df,x=value_x,y=value_y,hue=val_leg)

    return fig1, fig2



#===================== Pie Plot ================================

pie_layout = html.Div([
    html.H1('Pie Plot', style={'textAlign':'center'}),
    html.Br(),
    html.P('Checkbox to select Clean Dataset'),
    dcc.Checklist(
        id='select6a',
        options=[
            {'label':' Clean Dataset','value':'ON'},
        ],value=['OFF'],
    ),
    html.Br(),

    dcc.Graph(id = 'my-graph4'),
    html.P('select a variable'),
    dcc.Dropdown(
        id='select6',
        options=[
            {'label':'Gender','value':'Gender'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'Platform','value':'Platform'},
            {'label':'Game','value':'Game'},
            {'label':'Work','value':'Work'},
            {'label':'Degree','value':'Degree'},
            {'label':'Residence','value':'Residence'},
            {'label':'Res_continent','value':'Res_continent'},
            {'label':'Playstyle','value':'Playstyle'},
            {'label':'GAD_New','value':'GAD_New'},
            {'label':'SWL_New','value':'SWL_New'},
            {'label':'SPIN_New','value':'SPIN_New'},

        ],value='Gender'
    ),

])

@app.callback(
    Output(component_id='my-graph4', component_property='figure'),
    [Input(component_id='select6a',component_property='value'),
     Input(component_id='select6',component_property='value'),]
)

def display_color2(sel1,value_x):
    if 'ON' in sel1:
        if (type(df1.loc[0,value_x]))==type('hi'):
            uni1 = df1[value_x].unique()
            count1 = [len(df1[df1[value_x]==i]) for i in uni1]
            fig1 = px.pie(values=count1,names=uni1)
        else:
            fig1 = px.pie(df1,values=value_x,names=value_x)
    else:
        if (type(df.loc[0,value_x]))==type('hi'):
            uni1 = df[value_x].unique()
            count1 = [len(df[df[value_x]==i]) for i in uni1]
            fig1 = px.pie(values=count1,names=uni1)
        else:
            fig1 = px.pie(df,values=value_x,names=value_x)
    return fig1

#===================== Count Plot ================================

count_layout = html.Div([
    html.H1('Count Plot', style={'textAlign':'center'}),
    html.Br(),
    html.P('Checkbox to select Clean Dataset'),
    dcc.Checklist(
        id='select7a',
        options=[
            {'label':' Clean Dataset','value':'ON'},
        ],value=['OFF'],
    ),
    html.Br(),
    html.Br(),
    dcc.Graph(id = 'my-graph5'),
    html.P('select a variable'),
    dcc.Dropdown(
        id='select7',
        options=[
            {'label':'Gender','value':'Gender'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'Platform','value':'Platform'},
            {'label':'Game','value':'Game'},
            {'label':'Work','value':'Work'},
            {'label':'Degree','value':'Degree'},
            {'label':'Residence','value':'Residence'},
            {'label':'Res_continent','value':'Res_continent'},
            {'label':'Playstyle','value':'Playstyle'},
            {'label':'GAD_New','value':'GAD_New'},
            {'label':'SWL_New','value':'SWL_New'},
            {'label':'SPIN_New','value':'SPIN_New'},
        ],value='Gender'
    ),

])

@app.callback(
    Output(component_id='my-graph5', component_property='figure'),
    [Input(component_id='select7a',component_property='value'),
     Input(component_id='select7',component_property='value'),]
)

def display_color3(sel1,value_x):
    if 'ON' in sel1:
        uni1 = df1[value_x].unique()
        count1 = [len(df1[df1[value_x]==i]) for i in uni1]
        fig1 = px.bar(x=uni1, y=count1, color=uni1)
    else:
        uni1 = df[value_x].unique()
        count1 = [len(df[df[value_x]==i]) for i in uni1]
        fig1 = px.bar(x=uni1, y=count1, color=uni1)
    return fig1

#===================== Histogram Plot ================================

histogram_layout = html.Div([
    html.H1('Histogram Plot', style={'textAlign':'center'}),
    html.Br(),
    html.P('Checkbox to select Clean Dataset'),
    dcc.Checklist(
        id='select8a',
        options=[
            {'label':' Clean Dataset','value':'ON'},
        ],value=['OFF'],
    ),
    html.Br(),
    dcc.Graph(id = 'my-graph6'),
    html.P('select a variable'),
    dcc.Dropdown(
        id='select8',
        options=[
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
        ],value='Hours'
    ),

    html.Br(),
    html.Br(),
    html.P('X-axis Range Slider'),
    dcc.Slider(id='slider1', min=0, max=200, value=100,
               marks= {0:'0',10:'10',20:'20',30:'30',60:'60',80:'80',100:'100',200:'200'}),
    html.Br(),
    html.Br(),


    html.P('Number of bins'),
    dcc.Slider(id='bins1', min=10, max=100, value=10,
               marks= {10:'10',15:'15',20:'20',30:'30',40:'40',50:'50',60:'60',70:'70',80:'80',90:'90',100:'100'}),
    html.Br(),

])

@app.callback(
    Output(component_id='my-graph6', component_property='figure'),
    [Input(component_id='select8a',component_property='value'),
     Input(component_id='select8',component_property='value'),
     Input(component_id='slider1',component_property='value'),
     # Input(component_id='select9',component_property='value'),
     Input(component_id='bins1',component_property='value'),]
)

def display_color4(sel1,value_x, sli, bin):
    if 'ON' in sel1:
        fig1 = px.histogram(df1[df1[value_x]<=sli], x=value_x, nbins=int(bin))
    else:
        fig1 = px.histogram(df[df[value_x]<=sli], x=value_x, nbins=int(bin))
    return fig1

#===================== Scatter Plot ================================

scatter_layout = html.Div([
    html.H1('Scatter Plot', style={'textAlign':'center'}),
    html.Br(),
    html.P('Checkbox to select Clean Dataset'),
    dcc.Checklist(
        id='select9a',
        options=[
            {'label':' Clean Dataset','value':'ON'},
        ],value=['OFF'],
    ),
    html.Br(),

    dcc.Graph(id = 'my-graph7'),
    html.P('select x-axis'),
    dcc.Dropdown(
        id='select9',
        options=[
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
        ],value='Hours'
    ),

    html.Br(),
    html.P('select y-axis'),
    dcc.Dropdown(
        id='select10',
        options=[
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
        ],value='Age'
    ),
    html.Br(),
])

@app.callback(
    Output(component_id='my-graph7', component_property='figure'),
    [Input(component_id='select9a',component_property='value'),
     Input(component_id='select9',component_property='value'),
     Input(component_id='select10',component_property='value'),]
)

def display_color5(sel1,value_x,value_y):
    if 'ON' in sel1:
        fig1 = px.scatter(df1,value_x,value_y,color=value_y,trendline='ols')
    else:
        fig1 = px.scatter(df,value_x,value_y,color=value_y,trendline='ols')
    return fig1

#===================== Box Plot ================================

box_layout = html.Div([
    html.H1('Box Plot', style={'textAlign':'center'}),
    html.Br(),
    html.P('Checkbox to select Clean Dataset'),
    dcc.Checklist(
        id='select11a',
        options=[
            {'label':' Clean Dataset','value':'ON'},
        ],value=['OFF'],
    ),
    html.Br(),

    dcc.Graph(id = 'my-graph8'),
    html.P('select x-axis'),
    dcc.Dropdown(
        id='select11',
        options=[
            {'label':'Gender','value':'Gender'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'Platform','value':'Platform'},
            {'label':'Game','value':'Game'},
            {'label':'Work','value':'Work'},
            {'label':'Degree','value':'Degree'},
            {'label':'Residence','value':'Residence'},
            {'label':'Res_continent','value':'Res_continent'},
            {'label':'Playstyle','value':'Playstyle'},
        ],value='Gender'
    ),

    html.Br(),
    html.P('select y-axis'),
    dcc.Dropdown(
        id='select12',
        options=[
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
        ],value='Age'
    ),
    html.Br(),
])

@app.callback(
    Output(component_id='my-graph8', component_property='figure'),
    [Input(component_id='select11a',component_property='value'),
     Input(component_id='select11',component_property='value'),
     Input(component_id='select12',component_property='value'),]
)

def display_color5(sel1,value_x,value_y):
    if 'ON' in sel1:
        fig1 = px.box(df1,value_x,value_y,color=value_x)
    else:
        fig1 = px.box(df,value_x,value_y,color=value_x)

    return fig1

#===================== Heatmap ================================

heat_layout = html.Div([
    html.H1('Heatmap', style={'textAlign':'center'}),
    html.Br(),
    html.P('Checkbox to select Clean Dataset'),
    dcc.Checklist(
        id='select13a',
        options=[
            {'label':' Clean Dataset','value':'ON'},
        ],value=['OFF'],
    ),
    html.Br(),

    dcc.Graph(id='my-graph9'),
    html.P('select x-axis'),
    dcc.Dropdown(
        id='select13',
        options=[
            {'label':'Gender','value':'Gender'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'Platform','value':'Platform'},
            {'label':'Game','value':'Game'},
            {'label':'Work','value':'Work'},
            {'label':'Degree','value':'Degree'},
            {'label':'Playstyle','value':'Playstyle'},
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
        ],value='Gender'
    ),

    html.Br(),
    html.P('select y-axis'),
    dcc.Dropdown(
        id='select14',
        options=[
            {'label':'Gender','value':'Gender'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'Platform','value':'Platform'},
            {'label':'Game','value':'Game'},
            {'label':'Work','value':'Work'},
            {'label':'Degree','value':'Degree'},
            {'label':'Playstyle','value':'Playstyle'},
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
        ],value='Degree'
    ),
    html.Br(),

    html.P('select values'),
    dcc.Dropdown(
        id='select15',
        options=[
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
        ],value='GAD_T'
    ),
    html.Br(),
    # html.P('Checkbox for Aggregate function'),
    # dcc.Checklist(
    #     id = 'select16',
    #     options=[
    #         {'label': ' sum', 'value': 'sum'},
    #         {'label': ' count', 'value': 'count'},
    #         {'label': ' mean', 'value': 'mean'},
    #     ],
    #     value=['mean']
    # ),
    # html.Br(),
])

@app.callback(
    Output(component_id='my-graph9', component_property='figure'),
    [Input(component_id='select13a',component_property='value'),
     Input(component_id='select13',component_property='value'),
     Input(component_id='select14',component_property='value'),
     Input(component_id='select15',component_property='value'),
     ]
)

# Input(component_id='select16',component_property='value'),
# def display_color6(sel1,value_x,value_y,val_v,val_agg):

def display_color6(sel1,value_x,value_y,val_v):
    if 'ON' in sel1:
        # heatmap_df = df1.pivot_table(index = value_y,columns = value_x, values = val_v, aggfunc=val_agg)
        heatmap_df = df1.pivot_table(index = value_y,columns = value_x, values = val_v)
        fig1 = px.imshow(heatmap_df)
    else:
        # heatmap_df = df.pivot_table(index = value_y,columns = value_x, values = val_v, aggfunc=val_agg)
        heatmap_df = df.pivot_table(index = value_y,columns = value_x, values = val_v)
        fig1 = px.imshow(heatmap_df)

    return fig1

#===================== Pair plot ================================

pair_layout = html.Div([
    html.H1('Pair Plot', style={'textAlign':'center'}),
    html.Br(),
    dcc.Graph(id='my-graph10'),
    html.P('select dimensions'),
    dcc.Dropdown(
        id='select17',
        options=[
            {'label':'Gender','value':'Gender'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'Platform','value':'Platform'},
            {'label':'Game','value':'Game'},
            {'label':'Work','value':'Work'},
            {'label':'Degree','value':'Degree'},
            {'label':'Playstyle','value':'Playstyle'},
            {'label':'Residence','value':'Residence'},
            {'label':'Res_continent','value':'Res_continent'},
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
        ],value=['Gender','Age','Hours','Degree'],
        multi=True,
    ),

])

@app.callback(
    Output(component_id='my-graph10', component_property='figure'),
    [Input(component_id='select17',component_property='value'),]
)

def display_color7(value_x):
    fig1 = px.scatter_matrix(df,dimensions=value_x)
    return fig1

#===================== Violon plot ================================

violin_layout = html.Div([
    html.H1('Violin Plot', style={'textAlign':'center'}),
    html.Br(),
    html.P('Checkbox to select Clean Dataset'),
    dcc.Checklist(
        id='select18a',
        options=[
            {'label':' Clean Dataset','value':'ON'},
        ],value=['OFF'],
    ),
    html.Br(),

    dcc.Graph(id='my-graph11'),
    html.P('select variable'),
    dcc.Dropdown(
        id='select18',
        options=[
            {'label':'Gender','value':'Gender'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'Platform','value':'Platform'},
            {'label':'Game','value':'Game'},
            {'label':'Work','value':'Work'},
            {'label':'Degree','value':'Degree'},
            {'label':'Playstyle','value':'Playstyle'},
            {'label':'Residence','value':'Residence'},
            {'label':'Res_continent','value':'Res_continent'},
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
        ],value='Age',
    ),

])

@app.callback(
    Output(component_id='my-graph11', component_property='figure'),
    [Input(component_id='select18a',component_property='value'),
     Input(component_id='select18',component_property='value'),]
)

def display_color8(sel1,value_y):
    if 'ON' in sel1:
        fig1 = px.violin(df1,y = value_y)
    else:
        fig1 = px.violin(df,y = value_y)
    return fig1

#===================== KDE plot ================================

kde_layout = html.Div([
    html.H1('KDE Plot', style={'textAlign':'center'}),
    html.Br(),
    html.P('Checkbox to select Clean Dataset'),
    dcc.Checklist(
        id='select19a',
        options=[
            {'label':' Clean Dataset','value':'ON'},
        ],value=['OFF'],
    ),
    html.Br(),

    dcc.Graph(id='my-graph12'),
    html.P('select variable'),
    dcc.Dropdown(
        id='select19',
        options=[
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'Hours','value':'Hours'},
            {'label':'Age','value':'Age'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
        ],value='Age',
    ),
    html.Br(),
    html.Br(),
    html.P('Checkbox to select histogram'),
    dcc.Checklist(
        id='select20',
        options=[
            {'label':' Select Histogram','value':'ON'},
        ],value=['OFF'],
    )

])

@app.callback(
    Output(component_id='my-graph12', component_property='figure'),
    [Input(component_id='select19a',component_property='value'),
     Input(component_id='select19',component_property='value'),
     Input(component_id='select20',component_property='value'),]
)

def display_color9(sel2,value_y,sel1):
    group_labels = [value_y]
    if 'ON' in sel2:
        if 'ON' in sel1:
            fig1 = ff.create_distplot([df1[value_y]],group_labels)
        else:
            fig1 = ff.create_distplot([df1[value_y]],group_labels,show_hist=False)
    else:
        if 'ON' in sel1:
            fig1 = ff.create_distplot([df[value_y]],group_labels)
        else:
            fig1 = ff.create_distplot([df[value_y]],group_labels,show_hist=False)

    return fig1

#===================== Area Plot ================================

area_layout = html.Div([
    html.H1('Area Plot', style={'textAlign':'center'}),
    html.Br(),
    html.P('Checkbox to select Clean Dataset'),
    dcc.Checklist(
        id='select21a',
        options=[
            {'label':' Clean Dataset','value':'ON'},
        ],value=['OFF'],
    ),
    html.Br(),

    dcc.Graph(id = 'my-graph13'),
    html.P('select x-axis'),
    dcc.Dropdown(
        id='select21',
        options=[
            {'label':'Hours','value':'Hours'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'Age','value':'Age'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
        ],value='Hours'
    ),

    html.Br(),
    html.P('select y-axis'),
    dcc.Dropdown(
        id='select22',
        options=[
            {'label':'Hours','value':'Hours'},
            {'label':'Narcissism','value':'Narcissism'},
            {'label':'Age','value':'Age'},
            {'label':'GAD_T','value':'GAD_T'},
            {'label':'SWL_T','value':'SWL_T'},
            {'label':'SPIN_T','value':'SPIN_T'},
        ],value='Age'
    ),
    html.Br(),
    html.P('select color'),
    dcc.Dropdown(
        id='select23',
        options=[
            {'label':'Gender','value':'Gender'},
            {'label':'Platform','value':'Platform'},
            {'label':'Game','value':'Game'},
            {'label':'Work','value':'Work'},
            {'label':'Degree','value':'Degree'},
            {'label':'Playstyle','value':'Playstyle'},
        ],value='Platform'
    ),
    html.Br(),
    html.P('select linegroup'),
    dcc.Dropdown(
        id='select24',
        options=[
            {'label':'Gender','value':'Gender'},
            {'label':'Platform','value':'Platform'},
            {'label':'Game','value':'Game'},
            {'label':'Work','value':'Work'},
            {'label':'Degree','value':'Degree'},
            {'label':'Playstyle','value':'Playstyle'},
        ],value='Game'
    ),
    html.Br(),
    html.Br(),
    html.Br(),

])

@app.callback(
    Output(component_id='my-graph13', component_property='figure'),
    [Input(component_id='select21a',component_property='value'),
     Input(component_id='select21',component_property='value'),
     Input(component_id='select22',component_property='value'),
     Input(component_id='select23',component_property='value'),
     Input(component_id='select24',component_property='value'),]
)

def display_color5(sel1,value_x,value_y,col,line1):
    if 'ON' in sel1:
        fig1 = px.area(df1,x=value_x,y=value_y,color=col,line_group=line1)
    else:
        fig1 = px.area(df,x=value_x,y=value_y,color=col,line_group=line1)
    return fig1

#=========================== download component ================================================

download_layout = html.Div([
    html.H2('Download the cleaned dataset', style={'textAlign':'center'}),
    html.Br(),
    html.Br(),
    html.Label('Click button to download csv file'),
    html.Br(),
    html.Br(),
    html.Button(id='download1', children='Download'),
    dcc.Download(id='download2')
])

@app.callback(Output(component_id="download2", component_property="data"),
              Input(component_id="download1", component_property="n_clicks"),
              prevent_initial_call=True)

def displaydown_layout(sel1):
    return dcc1.send_data_frame(df1.to_csv, "OnlineGamers.csv")

#=========================== call back to Sidebar ================================================
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/line1":
        return line_layout
    elif pathname == "/stats1":
        return stats_layout
    elif pathname == "/bar1":
        return bar_layout
    # elif pathname == "/barg1":
    #     return barg_layout
    elif pathname == "/pie1":
        return pie_layout
    elif pathname == "/count1":
        return count_layout
    elif pathname == "/hist1":
        return histogram_layout
    elif pathname == "/scatter1":
        return scatter_layout
    elif pathname == "/box1":
        return box_layout
    elif pathname == "/heat1":
        return heat_layout
    elif pathname == "/pair1":
        return pair_layout
    elif pathname == "/violin1":
        return violin_layout
    elif pathname == "/kde1":
        return kde_layout
    elif pathname == "/area1":
        return area_layout
    elif pathname == "/download1":
        return download_layout

    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__=='__main__':
    app.run_server(port=8051)


