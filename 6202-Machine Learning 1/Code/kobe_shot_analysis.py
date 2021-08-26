import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
pd.options.mode.chained_assignment = None  # default='warn'
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib import rcParams

data=pd.read_csv('data.csv')

# printing the dataset shape
print("Dataset No. of Rows: ", data.shape[0])
print("Dataset No. of Columns: ", data.shape[1])

# printing the dataset observations
print("Dataset first few rows:\n ")
print(data.head(6))

# printing the struture of the dataset
print("Dataset info:\n ")
data.info()

#drop missing values
data1=data.dropna()

data1.loc[data1['shot_type']=='2PT Field Goal','shot_type']=2
data1.loc[data1['shot_type']=='3PT Field Goal','shot_type']=3

# EDA analysis

#1. average points per game - only 2FG and 3FG included
points_scored=data1.shot_made_flag * data1.shot_type

games_no=data1['game_id'].unique().size

PPG=points_scored.sum()/games_no

# scored data
data_yes=data1[data1['shot_made_flag']==1]
# failed to score data
data_no=data1[data1['shot_made_flag']==0]

# 2. Which action_type worked/failed the most
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
actions=list(data1['action_type'].unique())
shots_percent=[]
total_size=data1['shot_made_flag'].size

for i in actions:
    score_no=data_yes.loc[data_yes['action_type']==i,'shot_made_flag'].size
    total_size1=data1.loc[data1['action_type']==i,'shot_made_flag'].size
    shots_percent.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])

shots_percent.sort(reverse=True)
plot_df1=pd.DataFrame(shots_percent[:10],columns=['count','eff','type','avg'], dtype=float)
fig, ax1 = plt.subplots(figsize=(10,8))
ax2 = ax1.twinx()
plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
z=0
for bar in plots.patches:
    plots.annotate('%0.2f%%'%(shots_percent[z][3]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
    z=z+1

ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
ax2.set_ylabel('FG%')
plt.title('Histogram of top 10 action_type')
ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
plt.tight_layout()
# plt.savefig('Graphs/2.jpeg', dpi=300, bbox_inches='tight')
plt.show()


# 3. From which shot_zone_area did he score/fail the most
areas=list(data1['shot_zone_area'].unique())
shots_percent=[]

for i in areas:
    score_no=data_yes.loc[data_yes['shot_zone_area']==i,'shot_made_flag'].size
    total_size1=data1.loc[data1['shot_zone_area']==i,'shot_made_flag'].size
    shots_percent.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])

shots_percent.sort(reverse=True)
plot_df1=pd.DataFrame(shots_percent,columns=['count','eff','type','avg'], dtype=float)
fig, ax1 = plt.subplots(figsize=(10,8))
ax2 = ax1.twinx()
plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
z=0
for bar in plots.patches:
    plots.annotate('%0.2f%%'%(shots_percent[z][3]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
    z=z+1

ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
ax2.set_ylabel('FG%')
plt.title('Histogram of shot_zone_area')
ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
plt.tight_layout()
# plt.savefig('Graphs/3.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# 4. From which shot_zone_basic did he score/fail the most
basics=list(data1['shot_zone_basic'].unique())
shots_percent=[]
for i in basics:
    score_no=data_yes.loc[data_yes['shot_zone_basic']==i,'shot_made_flag'].size
    total_size1=data1.loc[data1['shot_zone_basic']==i,'shot_made_flag'].size
    shots_percent.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])

shots_percent.sort(reverse=True)
plot_df1=pd.DataFrame(shots_percent,columns=['count','eff','type','avg'], dtype=float)
fig, ax1 = plt.subplots(figsize=(10,8))
ax2 = ax1.twinx()
plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
z=0
for bar in plots.patches:
    plots.annotate('%0.2f%%'%(shots_percent[z][3]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
    z=z+1

ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
ax2.set_ylabel('FG%')
plt.title('Histogram of shot_zone_basic')
ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
plt.tight_layout()
# plt.savefig('Graphs/4.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# 5. From which shot_zone_range did he score/fail the most
ranges=list(data1['shot_zone_range'].unique())
shots_percent=[]
for i in ranges:
    score_no=data_yes.loc[data_yes['shot_zone_range']==i,'shot_made_flag'].size
    total_size1=data1.loc[data1['shot_zone_range']==i,'shot_made_flag'].size
    shots_percent.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])

shots_percent.sort(reverse=True)
plot_df1=pd.DataFrame(shots_percent,columns=['count','eff','type','avg'], dtype=float)
fig, ax1 = plt.subplots(figsize=(10,8))
ax2 = ax1.twinx()
plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
z=0
for bar in plots.patches:
    plots.annotate('%0.2f%%'%(shots_percent[z][3]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
    z=z+1

ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
ax2.set_ylabel('FG%')
plt.title('Histogram of shot_zone_range')
ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
plt.tight_layout()
# plt.savefig('Graphs/5.jpeg', dpi=300, bbox_inches='tight')
plt.show()


# 6. accuracy per season accuracy = yes/yes+no
seasons=list(data1['season'].unique())
seasons.sort()
seasons_dict={}
for i in seasons:
    accuracy=(data_yes.loc[data_yes['season']==i,'shot_made_flag'].size)/(data1.loc[data1['season']==i,'shot_made_flag'].size)
    seasons_dict[i]=accuracy

plt.figure(figsize=(10,8))
plt.plot(list(seasons_dict.keys()),list(seasons_dict.values()))
plt.scatter(list(seasons_dict.keys()),list(seasons_dict.values()),s=40)
plt.title('accuracy vs seasons')
plt.xticks(rotation=90)
plt.xlabel('seasons')
plt.ylabel('accuracy')
plt.tight_layout()
# plt.savefig('Graphs/7.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# 7. shots vs period

periods=list(data1['period'].unique())
shots_percent=[]
for i in periods:
    score_no=data_yes.loc[data_yes['period']==i,'shot_made_flag'].size
    total_size1=data1.loc[data1['period']==i,'shot_made_flag'].size
    shots_percent.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])

plot_df1=pd.DataFrame(shots_percent,columns=['count','eff','type','avg'], dtype=float)
fig, ax1 = plt.subplots(figsize=(10,8))
ax2 = ax1.twinx()
plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
z=0
for bar in plots.patches:
    plots.annotate('%0.2f%%'%(shots_percent[z][3]),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
    z=z+1

l1=np.array(list(periods))-1
ax2.plot(l1, plot_df1['eff'], 'b-')
ax2.scatter(l1, plot_df1['eff'],s=40)
ax2.set_ylabel('FG%')
ax2.set_ylim([20,50])
plt.title('shots attempted vs period')
ax1.set_xticklabels(labels=plot_df1['type'])
plt.tight_layout()
# plt.savefig('Graphs/8.jpeg', dpi=300, bbox_inches='tight')
plt.show()

data1['made'] = np.where(data1['shot_made_flag'] == 1.0, 'Made', 'Missed')


# Function to draw court and plot Bryant's shots
# Function obtained from external source
# http://savvastjortjoglou.com/nba-shot-sharts.html
def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

#8
from matplotlib.colors import ListedColormap
made_array = np.where(data1['made'] == 'Made', 1,0)
colours1 = ListedColormap(['r','g'])


plt.figure(figsize=(12,11))
scat1=plt.scatter(data1.loc_x, data1.loc_y, c=made_array,cmap=colours1)
draw_court(outer_lines=True)
# Descending values along the axis from left to right
plt.xlim(300,-300)
plt.ylim(-100,500)
plt.title('location vs shot_made_flag')
plt.legend(handles=scat1.legend_elements()[0], labels=['missed','made'],loc='upper right')
# plt.savefig('../Graphs/ShotChart_v1_draft.jpeg', dpi=300, bbox_inches='tight')
plt.show()

print(data1.shot_zone_area.unique())

#9
shot_zone_dict = {'Left Side(L)': 'red',
                  'Left Side Center(LC)': 'orange',
                  'Center(C)': 'purple',
                  'Right Side Center(RC)': 'green',
                  'Right Side(R)': 'blue',
                  'Back Court(BC)': 'grey'
                  }

plt.figure(figsize=(12,11))
plt.scatter(data1.loc_x, data1.loc_y, c=data1['shot_zone_area'].map(shot_zone_dict))
draw_court(outer_lines=True)
# Descending values along the axis from left to right
plt.xlim(300,-300)
plt.ylim(-100,500)
plt.title('location vs shot_zone_area')
# plt.savefig('../Graphs/ShotChart_v2_draft.jpeg', dpi=300, bbox_inches='tight')
plt.show()

#10
shot_zone_basic_dict = {
    'Mid-Range':'blue',
    'Restricted Area':'red',
    'In The Paint (Non-RA)':'orange',
    'Above the Break 3':'green',
    'Right Corner 3':'lightgreen',
    'Backcourt':'grey',
    'Left Corner 3':'greenyellow'

}

plt.figure(figsize=(12,11))
plt.scatter(data1.loc_x, data1.loc_y, c=data1['shot_zone_basic'].map(shot_zone_basic_dict))
draw_court(outer_lines=True)
# Descending values along the axis from left to right
plt.xlim(300,-300)
plt.ylim(-100,500)
plt.title('location vs shot_zone_basic')
# plt.savefig('../Graphs/ShotChart_v3_draft.jpeg', dpi=300, bbox_inches='tight')
plt.show()


datamost=data1[(data1['shot_zone_area']=='Center(C)') & ((data1['shot_zone_basic']=='Mid-Range') | (data1['shot_zone_basic']=='Restricted Area'))]

plt.figure(figsize=(12,11))
plt.scatter(datamost.loc_x, datamost.loc_y)
# plt.scatter(data1.loc_x, data1.loc_y, c=data1['shot_zone_basic'].map(shot_zone_basic_dict))
draw_court(outer_lines=True)
# Descending values along the axis from left to right
plt.xlim(300,-300)
plt.ylim(-100,500)
plt.title('most effective area')
plt.savefig('../Graphs/ShotChart_v6_draft.jpeg', dpi=300, bbox_inches='tight')
plt.show()

#10
# Add hex to aggregate shot values
def create_court(ax, color):
    # Short corner 3PT lines
    ax.plot([-220, -220], [0, 140], linewidth=2, color=color)
    ax.plot([220, 220], [0, 140], linewidth=2, color=color)

    # 3PT Arc
    ax.add_artist(Arc((0, 140), 440, 315, theta1=0, theta2=180, facecolor='none', edgecolor=color, lw=2))

    # Lane and Key
    ax.plot([-80, -80], [0, 190], linewidth=2, color=color)
    ax.plot([80, 80], [0, 190], linewidth=2, color=color)
    ax.plot([-60, -60], [0, 190], linewidth=2, color=color)
    ax.plot([60, 60], [0, 190], linewidth=2, color=color)
    ax.plot([-80, 80], [190, 190], linewidth=2, color=color)
    ax.add_artist(Circle((0, 190), 60, facecolor='none', edgecolor=color, lw=2))

    # Rim
    ax.add_artist(Circle((0, 60), 15, facecolor='none', edgecolor=color, lw=2))

    # Backboard
    ax.plot([-30, 30], [40, 40], linewidth=2, color=color)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set axis limits
    ax.set_xlim(-250, 250)
    ax.set_ylim(0, 470)

    return ax

rcParams['font.family'] = 'Avenir'
rcParams['font.size'] = 18
rcParams['axes.linewidth'] = 2


# Create figure and axes
fig = plt.figure(figsize=(4*2, 3.76*2)) # Was 4, 3.76
ax = fig.add_axes([0, 0, 1, 1])

# Draw court
ax = create_court(ax, 'black')

# Plot hexbin of shots
ax.hexbin(data1['loc_x'], data1['loc_y'] + 60,
          gridsize=(50, 50),
          extent=(-300, 300, 0, 940),
          bins=12,
          C=data1['shot_made_flag'],
          reduce_C_function=np.mean,
          cmap='RdYlGn')

# Annotate player name and season
ax.text(0, 1.05, 'Kobe Bryant\nCareer Shooting', transform=ax.transAxes, ha='left', va='baseline')

# Save and show figure
# plt.savefig('../Graphs/ShotChart_v4_draft.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------
### Pre-Processing ###

# Add avg field goal percentage by distance - do (1) bins and (2) continuous line
#pd.crosstab(data1.shot_distance, data1.made).apply(lambda r: r/r.sum(), axis=1)
#pd.crosstab(data1.shot_distance, data1.made).apply(lambda r: r.sum(), axis=1)
distances = pd.crosstab(data1.shot_distance, data1.made).apply(lambda r: r.sum(), axis=1).index
distance_attempts = pd.crosstab(data1.shot_distance, data1.made).apply(lambda r: r.sum(), axis=1)
distance_percent = pd.crosstab(data1.shot_distance, data1.made).apply(lambda r: r/r.sum(), axis=1)['Made']

fig, ax1 = plt.subplots()

color1 = 'tab:red'
ax1.set_xlabel('Distance from basket')
ax1.set_ylabel('Attempts', color=color1)
ax1.plot(distances, distance_attempts, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color2 = 'tab:blue'
ax2.set_ylabel('Field Goal Percentage', color=color2)  # we already handled the x-label with ax1
ax2.plot(distances, distance_percent, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# Features to consider
# ['action_type', 'combined_shot_type', 'period', 'playoffs',
#        'season', 'shot_distance', 'shot_made_flag',
#        'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
#        'made']

# Look for variation in field goal percentage in the feature values
# If a feature has that variation, we should add it to the model
print('crosstab results:')
# May or may not use action type since it has too many values, many with small sample
print(pd.crosstab(data1.action_type, data1.made))
print(pd.crosstab(data1.action_type, data1.made).apply(lambda r: r/r.sum(), axis=1))

# We will use combined shot type since it has more sample in each category and has variation
print(pd.crosstab(data1.combined_shot_type, data1.made))
print(pd.crosstab(data1.combined_shot_type, data1.made).apply(lambda r: r/r.sum(), axis=1))

# We will use period variable
print(pd.crosstab(data1.period, data1.made))
print(pd.crosstab(data1.period, data1.made).apply(lambda r: r/r.sum(), axis=1))

# We will may use the playoffs variable, but likelihood of making shot is nearly same in both classes
print(pd.crosstab(data1.playoffs, data1.made))
print(pd.crosstab(data1.playoffs, data1.made).apply(lambda r: r/r.sum(), axis=1))

# We will use season variable, although there is very little variation between seasons
print(pd.crosstab(data1.season, data1.made))
print(pd.crosstab(data1.season, data1.made).apply(lambda r: r/r.sum(), axis=1))

# We will use shot type, large variation between two pointers and three pointers
print(pd.crosstab(data1.shot_type, data1.made))
print(pd.crosstab(data1.shot_type, data1.made).apply(lambda r: r/r.sum(), axis=1))

# We will use shot zone area, percentages vary greatly between classes
print(pd.crosstab(data1.shot_zone_area, data1.made))
print(pd.crosstab(data1.shot_zone_area, data1.made).apply(lambda r: r/r.sum(), axis=1))

# We will use shot zone basic, it has variation between groups
print(pd.crosstab(data1.shot_zone_basic, data1.made))
print(pd.crosstab(data1.shot_zone_basic, data1.made).apply(lambda r: r/r.sum(), axis=1))

# will use shot zone range, it has variation between groups
print(pd.crosstab(data1.shot_zone_range, data1.made))
print(pd.crosstab(data1.shot_zone_range, data1.made).apply(lambda r: r/r.sum(), axis=1))

# Drop all columns that aren't going into the model
data1.drop(['game_event_id', 'game_id', 'lat',
            'loc_x', 'loc_y', 'lon', 'minutes_remaining',
            'seconds_remaining',
            'team_id', 'team_name', 'game_date', 'matchup', 'opponent', 'shot_id',
            'made'], axis=1, inplace=True)

# Change shot distance max to 40 to reduce variable range caused by outliers
data1['shot_distance'] = np.where(data1['shot_distance'] >= 40, 40, data1['shot_distance'])

# Replace some columns in anticipation of encoder
# Encoder works in alphabetical order
data1['combined_shot_type'].replace('Jump Shot', 'EE', inplace=True)
data1['combined_shot_type'].replace('Layup', 'BB', inplace=True)
data1['combined_shot_type'].replace('Dunk', 'AA', inplace=True)
data1['combined_shot_type'].replace('Tip Shot', 'CC', inplace=True)
data1['combined_shot_type'].replace('Hook Shot', 'DD', inplace=True)


data1['shot_zone_area'].replace('Center(C)', 'CC', inplace=True)
data1['shot_zone_area'].replace('Right Side Center(RC)', 'DD', inplace=True)
data1['shot_zone_area'].replace('Right Side(R)', 'EE', inplace=True)
data1['shot_zone_area'].replace('Left Side Center(LC)', 'BB', inplace=True)
data1['shot_zone_area'].replace('Left Side(L)', 'AA', inplace=True)

data1['shot_zone_basic'].replace('Restricted Area', 'AA', inplace=True)
data1['shot_zone_basic'].replace('In The Paint (Non-RA)', 'BB', inplace=True)
data1['shot_zone_basic'].replace('Mid-Range', 'CC', inplace=True)
data1['shot_zone_basic'].replace('Left Corner 3', 'DD', inplace=True)
data1['shot_zone_basic'].replace('Above the Break 3', 'EE', inplace=True)
data1['shot_zone_basic'].replace('Right Corner 3', 'FF', inplace=True)
data1['shot_zone_basic'].replace('Backcourt', 'GG', inplace=True)

data1['shot_zone_range'].replace('Less Than 8 ft.', 'AA', inplace=True)
data1['shot_zone_range'].replace('8-16 ft.', 'BB', inplace=True)
data1['shot_zone_range'].replace('16-24 ft.', 'CC', inplace=True)
data1['shot_zone_range'].replace('24+ ft.', 'DD', inplace=True)
data1['shot_zone_range'].replace('Back Court Shot', 'EE', inplace=True)


# Separate features and target
X = data1.drop(['shot_made_flag'], axis=1)
y = data1['shot_made_flag']

# Transform variables
# ['action_type', 'combined_shot_type', 'period', 'playoffs',
#        'season', 'shot_distance'
#        'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
#        ]
# Shot made flag needs to be label encoded
le = LabelEncoder()
mm = MinMaxScaler()

# May or may not use action type since it has too many values, many with small sample
# Decide to use it, as neural networks aren't as affected by curse of dimensionality
# Label encode it and scale 0 to 1
# Do this for the other columns as well
columns_to_transform =  ['action_type', 'combined_shot_type', 'period', 'playoffs',
                         'season', 'shot_distance'
                                   'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
                         ]

for c in X.columns:
    X[c] = le.fit_transform(X[c])
    X[c] = mm.fit_transform(X[c].values.reshape(-1,1))


# Label encode y
y = le.fit_transform(y)
print(y)

# Correlation matrix of features to output
# TO DO

# ----------------------------------------
### Modeling ###
## Random Forest
# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
#%%-----------------------------------------------------------------------
#perform training with random forest with all columns
# specify random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
clf = RandomForestClassifier(n_estimators=10,random_state=100)

# perform training
clf.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
#plot feature importances
# get feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, data1.loc[:,data1.columns!='shot_made_flag'].columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# show the plot
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------
#select features to perform training with random forest with k columns
# select the training dataset on k-features
# featuK=clf.feature_importances_.argsort()[::-1][:5]
featuK=f_importances.index[:6]
newX_train = X_train.loc[:,featuK]

# select the testing dataset on k-features
newX_test = X_test.loc[:, featuK]

#%%-----------------------------------------------------------------------
#perform training with random forest with k columns
# specify random forest classifier
clf_k_features = RandomForestClassifier(n_estimators=10,random_state=100)

# train the model
clf_k_features.fit(newX_train, y_train)

#%%-----------------------------------------------------------------------
#make predictions

# predicton on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

# prediction on test using k features
y_pred_k_features = clf_k_features.predict(newX_test)
y_pred_k_features_score = clf_k_features.predict_proba(newX_test)

# %%-----------------------------------------------------------------------
# calculate metrics for all features

print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

roc_score1=roc_auc_score(y_test,y_pred_score[:,1]) * 100
print("ROC_AUC : ", roc_score1)
print("\n")
print('confusion matrix - all features-RF')
print(confusion_matrix(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test,y_pred_score[:,1])

fig,ax=plt.subplots()
ax.plot(fpr, tpr, color='#90EE90', lw=3,
        label='ROC curve (area = %0.2f)' % roc_score1)
ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
ax.set_title('ROC of all features-RF')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


# calculate metrics Using K features
print("\n")
print("Results Using K features: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_k_features))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)
print("\n")
roc_score1=roc_auc_score(y_test,y_pred_k_features_score[:,1]) * 100
print("ROC_AUC : ", roc_score1)
print("\n")
print('confusion matrix - for K features-RF')
print(confusion_matrix(y_test, y_pred_k_features))

fpr, tpr, _ = roc_curve(y_test,y_pred_k_features_score[:,1])

fig,ax=plt.subplots()
ax.plot(fpr, tpr, color='#90EE90', lw=3,
        label='ROC curve (area = %0.2f)' % roc_score1)
ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
ax.set_title('ROC of for K features-RF')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# for this model - n_estimators=10, randomstate=100
# we can observe that the accuracy score and ROC_AUC increased slightly when we use feature reduction so we can build
# model with only 6 features - ['season', 'shot_distance', 'action_type', 'period', 'shot_zone_area','combined_shot_type']
# and from confusion matrix we can observe that true negatives increased  and true positives decreased very slightly.
# so the model with only six features is best model


# Train-test Split
X_train_NN, X_test_NN, y_train_NN, y_test_NN = train_test_split(X, y, random_state=100)

clf_NN = MLPClassifier(random_state=100, max_iter=200).fit(X_train_NN, y_train_NN)

y_pred_NN = clf_NN.predict(X_test_NN)
y_pred_score_NN = clf_NN.predict_proba(X_test_NN)

print("\n")
print('MLP modelling')
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test_NN,y_pred_NN))
print("\n")

print("Mean squared error : ",mean_squared_error(y_test_NN, y_pred_NN))
print("\n")

roc_score1=roc_auc_score(y_test_NN,y_pred_score_NN[:,1]) * 100
print("ROC_AUC : ", roc_score1)
print("\n")
print('confusion matrix - default - MLP')
print(confusion_matrix(y_test_NN, y_pred_NN))

fpr, tpr, _ = roc_curve(y_test_NN,y_pred_score_NN[:,1])

fig,ax=plt.subplots()
ax.plot(fpr, tpr, color='#90EE90', lw=3,
        label='ROC curve (area = %0.2f)' % roc_score1)
ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
ax.set_title('ROC of default - MLP')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


#%%-----------------------------------------------------------------------
#select features to perform training with random forest with k columns
# select the training dataset on k-features
# featuK=clf.feature_importances_.argsort()[::-1][:5]
# featuK=f_importances.index[:6]
newX_train_NN = X_train_NN.loc[:,featuK]

# select the testing dataset on k-features
newX_test_NN = X_test_NN.loc[:, featuK]

#%%-----------------------------------------------------------------------
#perform training with random forest with k columns
# specify MLP classifier and train
clf_k_features_NN = MLPClassifier(random_state=100, max_iter=200).fit(newX_train_NN, y_train_NN)

#%%-----------------------------------------------------------------------

# prediction on test using k features
y_pred_k_features_NN = clf_k_features_NN.predict(newX_test_NN)
y_pred_k_features_score_NN = clf_k_features_NN.predict_proba(newX_test_NN)

# calculate metrics Using K features
print("\n")
print("Results Using K features: \n")
print("Classification Report: ")
print(classification_report(y_test_NN,y_pred_k_features_NN))
print("\n")
print("Mean squared error : ", mean_squared_error(y_test_NN, y_pred_k_features_NN))
print("\n")

roc_score1=roc_auc_score(y_test_NN,y_pred_k_features_score_NN[:,1]) * 100
print("ROC_AUC : ", roc_score1)
print("\n")
print('confusion matrix - default - MLP')
print(confusion_matrix(y_test_NN, y_pred_k_features_NN))

fpr, tpr, _ = roc_curve(y_test_NN,y_pred_k_features_score_NN[:,1])

fig,ax=plt.subplots()
ax.plot(fpr, tpr, color='#90EE90', lw=3,
        label='ROC curve (area = %0.2f)' % roc_score1)
ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
ax.set_title('ROC of K features - MLP')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


# for this model MLP- random_state=100, max_iter=200
# we can observe that the accuracy score and ROC_AUC decreased slightly when we use feature reduction so we can build
# model with only 6 features - ['season', 'shot_distance', 'action_type', 'period', 'shot_zone_area','combined_shot_type']
# and from confusion matrix we can observe that true negatives increased  and true positives decreased.
# so the model with only six features is not best model - select the model with all the features
# though the MLP model has better accuracy,roc and high number of true negatives than the random forest, the true poitives
# have decreased from the random forest, since the main goal of this analysis is to predict the shots made, MLP model doesn't hold
# to random forest model

#parameter changes
# hidden_layer_sizes=(100,100)
clf_NN1 = MLPClassifier(hidden_layer_sizes=(100,100),random_state=100, max_iter=200).fit(X_train_NN, y_train_NN)

y_pred_NN1 = clf_NN1.predict(X_test_NN)
y_pred_score_NN1 = clf_NN1.predict_proba(X_test_NN)

print("\n")
print('MLP modelling, hidden_layer_sizes=(100,100)')
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test_NN,y_pred_NN1))
print("\n")

print("Mean squared error : ",mean_squared_error(y_test_NN, y_pred_NN1))
print("\n")

roc_score1=roc_auc_score(y_test_NN,y_pred_score_NN1[:,1]) * 100
print("ROC_AUC : ", roc_score1)
print("\n")
print('confusion matrix - hidden_layers=(100,100)')
print(confusion_matrix(y_test_NN, y_pred_NN1))

fpr, tpr, _ = roc_curve(y_test_NN,y_pred_score_NN1[:,1])

fig,ax=plt.subplots()
ax.plot(fpr, tpr, color='#90EE90', lw=3,
        label='ROC curve (area = %0.2f)' % roc_score1)
ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
ax.set_title('ROC of hidden_layers=(100,100)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# hidden_layer_sizes=(100,100,100)
clf_NN2 = MLPClassifier(hidden_layer_sizes=(100,100,100),random_state=100, max_iter=200).fit(X_train_NN, y_train_NN)

y_pred_NN2 = clf_NN2.predict(X_test_NN)
y_pred_score_NN2 = clf_NN2.predict_proba(X_test_NN)

print("\n")
print('MLP modelling, hidden_layer_sizes=(100,100,100)')
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test_NN,y_pred_NN2))
print("\n")

print("Mean squared error : ",mean_squared_error(y_test_NN, y_pred_NN2))
print("\n")

roc_score1=roc_auc_score(y_test_NN,y_pred_score_NN2[:,1]) * 100
print("ROC_AUC : ", roc_score1)
print("\n")
print('confusion matrix - hidden_layers=(100,100,100)')
print(confusion_matrix(y_test_NN, y_pred_NN2))

fpr, tpr, _ = roc_curve(y_test_NN,y_pred_score_NN2[:,1])

fig,ax=plt.subplots()
ax.plot(fpr, tpr, color='#90EE90', lw=3,
        label='ROC curve (area = %0.2f)' % roc_score1)
ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
ax.set_title('ROC of hidden_layers=(100,100,100)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


# hidden_layer_sizes=(100,100,100,100)
clf_NN3 = MLPClassifier(hidden_layer_sizes=(100,100,100,100),random_state=100, max_iter=200).fit(X_train_NN, y_train_NN)

y_pred_NN3 = clf_NN3.predict(X_test_NN)
y_pred_score_NN3 = clf_NN3.predict_proba(X_test_NN)

print("\n")
print('MLP modelling, hidden_layer_sizes=(100,100,100,100)')
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test_NN,y_pred_NN3))
print("\n")

print("Mean squared error : ",mean_squared_error(y_test_NN, y_pred_NN3))
print("\n")

roc_score1=roc_auc_score(y_test_NN,y_pred_score_NN3[:,1]) * 100
print("ROC_AUC : ", roc_score1)
print("\n")
print('confusion matrix - hidden_layers=(100,100,100,100)')
print(confusion_matrix(y_test_NN, y_pred_NN3))

fpr, tpr, _ = roc_curve(y_test_NN,y_pred_score_NN3[:,1])

fig,ax=plt.subplots()
ax.plot(fpr, tpr, color='#90EE90', lw=3,
        label='ROC curve (area = %0.2f)' % roc_score1)
ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
ax.set_title('ROC of hidden_layers=(100,100,100,100)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


# activation='logistic'
clf_NN4 = MLPClassifier(activation='logistic',random_state=100, max_iter=200).fit(X_train_NN, y_train_NN)

y_pred_NN4 = clf_NN4.predict(X_test_NN)
y_pred_score_NN4 = clf_NN4.predict_proba(X_test_NN)

print("\n")
print('MLP modelling, activation=logistic')
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test_NN,y_pred_NN4))
print("\n")

print("Mean squared error : ",mean_squared_error(y_test_NN, y_pred_NN4))
print("\n")

roc_score1=roc_auc_score(y_test_NN,y_pred_score_NN4[:,1]) * 100
print("ROC_AUC : ", roc_score1)
print("\n")
print('confusion matrix - activation=logistic ')
print(confusion_matrix(y_test_NN, y_pred_NN4))

fpr, tpr, _ = roc_curve(y_test_NN,y_pred_score_NN4[:,1])

fig,ax=plt.subplots()
ax.plot(fpr, tpr, color='#90EE90', lw=3,
        label='ROC curve (area = %0.2f)' % roc_score1)
ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
ax.set_title('ROC of activation=logistic ')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


# solver='sgd'
clf_NN5 = MLPClassifier(solver='sgd',random_state=100, max_iter=200).fit(X_train_NN, y_train_NN)

y_pred_NN5 = clf_NN5.predict(X_test_NN)
y_pred_score_NN5 = clf_NN5.predict_proba(X_test_NN)

print("\n")
print('MLP modelling, solver=sgd')
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test_NN,y_pred_NN5))
print("\n")

print("Mean squared error : ",mean_squared_error(y_test_NN, y_pred_NN5))
print("\n")

roc_score1=roc_auc_score(y_test_NN,y_pred_score_NN5[:,1]) * 100
print("ROC_AUC : ", roc_score1)
print("\n")
print('confusion matrix - solver=sgd')
print(confusion_matrix(y_test_NN, y_pred_NN5))

fpr, tpr, _ = roc_curve(y_test_NN,y_pred_score_NN5[:,1])

fig,ax=plt.subplots()
ax.plot(fpr, tpr, color='#90EE90', lw=3,
        label='ROC curve (area = %0.2f)' % roc_score1)
ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
ax.set_title('ROC of solver=sgd')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


# early_stopping=True
clf_NN6 = MLPClassifier(early_stopping=True,random_state=100, max_iter=200).fit(X_train_NN, y_train_NN)

y_pred_NN6 = clf_NN6.predict(X_test_NN)
y_pred_score_NN6 = clf_NN6.predict_proba(X_test_NN)

print("\n")
print('MLP modelling, early_stopping=True')
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test_NN,y_pred_NN6))
print("\n")

print("Mean squared error : ",mean_squared_error(y_test_NN, y_pred_NN6))
print("\n")

roc_score1=roc_auc_score(y_test_NN,y_pred_score_NN6[:,1]) * 100
print("ROC_AUC : ", roc_score1)
print("\n")
print('confusion matrix - early_stopping=True')
print(confusion_matrix(y_test_NN, y_pred_NN6))

fpr, tpr, _ = roc_curve(y_test_NN,y_pred_score_NN6[:,1])

fig,ax=plt.subplots()
ax.plot(fpr, tpr, color='#90EE90', lw=3,
        label='ROC curve (area = %0.2f)' % roc_score1)
ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
ax.set_title('ROC of early_stopping=True')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


# momentum=0.95
clf_NN7 = MLPClassifier(momentum=0.95,random_state=100, max_iter=200).fit(X_train_NN, y_train_NN)

y_pred_NN7 = clf_NN7.predict(X_test_NN)
y_pred_score_NN7 = clf_NN7.predict_proba(X_test_NN)

print("\n")
print('MLP modelling, momentum=0.95')
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test_NN,y_pred_NN7))
print("\n")

print("Mean squared error : ",mean_squared_error(y_test_NN, y_pred_NN7))
print("\n")

roc_score1=roc_auc_score(y_test_NN,y_pred_score_NN7[:,1]) * 100
print("ROC_AUC : ", roc_score1)
print("\n")
print('confusion matrix - momentum=0.95')
print(confusion_matrix(y_test_NN, y_pred_NN7))

fpr, tpr, _ = roc_curve(y_test_NN,y_pred_score_NN7[:,1])

fig,ax=plt.subplots()
ax.plot(fpr, tpr, color='#90EE90', lw=3,
        label='ROC curve (area = %0.2f)' % roc_score1)
ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
ax.set_title('ROC of momentum=0.95')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


# activation='logistic',solver='sgd',early_stopping=True,momentum=0.95,hidden_layer_sizes=(100,100,100)
clf_NN8 = MLPClassifier(activation='logistic',solver='sgd',early_stopping=True,momentum=0.95,hidden_layer_sizes=(100,100,100),random_state=100, max_iter=200).fit(X_train_NN, y_train_NN)

y_pred_NN8 = clf_NN8.predict(X_test_NN)
y_pred_score_NN8 = clf_NN8.predict_proba(X_test_NN)

print("\n")
print('MLP modelling, many changes')
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test_NN,y_pred_NN8))
print("\n")

print("Mean squared error : ",mean_squared_error(y_test_NN, y_pred_NN8))
print("\n")

roc_score1=roc_auc_score(y_test_NN,y_pred_score_NN8[:,1]) * 100
print("ROC_AUC : ", roc_score1)
print("\n")
print('confusion matrix - many changes')
print(confusion_matrix(y_test_NN, y_pred_NN8))

fpr, tpr, _ = roc_curve(y_test_NN,y_pred_score_NN8[:,1])

fig,ax=plt.subplots(figsize=(5,5))
ax.plot(fpr, tpr, color='#90EE90', lw=3,
        label='ROC curve (area = %0.2f)' % roc_score1)
ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--')
ax.set_title('ROC of many changes')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()









