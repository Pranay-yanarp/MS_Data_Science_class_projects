
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib import rcParams
pd.options.mode.chained_assignment = None  # default='warn'

data=pd.read_csv('data.csv')

data1=data.dropna()

data1.loc[data1['shot_type']=='2PT Field Goal','shot_type']=2
data1.loc[data1['shot_type']=='3PT Field Goal','shot_type']=3

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# 2002 analysis
# Against which team did he best/worst perform
data2002=data1[data1['season']=='2001-02']
opponent_teams=data2002['opponent'].unique()
opp_points=[]

for i in opponent_teams:
    points=data2002.loc[data2002['opponent']==i,'shot_made_flag'] * data2002.loc[data2002['opponent']==i,'shot_type']
    matches=data2002.loc[data2002['opponent']==i,'game_id'].unique().size
    print('team =',i,'points = ',points.sum(),'matches =',matches)
    opp_points.append([i,points.sum()/matches])

opp_points_df=pd.DataFrame(opp_points,columns=['team','avg'], dtype=float)

opp_points_df_sort=opp_points_df.sort_values(by='avg',ascending=False)

plt.figure(figsize=(10,8))
plots=sns.barplot(opp_points_df_sort.head(5)['team'],opp_points_df_sort.head(5)['avg'])
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')

plt.title('barplot of avg score vs 5 opponent teams - that kobe averaged highest')
# plt.savefig('Graphs2002/2002_1_1.jpeg', dpi=300, bbox_inches='tight')
plt.show()
lenght1=opp_points_df_sort['team'].size
plt.figure(figsize=(10,8))
plots=sns.barplot(opp_points_df_sort.tail(5)['team'],opp_points_df_sort.tail(5)['avg'])
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.title('barplot of avg score vs 5 opponent teams - that kobe averaged least')
# plt.savefig('Graphs2002/2002_1_2.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# 2 most effective action types against highest and lowest averaged teams

opp_points_df_sort.reset_index(inplace=True)
HA2teams=opp_points_df_sort.head(2)['team']
LA2teams=opp_points_df_sort.tail(2)['team']

data2002_yes=data2002[data2002['shot_made_flag']==1]

print('most effective action types against highest and lowest averaged teams')
print('for 2 teams kobe averaged highest')

for j in HA2teams:
    actions2002=list(data2002.loc[data2002['opponent']==j,'action_type'].unique())
    total_size=data2002.loc[data2002['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2002_yes.loc[(data2002_yes['opponent']==j) & (data2002_yes['action_type']==i),'shot_made_flag'].size
        total_size1=data2002.loc[(data2002['opponent']==j) & (data2002['action_type']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2[:5],columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of top 5 action_type for '+j+'- 2002')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2002/2002_2-2_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


print('for 2 teams kobe averaged lowest')
for j in LA2teams:
    actions2002=list(data2002.loc[data2002['opponent']==j,'action_type'].unique())
    total_size=data2002.loc[data2002['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2002_yes.loc[(data2002_yes['opponent']==j) & (data2002_yes['action_type']==i),'shot_made_flag'].size
        total_size1=data2002.loc[(data2002['opponent']==j) & (data2002['action_type']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2[:5],columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of top 5 action_type for '+j+'- 2002')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2002/2002_2-4_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()

# 3 most effective shot zone area against highest and lowest averaged teams
print('most effective shot zone area against highest and lowest averaged teams')
print('for 2 teams kobe averaged highest')
for j in HA2teams:
    actions2002=list(data2002.loc[data2002['opponent']==j,'shot_zone_area'].unique())
    total_size=data2002.loc[data2002['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2002_yes.loc[(data2002_yes['opponent']==j) & (data2002_yes['shot_zone_area']==i),'shot_made_flag'].size
        total_size1=data2002.loc[(data2002['opponent']==j) & (data2002['shot_zone_area']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of shot_zone_area for '+j+'- 2002')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2002/2002_3-1_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


print('for 2 teams kobe averaged lowest')
for j in LA2teams:
    actions2002=list(data2002.loc[data2002['opponent']==j,'shot_zone_area'].unique())
    total_size=data2002.loc[data2002['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2002_yes.loc[(data2002_yes['opponent']==j) & (data2002_yes['shot_zone_area']==i),'shot_made_flag'].size
        total_size1=data2002.loc[(data2002['opponent']==j) & (data2002['shot_zone_area']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of shot_zone_area for '+j+'- 2002')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2002/2002_3-2_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()

# 4 most effective shot zone basic against highest and lowest averaged teams
print('most effective shot zone basic against highest and lowest averaged teams')
print('for 2 teams kobe averaged highest')
for j in HA2teams:
    actions2002=list(data2002.loc[data2002['opponent']==j,'shot_zone_basic'].unique())
    total_size=data2002.loc[data2002['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2002_yes.loc[(data2002_yes['opponent']==j) & (data2002_yes['shot_zone_basic']==i),'shot_made_flag'].size
        total_size1=data2002.loc[(data2002['opponent']==j) & (data2002['shot_zone_basic']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of shot_zone_basic for '+j+'- 2002')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2002/2002_4-1_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


print('for 2 teams kobe averaged lowest')
for j in LA2teams:
    actions2002=list(data2002.loc[data2002['opponent']==j,'shot_zone_basic'].unique())
    total_size=data2002.loc[data2002['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2002_yes.loc[(data2002_yes['opponent']==j) & (data2002_yes['shot_zone_basic']==i),'shot_made_flag'].size
        total_size1=data2002.loc[(data2002['opponent']==j) & (data2002['shot_zone_basic']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of shot_zone_basic for '+j+'- 2002')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2002/2002_4-2_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()

# 5 most effective shot_zone_range against highest and lowest averaged teams
print('most effective shot_zone_range against highest and lowest averaged teams')
print('for 2 teams kobe averaged highest')
for j in HA2teams:
    actions2002=list(data2002.loc[data2002['opponent']==j,'shot_zone_range'].unique())
    total_size=data2002.loc[data2002['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2002_yes.loc[(data2002_yes['opponent']==j) & (data2002_yes['shot_zone_range']==i),'shot_made_flag'].size
        total_size1=data2002.loc[(data2002['opponent']==j) & (data2002['shot_zone_range']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of shot_zone_range for '+j+'- 2002')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2002/2002_5-1_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


print('for 2 teams kobe averaged lowest')
for j in LA2teams:
    actions2002=list(data2002.loc[data2002['opponent']==j,'shot_zone_range'].unique())
    total_size=data2002.loc[data2002['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2002_yes.loc[(data2002_yes['opponent']==j) & (data2002_yes['shot_zone_range']==i),'shot_made_flag'].size
        total_size1=data2002.loc[(data2002['opponent']==j) & (data2002['shot_zone_range']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of shot_zone_range for '+j+'- 2002')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2002/2002_5-2_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()

# 6 most effective period against highest and lowest averaged teams
print('most effective period against highest and lowest averaged teams')
print('for 2 teams kobe averaged highest')
for j in HA2teams:
    actions2002=list(data2002.loc[data2002['opponent']==j,'period'].unique())
    total_size=data2002.loc[data2002['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2002_yes.loc[(data2002_yes['opponent']==j) & (data2002_yes['period']==i),'shot_made_flag'].size
        total_size1=data2002.loc[(data2002['opponent']==j) & (data2002['period']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=10, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1

    l1=np.array(list(actions2002))-1
    ax2.plot(l1, plot_df1['eff'], 'b-')
    ax2.scatter(l1, plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    # ax2.set_ylim([20,50])
    plt.title('shots scored vs period for '+j+'- 2002')
    ax1.set_xticklabels(labels=plot_df1['type'])
    plt.tight_layout()
    # plt.savefig('Graphs2002/2002_6-1_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


print('for 2 teams kobe averaged lowest')
for j in LA2teams:
    actions2002=list(data2002.loc[data2002['opponent']==j,'period'].unique())
    total_size=data2002.loc[data2002['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2002_yes.loc[(data2002_yes['opponent']==j) & (data2002_yes['period']==i),'shot_made_flag'].size
        total_size1=data2002.loc[(data2002['opponent']==j) & (data2002['period']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=10, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1

    l1=np.array(list(actions2002))-1
    ax2.plot(l1, plot_df1['eff'], 'b-')
    ax2.scatter(l1, plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    # ax2.set_ylim([20,50])
    plt.title('shots scored vs period for '+j+'- 2002')
    ax1.set_xticklabels(labels=plot_df1['type'])
    plt.tight_layout()
    # plt.savefig('Graphs2002/2002_6-2_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()

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

for j in HA2teams:
    cmade={1:'green',0:'red'}
    plt.figure(figsize=(12,11))
    plt.scatter(data2002.loc[data2002['opponent']==j,'loc_x'], data2002.loc[data2002['opponent']==j,'loc_y'], c=data2002.loc[data2002['opponent']==j,'shot_made_flag'].map(cmade))
    draw_court(outer_lines=True)
    # Descending values along the axis from left to right
    plt.xlim(300,-300)
    plt.ylim(-100,500)
    plt.title('location vs shot_made_flag -'+j)
    # plt.savefig('Graphs2002/ShotChart_v1 for'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


for j in LA2teams:
    cmade={1:'green',0:'red'}
    plt.figure(figsize=(12,11))
    plt.scatter(data2002.loc[data2002['opponent']==j,'loc_x'], data2002.loc[data2002['opponent']==j,'loc_y'], c=data2002.loc[data2002['opponent']==j,'shot_made_flag'].map(cmade))
    draw_court(outer_lines=True)
    # Descending values along the axis from left to right
    plt.xlim(300,-300)
    plt.ylim(-100,500)
    plt.title('location vs shot_made_flag -'+j)
    # plt.savefig('Graphs2002/ShotChart_v1 '+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
data2009=data1[data1['season']=='2008-09']
opponent_teams=data2009['opponent'].unique()
opp_points=[]

for i in opponent_teams:
    points=data2009.loc[data2009['opponent']==i,'shot_made_flag'] * data2009.loc[data2009['opponent']==i,'shot_type']
    matches=data2009.loc[data2009['opponent']==i,'game_id'].unique().size
    print('team =',i,'points = ',points.sum(),'matches =',matches)
    opp_points.append([i,points.sum()/matches])

opp_points_df=pd.DataFrame(opp_points,columns=['team','avg'], dtype=float)

opp_points_df_sort=opp_points_df.sort_values(by='avg',ascending=False)

plt.figure(figsize=(10,8))
plots=sns.barplot(opp_points_df_sort.head(5)['team'],opp_points_df_sort.head(5)['avg'])
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')

plt.title('barplot of avg score vs 5 opponent teams - that kobe averaged highest')
# plt.savefig('Graphs2009/2002_1_1.jpeg', dpi=300, bbox_inches='tight')
plt.show()
lenght1=opp_points_df_sort['team'].size
plt.figure(figsize=(10,8))
plots=sns.barplot(opp_points_df_sort.tail(5)['team'],opp_points_df_sort.tail(5)['avg'])
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.title('barplot of avg score vs 5 opponent teams - that kobe averaged least')
# plt.savefig('Graphs2009/2002_1_2.jpeg', dpi=300, bbox_inches='tight')
plt.show()

# 2 most effective action types against highest and lowest averaged teams

opp_points_df_sort.reset_index(inplace=True)
HA2teams=opp_points_df_sort.head(2)['team']
LA2teams=opp_points_df_sort.tail(2)['team']

data2009_yes=data2009[data2009['shot_made_flag']==1]

print('most effective action types against highest and lowest averaged teams')
print('for 2 teams kobe averaged highest')

for j in HA2teams:
    actions2002=list(data2009.loc[data2009['opponent']==j,'action_type'].unique())
    total_size=data2009.loc[data2009['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2009_yes.loc[(data2009_yes['opponent']==j) & (data2009_yes['action_type']==i),'shot_made_flag'].size
        total_size1=data2009.loc[(data2009['opponent']==j) & (data2009['action_type']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2[:5],columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of top 5 action_type for '+j+'- 2009')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2009/2002_2-2_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


print('for 2 teams kobe averaged lowest')
for j in LA2teams:
    actions2002=list(data2009.loc[data2009['opponent']==j,'action_type'].unique())
    total_size=data2009.loc[data2009['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2009_yes.loc[(data2009_yes['opponent']==j) & (data2009_yes['action_type']==i),'shot_made_flag'].size
        total_size1=data2009.loc[(data2009['opponent']==j) & (data2009['action_type']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2[:5],columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of top 5 action_type for '+j+'- 2009')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2009/2002_2-4_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()

# 3 most effective shot zone area against highest and lowest averaged teams
print('most effective shot zone area against highest and lowest averaged teams')
print('for 2 teams kobe averaged highest')
for j in HA2teams:
    actions2002=list(data2009.loc[data2009['opponent']==j,'shot_zone_area'].unique())
    total_size=data2009.loc[data2009['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2009_yes.loc[(data2009_yes['opponent']==j) & (data2009_yes['shot_zone_area']==i),'shot_made_flag'].size
        total_size1=data2009.loc[(data2009['opponent']==j) & (data2009['shot_zone_area']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of shot_zone_area for '+j+'- 2009')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2009/2002_3-1_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


print('for 2 teams kobe averaged lowest')
for j in LA2teams:
    actions2002=list(data2009.loc[data2009['opponent']==j,'shot_zone_area'].unique())
    total_size=data2009.loc[data2009['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2009_yes.loc[(data2009_yes['opponent']==j) & (data2009_yes['shot_zone_area']==i),'shot_made_flag'].size
        total_size1=data2009.loc[(data2009['opponent']==j) & (data2009['shot_zone_area']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of shot_zone_area for '+j+'- 2009')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2009/2009_3-2_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()

# 4 most effective shot zone basic against highest and lowest averaged teams
print('most effective shot zone basic against highest and lowest averaged teams')
print('for 2 teams kobe averaged highest')
for j in HA2teams:
    actions2002=list(data2009.loc[data2009['opponent']==j,'shot_zone_basic'].unique())
    total_size=data2009.loc[data2009['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2009_yes.loc[(data2009_yes['opponent']==j) & (data2009_yes['shot_zone_basic']==i),'shot_made_flag'].size
        total_size1=data2009.loc[(data2009['opponent']==j) & (data2009['shot_zone_basic']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of shot_zone_basic for '+j+'- 2009')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2009/2009_4-1_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


print('for 2 teams kobe averaged lowest')
for j in LA2teams:
    actions2002=list(data2009.loc[data2009['opponent']==j,'shot_zone_basic'].unique())
    total_size=data2009.loc[data2009['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2009_yes.loc[(data2009_yes['opponent']==j) & (data2009_yes['shot_zone_basic']==i),'shot_made_flag'].size
        total_size1=data2009.loc[(data2009['opponent']==j) & (data2009['shot_zone_basic']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of shot_zone_basic for '+j+'- 2009')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2009/2009_4-2_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()

# 5 most effective shot_zone_range against highest and lowest averaged teams
print('most effective shot_zone_range against highest and lowest averaged teams')
print('for 2 teams kobe averaged highest')
for j in HA2teams:
    actions2002=list(data2009.loc[data2009['opponent']==j,'shot_zone_range'].unique())
    total_size=data2009.loc[data2009['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2009_yes.loc[(data2009_yes['opponent']==j) & (data2009_yes['shot_zone_range']==i),'shot_made_flag'].size
        total_size1=data2009.loc[(data2009['opponent']==j) & (data2009['shot_zone_range']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of shot_zone_range for '+j+'- 2009')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2009/2009_5-1_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


print('for 2 teams kobe averaged lowest')
for j in LA2teams:
    actions2002=list(data2009.loc[data2009['opponent']==j,'shot_zone_range'].unique())
    total_size=data2009.loc[data2009['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2009_yes.loc[(data2009_yes['opponent']==j) & (data2009_yes['shot_zone_range']==i),'shot_made_flag'].size
        total_size1=data2009.loc[(data2009['opponent']==j) & (data2009['shot_zone_range']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    shots_percent2.sort(reverse=True)
    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=15, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1
    ax2.plot(plot_df1['type'], plot_df1['eff'], 'b-')
    ax2.scatter(plot_df1['type'], plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    plt.title('Histogram of shot_zone_range for '+j+'- 2009')
    ax1.set_xticklabels(labels=plot_df1['type'],rotation=90)
    plt.tight_layout()
    # plt.savefig('Graphs2009/2009_5-2_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()

# 6 most effective period against highest and lowest averaged teams
print('most effective period against highest and lowest averaged teams')
print('for 2 teams kobe averaged highest')
for j in HA2teams:
    actions2002=list(data2009.loc[data2009['opponent']==j,'period'].unique())
    total_size=data2009.loc[data2009['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2009_yes.loc[(data2009_yes['opponent']==j) & (data2009_yes['period']==i),'shot_made_flag'].size
        total_size1=data2009.loc[(data2009['opponent']==j) & (data2009['period']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=10, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1

    l1=np.array(list(actions2002))-1
    ax2.plot(l1, plot_df1['eff'], 'b-')
    ax2.scatter(l1, plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    # ax2.set_ylim([20,50])
    plt.title('shots scored vs period for '+j+'- 2009')
    ax1.set_xticklabels(labels=plot_df1['type'])
    plt.tight_layout()
    # plt.savefig('Graphs2009/2009_6-1_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


print('for 2 teams kobe averaged lowest')
for j in LA2teams:
    actions2002=list(data2009.loc[data2009['opponent']==j,'period'].unique())
    total_size=data2009.loc[data2009['opponent']==j,'shot_made_flag'].size
    shots_percent2=[]

    for i in actions2002:
        score_no=data2009_yes.loc[(data2009_yes['opponent']==j) & (data2009_yes['period']==i),'shot_made_flag'].size
        total_size1=data2009.loc[(data2009['opponent']==j) & (data2009['period']==i),'shot_made_flag'].size
        shots_percent2.append([total_size1,(score_no*100)/total_size1,i,(total_size1*100)/total_size])
    print('for team : ',j,'for scored shots')
    print(shots_percent2)

    plot_df1=pd.DataFrame(shots_percent2,columns=['count','eff','type','avg'], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax2 = ax1.twinx()
    plots=sns.barplot(x='type',y='count',data=plot_df1,ax=ax1)
    z=0
    for bar in plots.patches:
        plots.annotate('%0.2f%%'%(shots_percent2[z][3]),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=10, xytext=(0, 8),
                       textcoords='offset points')
        z=z+1

    l1=np.array(list(actions2002))-1
    ax2.plot(l1, plot_df1['eff'], 'b-')
    ax2.scatter(l1, plot_df1['eff'],s=40)
    ax2.set_ylabel('FG%')
    # ax2.set_ylim([20,50])
    plt.title('shots scored vs period for '+j+'- 2009')
    ax1.set_xticklabels(labels=plot_df1['type'])
    plt.tight_layout()
    # plt.savefig('Graphs2009/2009_6-2_'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()

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

for j in HA2teams:
    cmade={1:'green',0:'red'}
    plt.figure(figsize=(12,11))
    plt.scatter(data2009.loc[data2009['opponent']==j,'loc_x'], data2009.loc[data2009['opponent']==j,'loc_y'], c=data2009.loc[data2009['opponent']==j,'shot_made_flag'].map(cmade))
    draw_court(outer_lines=True)
    # Descending values along the axis from left to right
    plt.xlim(300,-300)
    plt.ylim(-100,500)
    plt.title('location vs shot_made_flag -'+j)
    # plt.savefig('Graphs2009/ShotChart_v1 for'+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


for j in LA2teams:
    cmade={1:'green',0:'red'}
    plt.figure(figsize=(12,11))
    plt.scatter(data2009.loc[data2009['opponent']==j,'loc_x'], data2009.loc[data2009['opponent']==j,'loc_y'], c=data2009.loc[data2009['opponent']==j,'shot_made_flag'].map(cmade))
    draw_court(outer_lines=True)
    # Descending values along the axis from left to right
    plt.xlim(300,-300)
    plt.ylim(-100,500)
    plt.title('location vs shot_made_flag -'+j)
    # plt.savefig('Graphs2009/ShotChart_v1 '+j+'.jpeg', dpi=300, bbox_inches='tight')
    plt.show()


