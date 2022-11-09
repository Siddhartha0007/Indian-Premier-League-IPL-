# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 08:20:15 2022

@author: Siddhartha-PC
"""

###  Import Libreries  
import streamlit as st
from streamlit_option_menu import option_menu
st.set_option('deprecation.showPyplotGlobalUse', False)
import contractions 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image 
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from collections import  Counter
import inflect
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
import os
#for model-building
from sklearn.model_selection import train_test_split
import string
from tqdm import tqdm
#for model accuracy
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
#for visualization
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go
import pickle
from joblib import dump, load
import joblib
# Utils
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import sys
from sklearn.metrics import r2_score,accuracy_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error 
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from plotly import tools
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
import plotly.figure_factory as ff
import cufflinks as cf
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import scipy
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import winsorize
from scipy.stats import boxcox, probplot, norm
from scipy.special import inv_boxcox
import random
import datetime
import math
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import silhouette_score
## Hyperopt modules
#from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
#from functools import partial
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
from yellowbrick.classifier import PrecisionRecallCurve
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import plot_importance, to_graphviz
from xgboost import plot_tree
#For Model Acuracy
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import plot_importance, to_graphviz #
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
import plotly.graph_objs as go
import plotly.tools as tls
import cufflinks as cf
import plotly.figure_factory as ff
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import plot_importance, to_graphviz
from xgboost import plot_tree
#For Model Acuracy
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import plot_importance, to_graphviz #
from sklearn.metrics import classification_report
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import log_loss
# EDA Pkgs
import pandas as pd 
import codecs
from pandas_profiling import ProfileReport 
# Components Pkgs
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

# Custome Component Fxn
import sweetviz as sv 
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#lottie animations
import time
import requests
import json
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
#nltk libreries
import io
import requests
import json
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
#from surprise import SVD,Reader,Dataset
import matplotlib.patches as mpatches
import plotly.tools as tls
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins
from matplotlib import animation,rc
from IPython.display import HTML, display

from streamlit_folium import folium_static
import folium

###############################################Data Processing###########################
data=pd.read_csv('./matches.csv')
deliveries=pd.read_csv('./deliveries.csv')
df1 = pd.read_csv('./matches.csv')
df2=pd.read_csv('./deliveries.csv')
matches=data.copy()




def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_ev1cfn9h.json"
lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"
lottie_hello = load_lottieurl(lottie_url_hello)
project_url_1="https://assets9.lottiefiles.com/packages/lf20_bzgbs6lx.json"
project_url_2="https://assets6.lottiefiles.com/packages/lf20_eeuhulsy.json"
report_url="https://assets9.lottiefiles.com/packages/lf20_zrqthn6o.json"
about_url="https://assets2.lottiefiles.com/packages/lf20_k86wxpgr.json"

about_1=load_lottieurl(about_url)
report_1=load_lottieurl(report_url)
project_1=load_lottieurl(project_url_1)
project_2=load_lottieurl(project_url_2)

lottie_download = load_lottieurl(lottie_url_download)


def st_display_sweetviz(report_html,width=1000,height=500):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)



###############################################Streamlit Main###############################################

def main():
    data=pd.read_csv('./matches.csv')
    deliveries=pd.read_csv('./deliveries.csv')
    df1 = pd.read_csv('./matches.csv')
    df2=pd.read_csv('./deliveries.csv')
    matches=data.copy()
    # set page title           
    # 2. horizontal menu with custom style
    selected = option_menu(menu_title= None,options=["Home", "Project","Report" ,"About"], icons=["house-door", "cast","clipboard","file-person"],  menu_icon="cast", default_index=0, orientation="horizontal", styles={"container": {"padding": "0!important", "background-color": "cyan"},"icon": {"color": "#6c3483", "font-size": "25px"}, "nav-link": {"font-size": "25px","text-align": "left","margin": "0px","--hover-color": "#ff5733", },"nav-link-selected":{"background-color":"#2874a6"},},)
    
    #horizontal Home selected
    if selected == "Home":
        col1,col2=st.columns(2)
        with col1:
            lottie_url_hello2 = "https://assets5.lottiefiles.com/packages/lf20_qdmxvg00.json"
            lottie_hello2 = load_lottieurl(lottie_url_hello2)
            st_lottie(lottie_hello2, key="hello226",)
        with col2:
            image= Image.open("home1.png")
            st.image(image,use_column_width=True)
        
            
        st.sidebar.title("Home")        
        with st.sidebar:
            lottie_url_hello2 = "https://assets10.lottiefiles.com/packages/lf20_zdm1abxk.json"
            lottie_hello2 = load_lottieurl(lottie_url_hello2)
            st_lottie(lottie_hello2, key="hello2",)
            image= Image.open("home_side.png")
            st.image(image,use_column_width=True)                       
                   
        def header(url):
            st.sidebar.markdown(f'<p style="background-color:#a569bd ;color:white;font-size:15px;border-radius:1%;">{url}', unsafe_allow_html=True)    
        html_45=""" A Quick Youtube Video for understanding the MOVIE RECOMMENDATION SYSTEMS for Educational Purpose ."""
        
        #st.sidebar.video("https://www.youtube.com/watch?v=n3RKsY2H-NE")
        with st.sidebar:
            #image= Image.open("Home1.png")
            st.write('Author@ Siddhartha Sarkar')
            st.write('Data Scientist ')
            st.write('References:')
            st.markdown("[ Indian Premier League:](https://www.iplt20.com/)")
            st.markdown("[  Wikipedia:](https://en.wikipedia.org/wiki/Indian_Premier_League)")
            
        st.balloons()
        #header(html_45)
        #st.sidebar.video("https://www.youtube.com/watch?v=n3RKsY2H-NE")
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;"> 🎥 Exploratory Data Analysis(EDA) on Indian Premier League(IPL) Data</h1>
		</div>  """
        
		
        components.html(html_temp)
        st.subheader('Table of Contents:')
        def header(url):
            st.markdown(f'<p style="background-color:#d7bde2 ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        
        html_temp11 = """
		           Table of  CONTENTS:<br>
                  + Description of Terrorism <br>
                  + Brief Description of the dataset <br>
                  + Importing of libraries/modules <br>
                  + Information of Dataset <br>
                  + Data Cleaning and Prepartion of Dataset <br>
                  + Analysis of the Dataset <br>
                  
                  


        
		  """
        
		
        #header(html_temp11)
        
        
        
        def plot11():
            
            values = [['1', '2', '3', '4','5','->' ,'->' ,'->' ,'->' ,'->' ,'->' ,'->' ,'->' ,'->','6'], #1st col
            ['  Brief Description about the Dataset', 'Importing of Libraries', 'Loading Dataset and displaying Information',
       'Data Understanding & Cleaning', 'Exploratory Data Analysis and Data Visualization',\
   'Number of matches played in different cities','Number of matches held in different venues',\
   'Number of matches won by each team','Information of match won by most runs out of all seasons',\
   'Pie chart of Win Type(Bat First/ Bowl First)','Bar graph of number of matches won by batting or bowling first',\
   'Pie chart of decisions taken after winning tos','Number of matches in which the winner of the toss has also won the match',\
   'Number of matches in which the winner of the toss has also won the match.','Conclusion']]


            fig = go.Figure(data=[go.Table(
  columnorder = [1,2],
  columnwidth = [40,400],
  header = dict(
    values = [['<b>Serial No</b>'],
                  ['<b>Table of Contents</b>']],
    line_color='red',
    fill_color='forestgreen',
    align=['left','center'],
    font=dict(color='white', size=12),
    height=40
  ),
  cells=dict(
    values=values,
    line_color='red',
    fill=dict(color=['rebeccapurple', '#4d004c']),
    align=['left', 'center'],
    font_size=12,
    font=dict(color='white', size=12),
    height=20)
    )
])

            return fig

        p11=plot11()
        st.plotly_chart(p11)
        
        
        def header(url):
            st.markdown(f'<p style="background-color: #d7bde2 ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        html_temp12 = """
		 The Indian Premier League (IPL) is a professional men's Twenty20 cricket league, contested by ten teams 
         based out of ten Indian cities. The league was founded by the Board of Control for Cricket in India (BCCI) 
         in 2007. It is usually held between March and May of every year and has an exclusive window in the ICC 
         Future Tours Programme.¶<br>
         The IPL is the most-attended cricket league in the world and in 2014 was ranked sixth by average 
         attendance among all sports leagues. In 2010, the IPL became the first sporting event in the world to be 
         broadcast live on YouTube. The brand value of the IPL in 2019 was ₹47,500 crore (6.3 billion US dollar), 
         according to Duff & Phelps. According to BCCI, the 2015 IPL season contributed ₹1,150 crore (150 million US dollar)
         to the GDP of the Indian economy. The 2020 IPL season set a massive viewership record with 31.57 million average 
         impressions and with an overall consumption increase of 23 per cent from the 2019 season.<br>
         There have been fourteen seasons of the IPL tournament. The current IPL title holders are the Chennai Super Kings, 
         winning the 2021 season. The venue for the 2020 season was moved due to the COVID-19 pandemic and games were 
         played in the United Arab Emirates.<br>
		  """
        		                      
        header(html_temp12)
        
        st.subheader(' Descriptions of DataSet-1')
        def plot11():
            values = [['id', 'season', 'city', 'date', 'team1', 'team2', 'toss_winner',
            'toss_decision', 'result', 'dl_applied', 'winner', 'win_by_runs',
            'win_by_wickets', 'player_of_match', 'venue', 'umpire1', 'umpire2'], #1st col
            [' Unique id Number', 'Season', 'City', 'Date', 'Team1', 'Team2', 'Toss Winner',
            'Toss_Decision', 'Result', 'Dl_Applied', 'Winner', 'Win_by_Runs',
            'Win_by_Wickets', 'Player of the Match', 'Venue', 'Main Umpire', 'Leg Umpire']]

            fig = go.Figure(data=[go.Table(
            columnorder = [1,2],
            columnwidth = [80,400],
            header = dict(
            values = [['<b>Columns<br>of the Dataset</b>'],
                  ['<b>DESCRIPTION</b>']],
            line_color='darkslategray',
            fill_color='royalblue',
            align=['left','center'],
            font=dict(color='white', size=12),
            height=40
                       ),
            cells=dict(
            values=values,
            line_color='red',
            fill=dict(color=['#6495ED', '#4d004c']),
            align=['left', 'center'],
            font_size=12,
            font=dict(color='white', size=12),
            height=20)
                     )
                  ])

            return fig

        p11=plot11()
        st.plotly_chart(p11)
        
        st.subheader(' Descriptions of DataSet-2')
        def plot11():
            
            values = [['id', 'inning', 'batting_team', 'bowling_team', 'over', 'ball',
            'batsman', 'non_striker', 'bowler', 'is_super_over', 'wide_runs',
            'bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs',
       'batsman_runs', 'extra_runs', 'total_runs', 'player_dismissed',
       'dismissal_kind', 'fielder'], #1st col
  ['Unique id Number', 'Innings', 'Batting Team', 'Bowling Team', 'Over', 'Ball',
       'Batsman', 'Non Striker', 'Bowler', 'Super over Occured/NOt', 'Wide Runs',
       'Bye runs', 'legbye runs', 'noball runs', 'penalty runs',
       'batsman runs', 'extra runs', 'total runs', 'Player Dismissed',
       'Dismissal Kind', 'Fielder']]


            fig = go.Figure(data=[go.Table(
  columnorder = [1,2],
  columnwidth = [80,400],
  header = dict(
    values = [['<b>Columns<br>of the Dataset</b>'],
                  ['<b>DESCRIPTION</b>']],
    line_color='darkslategray',
    fill_color='darkturquoise',
    align=['left','center'],
    font=dict(color='white', size=12),
    height=40
  ),
  cells=dict(
    values=values,
    line_color='red',
    fill=dict(color=['forestgreen', '#4d004c']),
    align=['left', 'center'],
    font_size=12,
    font=dict(color='white', size=12),
    height=20)
    )
])

            return fig

        p11=plot11()
        st.plotly_chart(p11)
        
        
        st.subheader('  DataSet-1')
        def plot12():
            import plotly.figure_factory as ff
            df_sample = data.iloc[0:10,1:10]
            colorscale = [[0, '#4d004c'],[.5, '#6495ED'],[1, '#9FE2BF']]
            font_colors=['#ffffff', '#000000','#000000']
            fig =  ff.create_table(df_sample, colorscale=colorscale,font_colors=font_colors,index=True)

            return fig
        p12=plot12()
        st.write("Data Table")
        st.plotly_chart(p12)
        
        st.subheader('DataSet-2')
        def plot12():
            import plotly.figure_factory as ff
            df_sample = df2.iloc[0:10,1:10]
            colorscale = [[0, '#4d004c'],[.5, '#6495ED'],[1, '#9FE2BF']]
            font_colors=['#ffffff', '#000000','#000000']
            fig =  ff.create_table(df_sample, colorscale=colorscale,font_colors=font_colors,index=True)

            return fig
        p12=plot12()
        st.write("Data Table")
        st.plotly_chart(p12)


        def header(url):
            
            st.markdown(f'<p style="background-color: #d7bde2 ;color:black;font-size:20px;border-radius:0%;">{url}', unsafe_allow_html=True)
        html_temp13 = """
		 Objective:<br>
         =================================================================<br>
         1.Perform 'Exploratory Data Analysis' on ‘Indian Premier League’ dataset.<br>
         2.Analyse, Various factors contributing for win or loss of a team .<br>
         3.Suggest best teams or players , a company should endorse for its products.<br>
         =================================================================
      
		  """      		
        header(html_temp13)
        
        st.markdown("""
                #### Tasks Perform by the app:
                + App covers the most basic Machine Learning task of  Analysis, Correlation between variables, Basic Statistics.
                + Exploratory Data Analysis .
                
                """)
                
    #Horizontal About selected
    if selected == "About":
        #st.title(f"You have selected {selected}")
        
        st.sidebar.title("About")
        with st.sidebar:
            image= Image.open("About-Us-PNG-Isolated-Photo.png")
            add_image=st.image(image,use_column_width=True)
        
        st_lottie(about_1,key='ab1')
        #image2= Image.open("about.jpg")
        #st.image(image2,use_column_width=True)
        #st.sidebar.write("This Youtube Video Shows and Describes Different Kind Of Mushrooms for Learning Purpose ")
        #st.sidebar.video('https://www.youtube.com/watch?v=6PNq6paMBXU')
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Exploratory Data Analysis on Indian Premier League(IPL) Data </h1>
		</div>  """
        
		
        components.html(html_temp)
        def header(url):
            st.markdown(f'<p style="background-color: forestgreen ;color:black;font-size:30px;border-radius:2%;">{url}', unsafe_allow_html=True)
        html_99   =  """
        In this project,I tried to analyse the Indian Premier League(IPL) data to gain insight about the several factors contributing to 
        win or loss .Also this tried to find out best performing teams and players, performance of the man of the match ,
        batsman with best strike rates ,top scorers , top wicket takers etc ."""
        header(html_99)
        
        st.sidebar.markdown("""
                    #### + Project Done By :        
                    #### @Author Mr. Siddhartha Sarkar
                    
        
                    """)
        st.snow()
        
        #st.sidebar.markdown("[ Visit To Github Repositories](.git)")
    #Horizontal Project_Report selected
    if selected == "Report":
        #report_1
        #st.title("Profile Report")
        st.sidebar.title("Project Profile Report")
        
        with st.sidebar:
            lottie_url_hello2 = "https://assets5.lottiefiles.com/packages/lf20_5mccxzyl.json"
            lottie_hello2 = load_lottieurl(lottie_url_hello2)
            st_lottie(lottie_hello2, key="hello2",)
            
        
        st.balloons()    
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Profile Report Generation </h1>
		</div>  """
        
		
        components.html(html_temp)
        html_temp1 = """
			<style>
			* {box-sizing: border-box}
			body {font-family: Verdana, sans-serif; margin:0}
			.mySlides {display: none}
			img {vertical-align: middle;}
			/* Slideshow container */
			.slideshow-container {
			  max-width: 1500px;
			  position: relative;
			  margin: auto;
			}
			/* Next & previous buttons */
			.prev, .next {
			  cursor: pointer;
			  position: absolute;
			  top: 50%;
			  width: auto;
			  padding: 16px;
			  margin-top: -22px;
			  color: white;
			  font-weight: bold;
			  font-size: 18px;
			  transition: 0.6s ease;
			  border-radius: 0 3px 3px 0;
			  user-select: none;
			}
			/* Position the "next button" to the right */
			.next {
			  right: 0;
			  border-radius: 3px 0 0 3px;
			}
			/* On hover, add a black background color with a little bit see-through */
			.prev:hover, .next:hover {
			  background-color: rgba(0,0,0,0.8);
			}
			/* Caption text */
			.text {
			  color: #f2f2f2;
			  font-size: 15px;
			  padding: 8px 12px;
			  position: absolute;
			  bottom: 8px;
			  width: 100%;
			  text-align: center;
			}
			/* Number text (1/3 etc) */
			.numbertext {
			  color: #f2f2f2;
			  font-size: 12px;
			  padding: 8px 12px;
			  position: absolute;
			  top: 0;
			}
			/* The dots/bullets/indicators */
			.dot {
			  cursor: pointer;
			  height: 15px;
			  width: 15px;
			  margin: 0 2px;
			  background-color: #bbb;
			  border-radius: 50%;
			  display: inline-block;
			  transition: background-color 0.6s ease;
			}
			.active, .dot:hover {
			  background-color: #717171;
			}
			/* Fading animation */
			.fade {
			  -webkit-animation-name: fade;
			  -webkit-animation-duration: 1.5s;
			  animation-name: fade;
			  animation-duration: 1.5s;
			}
			@-webkit-keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}
			@keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}
			/* On smaller screens, decrease text size */
			@media only screen and (max-width: 300px) {
			  .prev, .next,.text {font-size: 11px}
			}
			</style>
			</head>
			<body>
			<div class="slideshow-container">
			<div class="mySlides fade">
			  <div class="numbertext">1 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_nature_wide.jpg" style="width:100%"> 
			  <div class="text">Caption Text</div>
			</div>
			<div class="mySlides fade">
			  <div class="numbertext">2 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_snow_wide.jpg" style="width:100%">
			  <div class="text">Caption Two</div>
			</div>
			<div class="mySlides fade">
			  <div class="numbertext">3 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_mountains_wide.jpg" style="width:100%">
			  <div class="text">Caption Three</div>
			</div>
			<a class="prev" onclick="plusSlides(-1)">&#10094;</a>
			<a class="next" onclick="plusSlides(1)">&#10095;</a>
			</div>
			<br>
			<div style="text-align:center">
			  <span class="dot" onclick="currentSlide(1)"></span> 
			  <span class="dot" onclick="currentSlide(2)"></span> 
			  <span class="dot" onclick="currentSlide(3)"></span> 
			</div>
			<script>
			var slideIndex = 1;
			showSlides(slideIndex);
			function plusSlides(n) {
			  showSlides(slideIndex += n);
			}
			function currentSlide(n) {
			  showSlides(slideIndex = n);
			}
			function showSlides(n) {
			  var i;
			  var slides = document.getElementsByClassName("mySlides");
			  var dots = document.getElementsByClassName("dot");
			  if (n > slides.length) {slideIndex = 1}    
			  if (n < 1) {slideIndex = slides.length}
			  for (i = 0; i < slides.length; i++) {
			      slides[i].style.display = "none";  
			  }
			  for (i = 0; i < dots.length; i++) {
			      dots[i].className = dots[i].className.replace(" active", "");
			  }
			  slides[slideIndex-1].style.display = "block";  
			  dots[slideIndex-1].className += " active";
			}
			</script>
			"""
        components.html(html_temp1)
        st.sidebar.title("Navigation")
        menu = ['None',"Sweetviz","Pandas Profile"]
        choice = st.sidebar.radio("Menu",menu)
        if choice == 'None':
            st.markdown("""
                        #### Kindly select from left Menu.
                       # """)
        elif choice == "Pandas Profile":
            st.subheader("Automated EDA with Pandas Profile")
            st.subheader("Upload Your File to Perform Report analysis")
            data_file= st.file_uploader("Upload CSV",type=['csv'])
            if data_file is not None:
                df = pd.read_csv(data_file)
                st.table(df.head(10))
                m = st.markdown("""
                       <style>
                    div.stButton > button:first-child {
                  background-color: #0099ff;
                      color:#ffffff;
                                   }
                    div.stButton > button:hover {
                     background-color: #00ff00;
                         color:#ff0000;
                          }
                    </style>""", unsafe_allow_html=True)

                if st.button("Generate Profile Report"):
                    profile= ProfileReport(df)
                    st_profile_report(profile)
            
            
        elif choice == "Sweetviz":
            st.subheader("Automated EDA with Sweetviz")
            st.subheader("Upload Your File to Perform Report analysis")
            data_file = st.file_uploader("Upload CSV",type=['csv'])
            if data_file is not None:
                df =  pd.read_csv(data_file)
                st.dataframe(df.head(10))
                m = st.markdown("""
                            <style>
                         div.stButton > button:first-child {
                                 background-color: #0099ff;
                                      color:#ffffff;
                                        }
                            div.stButton > button:hover {
                               background-color: #00ff00;
                                       color:#ff0000;
                                             }
                        </style>""", unsafe_allow_html=True)

                if st.button("Generate Sweetviz Report"):
                    # Normal Workflow
                    report = sv.analyze(df)
                    report.show_html()
                    st_display_sweetviz("SWEETVIZ_REPORT.html") 
               			
		      
    #Horizontal Project selected
    if selected == "Project":
            data=pd.read_csv('./matches.csv')
            deliveries=pd.read_csv('./deliveries.csv')
            df1 = pd.read_csv('./matches.csv')
            df2=pd.read_csv('./deliveries.csv')
            matches=data.copy()
            
            with st.sidebar:
                st_lottie(project_1, key="pro1125")                
            import time                
            st_lottie(project_2, key="pro2225")
            st.title("Projects")              
            
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            st.sidebar.title("Navigation")
            with st.sidebar:
                
                menu_Pre_Exp = option_menu("Exploratory Data Analysis", ['Basic Statistics',"Visualisations","Conclusions"],
                         icons=['back','bar-chart'],
                         menu_icon="app-indicator", default_index=0,orientation="vertical",
                         styles={
        "container": {"padding": "5!important", "background-color": "#c39bd3"},
        "icon": {"color": "#2980b9", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#ec7063"},
        "nav-link-selected": {"background-color": "green"}, } )

  
            if  menu_Pre_Exp == 'Basic Statistics' : # and selected == "Projects"
                    st.title('Basic Statistics')
                    
                    st.subheader('Dataset-1')
                    st.write(data.head(10))
                    st.subheader('Dataset-2')
                    st.write(df2.head(10))
                    st.subheader("Some Basic Insights Of the Data")
                    numeric_data=data.select_dtypes(exclude=["object"]).columns.to_list()
                    categorical_data=data.select_dtypes(include=["object"]).columns.to_list()
                    st.write("Numeric Columns :\n",numeric_data)
                    st.write("--"*40)
                    st.write("Categorical Data :\n",categorical_data)
                    # printing unique values of each column
                    for col in data[categorical_data].columns:
                        st.write(f"{col}: \n{data[col].unique()}\n")                           
                    st.write('Columns:',deliveries.columns)
                    st.write('batting_team\n',df2.batting_team.unique())
                    st.write('bowling_team\n',df2.bowling_team.unique())
                    
                    df2["total_extra_runs"]=df2["wide_runs"]+df2["bye_runs"]+df2['legbye_runs']+df2['noball_runs']+df2['penalty_runs']+df2['extra_runs']
                    df2.drop(columns=['wide_runs','bye_runs','legbye_runs','noball_runs','penalty_runs','extra_runs'], inplace = True)
                    st.write(df1.groupby(['city','venue']).count()['id'].reset_index().style.background_gradient())
                    st.subheader("Basic Statistics Of the Data:")
                    def plot12():
                        import plotly.figure_factory as ff
                        df_sample = data.describe(include='all').round(2).T
                        colorscale = [[0, '#4d004c'],[.5, '#6495ED'],[1, '#9FE2BF']]
                        font_colors=['#ffffff', '#000000','#000000']
                        fig =  ff.create_table(df_sample, colorscale=colorscale,font_colors=font_colors,index=True)
                        return fig
                    p12=plot12()
                    st.subheader('Dataset-1')
                    st.plotly_chart(p12)
                    
                    def plot12():
                        import plotly.figure_factory as ff
                        df_sample = df2.describe(include='all').round(2).T
                        colorscale = [[0, '#4d004c'],[.5, '#6495ED'],[1, '#9FE2BF']]
                        font_colors=['#ffffff', '#000000','#000000']
                        fig =  ff.create_table(df_sample, colorscale=colorscale,font_colors=font_colors,index=True)
                        return fig
                    p12=plot12()
                    st.subheader('Dataset-2')
                    st.plotly_chart(p12)
                    
            if  menu_Pre_Exp == 'Visualisations' : # and selected == "Projects"
                    st.title('Visualisations')
                    
                    m = st.markdown("""
                                  <style>
                                  div.stButton > button:first-child {
                                   background-color: #0099ff;
                                       color:#ffffff;
                                                         }
                                   div.stButton > button:hover {
                                   background-color: #00ff00;
                                   color:#ff0000;
                                                   }
                                  </style>""", unsafe_allow_html=True)
                    submit = st.button(label='Generate Visualizations')
                    if submit:
                        
                        image2= Image.open("img1.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img2.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img3.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img4.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img5.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img6.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img7.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img8.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img9.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img10.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img11.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img12.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img13.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img14.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img15.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img16.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img17.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img18.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img19.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img20.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img21.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img22.png")
                        st.image(image2,use_column_width=True)
                        image2= Image.open("img23.png")
                        st.image(image2,use_column_width=True)
                        
                        
                        matches_played=pd.concat([data['team1'],data['team2']])
                        matches_played=matches_played.value_counts().reset_index()
                        matches_played.columns=['Team','Total Matches']
                        matches_played['wins']=data['winner'].value_counts().reset_index()['winner']

                        matches_played.set_index('Team',inplace=True)
                        totm = matches_played.reset_index().head(10)
                        trace1 = go.Bar(x=matches_played.index,y=matches_played['Total Matches'],
                        name='Total Matches',marker=dict(color='blue'),opacity=0.4)

                        trace2 = go.Bar(x=matches_played.index,y=matches_played['wins'],
                        name='Matches Won',marker=dict(color='green'),opacity=0.4)

                        trace3 = go.Bar(x=matches_played.index,
               y=(round(matches_played['wins']/matches_played['Total Matches'],3)*100),
               name='Win Percentage',opacity=0.6,marker=dict(color='gold'))

                        data2 = [trace1, trace2, trace3]

                        layout = go.Layout(title='Match Played, Wins And Win Percentage',xaxis=dict(title='Teams'),
                         yaxis=dict(title='Count'),bargap=0.2,bargroupgap=0.1, plot_bgcolor='rgb(245,245,245)')

                        fig = go.Figure(data=data2, layout=layout)
                        st.plotly_chart(fig)
                        ump=pd.concat([data['umpire1'],data['umpire2']])
                        ump=ump.value_counts()
                        umps=ump.to_frame().reset_index()
                        data3 = [go.Bar(x=umps['index'],y=umps[0],marker=dict(color='#4d004c'),opacity=0.4)]

                        layout = go.Layout(title='Umpires in Matches',
                              yaxis=dict(title='Matches'),bargap=0.2, plot_bgcolor='rgb(245,245,245)')

                        fig = go.Figure(data=data3, layout=layout)                                                                                               
                        st.plotly_chart(fig)
                        
                        deliveries.rename(columns={'match_id': 'id'},inplace=True)
                        batsmen = matches[['id','season']].merge(deliveries, left_on = 'id', right_on = 'id', how = 'left').drop('id', axis = 1)
                        season=batsmen.groupby(['season'])['total_runs'].sum().reset_index()
                        avgruns_each_season=matches.groupby(['season']).count().id.reset_index()
                        avgruns_each_season.rename(columns={'id':'matches'},inplace=1)
                        avgruns_each_season['total_runs']=season['total_runs']
                        avgruns_each_season['average_runs_per_match']=avgruns_each_season['total_runs']/avgruns_each_season['matches']
                                   
                        fig = {"data" : [{"x" : season["season"],"y" : season["total_runs"],
                        "name" : "Total Run","marker" : {"color" : "lightblue","size": 12},
                        "line": {"width" : 3},"type" : "scatter","mode" : "lines+markers" },
        
                        {"x" : season["season"],"y" : avgruns_each_season["average_runs_per_match"],
                        "name" : "Average Run","marker" : {"color" : "aqua","size": 12},
                        "type" : "scatter","line": {"width" : 3},"mode" : "lines+markers",
                        "xaxis" : "x2","yaxis" : "y2",}],
       
                        "layout" : {"title": "Total and Average run per Season",
                        "xaxis2" : {"domain" : [0, 1],"anchor" : "y2",
                        "showticklabels" : False},"margin" : {"b" : 111},
                        "yaxis2" : {"domain" : [.55, 1],"anchor" : "x2","title": "Average Run"},                    
                        "xaxis" : {"domain" : [0, 1],"tickmode":'linear',"title": "Year"},
                        "yaxis" : {"domain" :[0, .45], "title": "Total Run"}}}

                        st.plotly_chart(fig)
                        
                        Season_boundaries=batsmen.groupby("season")["batsman_runs"].agg(lambda x: (x==6).sum()).reset_index()
                        fours=batsmen.groupby("season")["batsman_runs"].agg(lambda x: (x==4).sum()).reset_index()
                        Season_boundaries=Season_boundaries.merge(fours,left_on='season',right_on='season',how='left')
                        Season_boundaries=Season_boundaries.rename(columns={'batsman_runs_x':'6"s','batsman_runs_y':'4"s'})
                        Season_boundaries['6"s'] = Season_boundaries['6"s']*6
                        Season_boundaries['4"s'] = Season_boundaries['4"s']*4
                        Season_boundaries['total_runs'] = season['total_runs']
                        trace1 = go.Bar(
                                x=Season_boundaries['season'],
                                    y=Season_boundaries['total_runs']-(Season_boundaries['6"s']+Season_boundaries['4"s']),
                                 marker = dict(line=dict(color='#000000', width=1)),
                        name='Remaining runs',opacity=0.6)

                        trace2 = go.Bar(
                          x=Season_boundaries['season'],
                          y=Season_boundaries['4"s'],
                                   marker = dict(line=dict(color='#000000', width=1)),
                            name='Run by 4"s',opacity=0.7)

                        trace3 = go.Bar(
                         x=Season_boundaries['season'],
                                           y=Season_boundaries['6"s'],
                                marker = dict(line=dict(color='#000000', width=1)),
                                          name='Run by 6"s',opacity=0.7)


                        data12 = [trace1, trace2, trace3]
                        layout = go.Layout(title="Run Distribution per year",barmode='stack',xaxis = dict(tickmode='linear',title="Year"),
                                    yaxis = dict(title= "Run Distribution"), plot_bgcolor='rgb(245,245,245)')

                        fig = go.Figure(data=data12, layout=layout)
                        st.plotly_chart(fig)

                        
                        bowlers=deliveries.groupby('bowler').sum().reset_index()
                        bowl=deliveries['bowler'].value_counts().reset_index()
                        bowlers=bowlers.merge(bowl,left_on='bowler',right_on='index',how='left')
                        bowlers=bowlers[['bowler_x','total_runs','bowler_y']]
                        bowlers.rename({'bowler_x':'bowler','total_runs':'runs_given','bowler_y':'balls'},axis=1,inplace=True)
                        bowlers['overs']=(bowlers['balls']//6)
                        dismissal_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]  
                        ct=deliveries[deliveries["dismissal_kind"].isin(dismissal_kinds)]
                        ct=ct['bowler'].value_counts().reset_index()
                        bowlers=bowlers.merge(ct,left_on='bowler',right_on='index',how='left').dropna()
                        bowlers=bowlers[['bowler_x','runs_given','overs','bowler_y']]
                        bowlers.rename({'bowler_x':'bowler','bowler_y':'wickets'},axis=1,inplace=True)
                        bowlers['economy']=(bowlers['runs_given']/bowlers['overs'])
                        bowlers_top=bowlers.sort_values(by='runs_given',ascending=False)
                        bowlers_top=bowlers_top.head(20)
                        trace = go.Scatter(y = bowlers_top['wickets'],x = bowlers_top['bowler'],mode='markers',
                        marker=dict(size= bowlers_top['wickets'].values,
                               color = bowlers_top['economy'].values,
                               colorscale='rainbow',
                               showscale=True,
                               colorbar = dict(title = 'Economy')),
                        text = bowlers['overs'].values)

                        data15 = [(trace)]

                        layout= go.Layout(autosize= True,
                        title= 'Top 20 Wicket Taking Bowlers',
                        hovermode= 'closest',
                        xaxis=dict(showgrid=False,zeroline=False,
                             showline=False),
                        yaxis=dict(title= 'Wickets Taken',ticklen= 5,
                             gridwidth= 2,showgrid=False,
                             zeroline=False,showline=False),
                        showlegend= False,plot_bgcolor='rgb(245,245,245)')

                        fig = go.Figure(data=data15, layout=layout)
                        st.plotly_chart(fig)

            if  menu_Pre_Exp == 'Conclusions' :
                st.subheader('Conclusions')
                st.markdown("""
                            
                            + IPL Season 2011,2012,2013 had more number of matches played.
                            + Mumbai Indians is the most successful team in IPL as they have won more number of matches as well as toss.
                            + When defending a total, the biggest victory was by 146 runs(Mumbai Indians defeated Delhi Daredevils by 146 runs on 06 May 2017 at Feroz Shah Kotla stadium, Delhi)
                            + Mumbai city has hosted the most number of IPL matches.
                            + The team winning the toss has 51% chance of winning the match.
                            + The team winning toss choose to field first as it has higher chances of winning.
                            + Bowling First has proven more advantageous in most of the venues.
                            + Chinnaswamy stadium has hosted most number of IPL matches in thr history of IPL.
                            + Only Mumbai and Pune has more than one venues, others have only one.
                            + IPL season 2013 witnessed most runs made.
                            + Virat Kohli and SK Raina are the top run scoring batsmen in IPL.
                            + SL Malinga is the top wicket taking bowler.
                            + CH Gayle has most number of sixes.
                            + CH Gayle,KA Pollard, DA Warner,SR Watson and BB McCullum have good strike rates compared to other batsmen.
                            
                            
                    
                    
                    
                    
                    
                    
                    """)


            
              
if __name__=='__main__':
    
    main()            









