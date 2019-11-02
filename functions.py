import pandas as pd
import seaborn as sns
import datetime
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.tsaplots import plot_pacf
from fbprophet import Prophet
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import plotly.graph_objects as go
plt.style.use('fivethirtyeight')

## Functions for Time Series

def convert_to_dt(row):
    """ it takes a row and conveted to a date time ojbect"""
    dt = datetime.datetime(row.Year, 1, 1)
    return dt

def convert_to_dt1(df):
    """ applys convert to date time to the feature Year"""
    df_copy = df.copy()
    df_copy.Year = df_copy.apply(convert_to_dt , axis =1)
    return df_copy

def subseting (df ,state):
    """ It takes a df and return only Year 
    and Deaths of that State"""
    df_copy = df.copy()
    df = df[df.State == state]
    df_copy = df[[ 'Year' , 'Deaths']]
    df = pd.to_numeric(df['Deaths'])
    return df_copy

def indexing_column_year(list_):
    """ it takes a list of dfs and set the Year as index"""
    for i in list_:
        i.index = i['Year']
#         i = i.drop('Year', axis = 1)
        
def dickey_fuller(list_of_dfs):
    """ it takes a list of dfs and return the dicey fuller test"""
    for i in list_of_dfs:
        X = i["Deaths"].values
        result = adfuller(X)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
            
def log_trans(list_dfs):
    """ it takes a list of dfs and compute the log and the differentiation
    of the Deaths feature"""
    for i in list_dfs:
        i["Deaths"] = np.log(i["Deaths"])
        i['Deaths'] = i['Deaths'].diff()
        
        
def clean_na (df):
    """ it takes a df and replace the values inf and - inf,
    converted to NA and then drop it"""
    df_copy = df.copy()
    df_copy["Deaths"] = df_copy["Deaths"].replace([np.inf, -np.inf], np.nan).dropna()
    return df_copy

def autocorrelation (list_):
    """ takes a list of dfs and return a auntocorrelation plot"""
    for df in list_:
        plot = plot_acf(df["Deaths"], alpha=.05)
    
    return plot

def partial_autocorrealtion (list_):
    """ it takes a list of dfs and return a partial autocorrelation plot"""
    for df in list_:
        plot = plot_pacf(df['Deaths'], alpha=.05, lags=20)
        warnings.filterwarnings("ignore")

    return plot

def predictions( list_ , list_2):
    """ it takes a list of states  and a list of dfs and return the predict value,
     the expected value and the MSE for each State"""
    for i ,u  in zip(list_ , list_2):
        X = i['Deaths']
        # size = int(len(X) * 0.66)
        train, test = X[0:9], X[9:len(X)]
        history = [x for x in train]
        predictions = list()
        try:
            for t in test:
                model = ARIMA(history, order=(0,0,1))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = t
                history.append(obs)
                print(f' State : {u} , predicted : {yhat} ,expected= {obs} ')
            error = mean_squared_error(test, predictions)
            print('Test MSE: %.3f' % error)
        
        
        except ValueError:
            pass
        

def Arima_model(list_df):

    
    """ It takes a lisst of dfs and return the ARIMA summary for each State"""

    list_of_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
       'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
       'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
       'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
       'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
       'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
       'New Jersey', 'New Mexico', 'New York', 'North Carolina',
       'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
       'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
       'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
       'West Virginia', 'Wisconsin', 'Wyoming']
    try:
        for i , u in zip(list_df , list_of_states):
            model = ARIMA(i['Deaths'], order=(0,0,1))
            model_fit = model.fit(disp=0)
            print(model_fit.summary())


    except ValueError:
        pass
    
def plot_prediction (list_df):
    warnings.filterwarnings("ignore")

    
    """ It takes a lisst of dfs and return the ARIMA summary for each State"""
    pd.plotting.register_matplotlib_converters()

    list_of_states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
       'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
       'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
       'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
       'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
       'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
       'New Jersey', 'New Mexico', 'New York', 'North Carolina',
       'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
       'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
       'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
       'West Virginia', 'Wisconsin', 'Wyoming']
    try:
        for i , u in zip(list_df , list_of_states):
            model = ARIMA(i['Deaths'], order=(0,0,1))
            model_fit = model.fit(disp=0)
            pre_1 = model_fit.plot_predict()
            
            plt.show()


    except ValueError:
        pass    

# map plot functions

def procesing_data_by_year (df_ , year):
    """ Subseting df by year and adding a new column """
    df_ = df_.copy()
    df_ = df_.loc[df_["Year"] == year]
    df_['code'] = ['AL','AK','AZ','AR','CA','CO', 'CT','DE','DD','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME',
    'MD', 'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN',
     'TX','UT','VT','VA','WA','WV','WI','WY']
    return df_

def plot_map (_df , year):
    """ Map data by expecific year """
    fig = go.Figure(data=go.Choropleth(
    locations=_df['code'], # Spatial coordinates
    z = _df['Deaths'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Deaths",
))

    fig.update_layout(
    title_text = f'US {year} Death Overdoses by State',
    geo_scope='usa', # limite map scope to USA
)

    fig.show()
    return

