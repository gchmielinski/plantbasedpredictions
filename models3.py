# from statistics import LinearRegression
import pandas as pd
import numpy as np

# regression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
# from statsmodels.genmod.families import BetaModel
# from statsmodels.base.model import GeneralLikelihoodModel
from statsmodels.othermod.betareg import BetaModel
# from statsmodels.genmod import families

# correlation map
import seaborn as sns
import matplotlib.pyplot as plt

# svm
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# map
import plotly.express as px
from urllib.request import urlopen
import json

# gam
from pygam import LogisticGAM, s
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def yes_df():
    '''
    '''   
    # load data
    yes = pd.read_csv('surveyYes.csv',  header = 3, skiprows = [0, 3], skipfooter = 1, usecols = [0,1], engine='python')

    # split geographies into zip codes and county names
    yes[['ZIP', 'NAME']] = yes['Geographies'].str.split(' - ', expand=True)

    yes.rename(columns={'Target Sample':'yeses'}, inplace = True)
    yes = yes.drop(columns=['NAME', 'Geographies'])
    return yes

def no_df():
    '''
    '''
    # load data
    no = pd.read_csv('surveyNo.csv',  header = 3, skiprows = [0, 3], skipfooter = 1, usecols = [0,1], engine='python')

    # split geographies into zip codes and county names    
    no[['ZIP', 'NAME']] = no['Geographies'].str.split(' - ', expand=True)

    no.rename(columns={'Target Sample':'nos'}, inplace = True)
    no = no.drop(columns=['NAME', 'Geographies'])
    return no

def marketprofile_df():
    '''
    '''
    df = pd.read_csv('marketprofile.csv', header = 0, usecols = [0,1,2,11,12,14,15,22,29], engine='python') 
    df['URBANPROP'] = df['URBANPOP']/df['TOT_POP1']
    df['POVPROP'] = df['POV_TOTAL']/df['TOT_POP1']
    df = df.drop(columns = ['URBANPOP', 'TOT_POP1', 'POV_TOTAL'])
    return df

def access_df():
    # import USDA data
    access = pd.read_csv('FoodAccessResearchAtlasData2019.csv', header = 0, usecols = [0, 1, 2, 32, 58])

    access.rename(columns={'lapophalfshare':'LAPOPHALF', 'lapop1share':'LAPOP1'}, inplace = True)

    # # convert the Census Tract numbers to FIPS codes
    access['CensusTract'] = access['CensusTract'].astype(str)
    access['FIPS'] = access['CensusTract'].str[:5]

    # average over each county
    return access.groupby('FIPS').mean()

def convert_fips(df, col):
    '''
    convert Simmons data, downloaded per zip code, to countywide FIPS codes

    use conversion sheet 
    sum samples for each zip code across FIPS codes
    '''
    conversion = pd.read_csv('ZIP-COUNTY-FIPS_2017-06.csv')
    conversion['ZIP'] = conversion['ZIP'].astype(str)
    conversion['STCOUNTYFP'] = conversion['STCOUNTYFP'].astype(str)

    join_FIPS = pd.merge(df, conversion, on='ZIP', how='inner')
    join_FIPS[col] = join_FIPS[col].str.replace(',', '').astype(int)

    return join_FIPS.groupby('STCOUNTYFP').sum()

def geography():
    '''
    '''
    geography = pd.read_csv('us-county-boundaries.csv', header = 0, usecols = [0,1,2,3, 8])

    # add leading zeros to county and create FIPS code from state and county codes
    geography['FIPS'] = np.nan
    for i in geography.index:
        county = geography.loc[i, 'COUNTYFP']
        zeros = 3 - len(str(county))
        geography.loc[i, 'FIPS'] = geography.loc[i,'STATEFP'].astype(str) + '0'*zeros + geography.loc[i,'COUNTYFP'].astype(str)

    geography['LAT'] = np.nan
    geography['LONG'] = np.nan
    for i in geography.index:
        geography.loc[i,'LAT'], geography.loc[i,'LONG'] = geography.loc[i,'Geo Point'].split(', ')
    
    geography['LAT'] = geography['LAT'].astype(float)
    geography['LONG'] = geography['LONG'].astype(float)

    return geography

def clean(market, access, yeses, nos, geography):
    market['Geo_FIPS'] = market['Geo_FIPS'].astype(str)
    # print(len(market.index))
    df = market.merge(access, left_on = 'Geo_FIPS', right_on = 'FIPS', how = 'inner')
    
    df = df.merge(yeses, left_on = 'Geo_FIPS', right_on = 'STCOUNTYFP', how = 'inner')
    df = df.merge(nos, left_on = 'Geo_FIPS', right_on = 'STCOUNTYFP', how = 'inner')

    df['MEATALTPCT'] = np.nan
    for row in df.index:
        df.loc[row, 'MEATALTPCT'] = df.loc[row,'yeses']/( df.loc[row,'yeses'] + df.loc[row,'nos']) 

    df['Geo_FIPS'] = df['Geo_FIPS'].astype(int)
    geography['FIPS'] = geography['FIPS'].astype(int)
    df_merge = pd.merge(df, geography, left_on = 'Geo_FIPS', right_on = 'FIPS', how = 'inner')

    df_merge = df_merge.drop(columns = ['FIPS', 'STATEFP', 'COUNTYFP', 'yeses', 'nos'])
    df_merge = df_merge.rename(columns ={'Geo_NAME':'NAME', 'Geo_QNAME':'NAME_', 'Geo_FIPS':'FIPS'})
    return df_merge

def explore(df):
    '''
    preliminary analysis
    '''
    df1 = df.drop(columns = ['NAME', 'NAME_', 'Geo Point', 'Geo Shape', 'FIPS', 'LAT', 'LONG'])
    df1 = df1.dropna()


    X = df1.loc[:, ['MEDHHINC', 'FOODCPI', 'FOODHOME', 'POVPROP',
       'URBANPROP', 'LAPOPHALF', 'LAPOP1']]

    scaler = StandardScaler()
    num = X.select_dtypes(include=['float64', 'int64']).columns
    X[num] = scaler.fit_transform(X[num])

    y = df1['MEATALTPCT']

    # build a beta regression model
    # model = BetaModel(y, X).fit()
    # print(model.summary()) 

    # linear regression
    # X = sm.add_constant(X)
    # model = sm.OLS(y, X).fit()
    # print(model.summary())

    # plot the correlation matrix as a heatmap
    # sns.heatmap(df1.corr(), annot=True, cmap='coolwarm')
    # plt.show()

def svm(df):
    df1 = df.drop(columns = ['NAME', 'NAME_', 'Geo Point', 'Geo Shape', 'FIPS', 'LAT', 'LONG'])
    df1 = df1.dropna()


    X = df1.loc[:, ['MEDHHINC', 'FOODCPI', 'FOODHOME', 'POVPROP',
       'URBANPROP', 'LAPOPHALF', 'LAPOP1']]

    scaler = StandardScaler()
    num = X.select_dtypes(include=['float64', 'int64']).columns
    X[num] = scaler.fit_transform(X[num])

    y = df1['MEATALTPCT']

    # Grid Search CV
    param_grid = {'C': [0.1, 1, 10, 100],
              'epsilon': [0, 0.01, 0.1, 1, 10]}
    cv = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5)
    cv.fit(X,y)
    # print(pd.DataFrame(cv.cv_results_))

    # 
    best_model = cv.best_estimator_
    c, eps = best_model.C, best_model.epsilon
    # print(best_model.coef_)
    print(c, eps)

    # 0.1 0.01

    # build a new svm and print coefficients
    # model.coef_, which requires a linear kernel

def map(df):
    '''
    choropleth map
    '''

    print(df)
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    counties["features"][0]

    fig = px.choropleth(df, geojson=counties, locations='FIPS', color='MEATALTPCT',
                           color_continuous_scale="Bluyl",
                           range_color=(0, df['MEATALTPCT'].max()),
                           scope="usa")

    fig.show()

def gam(df):
    '''
    '''
    df = df.dropna()
    X = df.loc[:, ['LAT', 'LONG']].values
    y = df['MEATALTPCT']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    gam = LogisticGAM(s(0, by = 1)).gridsearch(X, y) # combine spline for lat/long, 

    # df = np.arange(1, 2)
    # param_grid = {'s(0, 1)': df}
    # grid_search = GridSearchCV(gam, param_grid=param_grid, cv=5, scoring='accuracy')

    # grid_search.fit(X_train, y_train)
    # print(grid_search.cv_results_)
    gam.accuracy(X, y)
    gam.summary()
    
    # print(cross_val_score(gam, X, y, cv=5)) # disable estimate R2 if possible, check data (transform response into binary), use multiple thresholds and evaluate


def main():
    # clean MRI Simmons Survey data
    yeses = yes_df()
    nos = no_df()

    # U.S. data
    market = marketprofile_df()
    access = access_df()

    # convert zip codes to counties
    yeses = convert_fips(yeses, 'yeses')
    nos = convert_fips(nos, 'nos')

    geographies = geography()

    # clean(market, access, yeses, nos, geography)
    df = clean(market, access, yeses, nos, geographies)
    # df = df.drop(columns = 'Geo Shape')
    # df.to_csv('clean.csv', index=False)
    # for i in df.index:
    #     print(df.loc[i, :])

    # preliminary analysis
    # explore(df)

    # svm
    # svm(df)

    # map(df)

    # test just Californa, for example
    # print(df[df['STUSAB'] == 'AZ'])
    # map(df[df['STUSAB'] == 'AZ'])

    # gam(df)

main()