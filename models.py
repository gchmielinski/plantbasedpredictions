import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from patsy import dmatrix
from sklearn.svm import LinearSVR #from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import plotly.figure_factory as ff
import plotly.express as px

from urllib.request import urlopen
import json

def preprocess():
    '''
    USDA data
    Market Profile Data
    Simmons Survey Responses

    Returns an aggregate dataframe with each row representing a county
    '''
    # import USDA data
    atlas = pd.read_csv('FoodAccessResearchAtlasData2019.csv', header = 0, usecols = [0, 1, 2, 31, 57])
    atlas.rename(columns={'lapophalf':'LAPOPHALF', 'lapop1':'LAPOP1'}, inplace = True)
    # convert the Census Tract numbers to FIPS codes
    atlas['CensusTract'] = atlas['CensusTract'].astype(str)
    atlas['FIPS'] = atlas['CensusTract'].str[:5]
    # sum over each county
    county_access = atlas.groupby('FIPS').sum()

    # import market profile data
    market = pd.read_csv('marketprofile.csv', usecols = [0, 1, 15, 16, 17, 19, 21, 22, 23, 24, 26, 38], header = 1, skiprows = 0)

    # import survey data
    no = pd.read_csv('surveyNo.csv',  header = 3, skiprows = [0, 3], skipfooter = 1, usecols = [0,1], engine='python')
    yes = pd.read_csv('surveyYes.csv',  header = 3, skiprows = [0, 3], skipfooter = 1, usecols = [0,1], engine='python')

    # rename Target Sample
    no.rename(columns={'Target Sample':'nos'}, inplace = True)
    yes.rename(columns={'Target Sample':'yeses'}, inplace = True)   

    # join market and county_access on FIPS code
    market['Geo_FIPS'] = market['Geo_FIPS'].astype(str)
    df = county_access.merge(market, left_on='FIPS', right_on='Geo_FIPS')

    # join df and surveys on Geographies and Geo_NAME
    df['Geo_NAME'] = df['Geo_NAME'].str.upper()
    df = df.merge(no, left_on = 'Geo_NAME', right_on = 'Geographies')
    df = df.merge(yes, left_on = 'Geo_NAME', right_on = 'Geographies')

    # df = df.drop(columns = ['Geo_FIPS', 'Geo_NAME', 'Geographies_x', 'Geographies_y'])

    # create target variable
    df['MEATALTPCT'] = np.nan
    df['nos'] = df['nos'].str.replace(',', '').astype(int)
    df['yeses'] = df['yeses'].str.replace(',', '').astype(int)

    for row in df.index:
        df.loc[row, 'MEATALTPCT'] = df.loc[row,'yeses']/( df.loc[row,'yeses'] + df.loc[row,'nos'])

    return df


def explore(df):   
    '''
    Explore the MEATALTPCT target variable

    Returns the linear regression summary and the standardized dataframe
    Outcome:
        highest correlation between population-based variables (excluding rural)
        high correlation between population-based variables (excluding rural) and expenditure of food at home
        high correlation between population-based variables (excluding rural) and poverty population
        meataltpct has a low correlation with all other variables ...
    '''
    df = df.drop(columns = ['yeses', 'nos'])
    # scale all (numeric) features
    scaler = StandardScaler()
    num = df.select_dtypes(include=['float64', 'int64']).columns
    df[num] = scaler.fit_transform(df[num])

    X = df.loc[:, 'LAPOPHALF':'FOODCPI']
    y = df['MEATALTPCT']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    # print(model.summary())

    # scatterplots showing some of the lower p-values are very uninformative

    # plot the correlation matrix as a heatmap
    # sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    # plt.show()


    # plt.scatter(df['FOODHOME'], df['TOT_POP1'])
    # plt.show()
    # plt.scatter(df['FOODHOME'], df['LAPOP1'])
    # plt.show()

    # boxplot of MEATALTPCT
    # sns.boxplot(y=df['MEATALTPCT'])
    # plt.show()

    return model.summary(), df

def map(df):
    '''
    choropleth map
    '''
    # fig = ff.create_choropleth(fips=df['Geo_FIPS'], values=df['MEATALTPCT'])
    # fig.layout.template = None
    # fig.show()
    values = df['yeses'].tolist()
    fips = df['Geo_FIPS'].tolist()

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    counties["features"][0]

    fig = px.choropleth(df, geojson=counties, locations='Geo_FIPS', color='MEATALTPCT',
                           color_continuous_scale="Bluyl",
                           range_color=(0, df['MEATALTPCT'].max()),
                           scope="usa",
                           labels={'unemp':'unemployment rate'}
                          )
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

    # no clear pockets of higher colors

def svr(df):

    # Separate the target variable from the features
    # drop total population since it is multicollinear with urban and rural pop
    X = df.loc[:, ['LAPOPHALF', 'LAPOP1', 'URBANPOP', 'RURALPOP', 'MEDHHINC', 
       'POV_TOTAL', 
       'CULT_INDX', 'RELIG_INDX', 'EDU_INDX', 
       'FOODHOME', 'FOODCPI', 'MEATALTPCT']]
    y = df['MEATALTPCT']

    # Grid Search CV
    param_grid = {'C': [0.1, 1, 10, 100],
              'epsilon': [0, 0.01, 0.1, 1, 10]}
    cv = GridSearchCV(LinearSVR(random_state=0, max_iter = 1000), param_grid, cv=5)
    cv.fit(X,y)
    print(pd.DataFrame(cv.cv_results_))

    # would not converge, cannot interpret output

def brand_preprocess():
    # load all csvs 
    gardein = clean_brand('GARDEIN.csv', 'gardein')
    boca = clean_brand('BOCA.csv', 'boca')
    yves = clean_brand('YVES.csv', 'yves')
    gardenburger = clean_brand('GARDENBURGER.csv', 'gardenburger')
    lightlife = clean_brand('LIGHTLIFE.csv', 'lightlife')
    tofurky = clean_brand('TOFURKY.csv', 'tofurky')
    morningstar = clean_brand('MORNINGSTAR.csv', 'morningstar')
    amys = clean_brand('AMYS.csv', 'amys')
    quorn = clean_brand('QUORN.csv', 'quorn')

    df_list = [gardein, boca, yves, gardenburger, lightlife, tofurky, morningstar, amys, quorn]
    
    # merge all tables into one aggregate dataframe with each row being a county and each column representing a brand
    merge_col = 'Geographies'
    merged_df = df_list[0]

    for i in range(1,len(df_list)):
        merged_df = merged_df.merge(df_list[i], how='inner', left_index = True, right_index = True)
        if i % 2 == 0:
            merged_df = merged_df.drop(columns='Geographies_x')
        else:
            merged_df = merged_df.drop(columns='Geographies_y')

    merged_df = merged_df.set_index('Geographies')
    print(merged_df)


def clean_brand(fn, brand):
    df = pd.read_csv(fn, header = 3, skiprows = [0, 3], skipfooter = 1, usecols = [0,1], engine='python')
    df.rename(columns={'Target Sample':brand}, inplace = True)
    return df

def main():
    df = preprocess()
    print(df.describe())
    # df1 = df.drop(columns = ['Geo_FIPS', 'Geo_NAME', 'Geographies_x', 'Geographies_y'])
    # brand_preprocess()
    # linear_model, df1 = explore(df1)
    # print(df.describe())
    # map(df)
    # feature_selection(df1)
    # svr(df)



main()