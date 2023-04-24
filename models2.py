import pandas as pd
import numpy as np

# linear regression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# map
# import plotly.figure_factory as ff2
import plotly.express as px
from urllib.request import urlopen
import json

# svr
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR 

# gam
from patsy import dmatrix
from sklearn.linear_model import LinearRegression

def brand_df():
    '''
    Import identical csvs from brands gardein, boca, yves, gardenburger, lightlife, tofurky, morningstar, amys, and quorn

    Returns a single dataframe with each row representing a zip code 
    '''
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
            merged_df = merged_df.drop(columns=['ZIP_x', 'Geographies_x'])
        else:
            merged_df = merged_df.drop(columns=['ZIP_y', 'Geographies_y'])

    return merged_df

def clean_brand(fn, brand):
    df = pd.read_csv(fn, header = 3, skiprows = [0, 3], skipfooter = 1, usecols = [0,1], engine='python')
    df[['ZIP', 'NAME']] = df['Geographies'].str.split(' - ', expand=True)
    df.rename(columns={'Target Sample':brand}, inplace = True)
    df = df.drop(columns='NAME')
    return df

def yes_df():
    '''
    '''   
    yes = pd.read_csv('surveyYes.csv',  header = 3, skiprows = [0, 3], skipfooter = 1, usecols = [0,1], engine='python')
    yes[['ZIP', 'NAME']] = yes['Geographies'].str.split(' - ', expand=True)
    # for index in yes.index:
    #     if yes.loc[index, 'ZIP'][0] == '9':
    #         print(yes.loc[index, 'ZIP'])
    yes.rename(columns={'Target Sample':'yeses'}, inplace = True)
    yes = yes.drop(columns=['NAME', 'Geographies'])
    return yes

def no_df():
    '''
    '''
    no = pd.read_csv('surveyNo.csv',  header = 3, skiprows = [0, 3], skipfooter = 1, usecols = [0,1], engine='python')
    no[['ZIP', 'NAME']] = no['Geographies'].str.split(' - ', expand=True)
    # print(len(no["ZIP"].index))
    no.rename(columns={'Target Sample':'nos'}, inplace = True)
    no = no.drop(columns=['NAME', 'Geographies'])
    return no

def marketprofile_df():
    return pd.read_csv('marketprofile.csv', header = 0, usecols = [0,1,2,12,13,14,15,16,17,18,22,29], engine='python')

def atlas_df():
    # import USDA data
    atlas = pd.read_csv('FoodAccessResearchAtlasData2019.csv', header = 0, usecols = [0, 1, 2, 31, 57])
    atlas.rename(columns={'lapophalf':'LAPOPHALF', 'lapop1':'LAPOP1'}, inplace = True)
    # convert the Census Tract numbers to FIPS codes
    atlas['CensusTract'] = atlas['CensusTract'].astype(str)
    atlas['FIPS'] = atlas['CensusTract'].str[:5]
    # sum over each county
    return atlas.groupby('FIPS').sum()

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

def convert_fips2(df):
    conversion = pd.read_csv('ZIP-COUNTY-FIPS_2017-06.csv')
    conversion['ZIP'] = conversion['ZIP'].astype(str)
    conversion['STCOUNTYFP'] = conversion['STCOUNTYFP'].astype(str)
    join_FIPS = pd.merge(df, conversion, on='ZIP', how='inner')
    join_FIPS = join_FIPS.drop(columns = ['Geographies', 'ZIP', 'COUNTYNAME', 'STATE', 'CLASSFP'])
    return join_FIPS.groupby('STCOUNTYFP').sum()

def pref_clean(market, atlas, yes, no):
    market['Geo_FIPS'] = market['Geo_FIPS'].astype(str)
    df = market.merge(atlas, left_on = 'Geo_FIPS', right_on = 'FIPS', how = 'inner')
    df = df.merge(yes, left_on = 'Geo_FIPS', right_on = 'STCOUNTYFP', how = 'inner')
    df = df.merge(no, left_on = 'Geo_FIPS', right_on = 'STCOUNTYFP', how = 'inner')
    return df

def brand_clean():
    pass

def explore(df):
    df['MEATALTPCT'] = np.nan
    for row in df.index:
        df.loc[row, 'MEATALTPCT'] = df.loc[row,'yeses']/( df.loc[row,'yeses'] + df.loc[row,'nos'])
    df1 = df.drop(columns = ['nos', 'Geo_FIPS', 'Geo_NAME', 'Geo_QNAME', 'MEATALTPCT'])
    scaler = StandardScaler()
    num = df1.select_dtypes(include=['float64', 'int64']).columns
    df1[num] = scaler.fit_transform(df1[num])

    X = df1.loc[:, ['LAPOPHALF', 'LAPOP1', 'URBANPOP', 'RURALPOP', 'MEDHHINC', 
       'POV_TOTAL', 'CULT_INDX', 'RELIG_INDX', 'EDU_INDX', 
       'FOODHOME', 'FOODCPI']]
    y = df1['yeses']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

    df2 = df.drop(columns = ['nos', 'yeses', 'Geo_FIPS', 'Geo_NAME', 'Geo_QNAME'])
    scaler = StandardScaler()
    num = df2.select_dtypes(include=['float64', 'int64']).columns
    df2[num] = scaler.fit_transform(df2[num])

    X = df2.loc[:, ['LAPOPHALF', 'LAPOP1', 'URBANPOP', 'RURALPOP', 'MEDHHINC', 
       'POV_TOTAL', 'CULT_INDX', 'RELIG_INDX', 'EDU_INDX', 
       'FOODHOME', 'FOODCPI']]
    y = df2['MEATALTPCT']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    # print(model.summary())

    return df1, df2

def svr(df):
    # Separate the target variable from the features
    # drop total population since it is multicollinear with urban and rural pop
    # print(df.columns)
    X = df.loc[:, ['LAPOPHALF', 'LAPOP1', 'URBANPOP', 'RURALPOP', 'MEDHHINC', 
       'POV_TOTAL', 'CULT_INDX', 'RELIG_INDX', 'EDU_INDX', 
       'FOODHOME', 'FOODCPI']]
    y = df['yeses']

    # Grid Search CV
    param_grid = {'C': [0.1, 1, 10, 100],
              'epsilon': [0, 0.01, 0.1, 1, 10]}
    cv = GridSearchCV(LinearSVR(random_state=0, max_iter = 1000), param_grid, cv=5)
    cv.fit(X,y)
    print(pd.DataFrame(cv.cv_results_))

def map(df):
    '''
    choropleth map
    '''

    scaler = StandardScaler()
    num = df.select_dtypes(include=['float64', 'int64']).columns
    df[num] = scaler.fit_transform(df[num])

    values = df['yeses'].tolist()
    # fips = df['Geo_FIPS'].tolist()

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    counties["features"][0]

    fig = px.choropleth(df, geojson=counties, locations='STCOUNTYFP', color='yeses',
                           color_continuous_scale="Bluyl",
                           range_color=(0, df['yeses'].max()),
                           scope="usa")

    fig.show()

def gam(df):
    boundaries = pd.read_csv('us-county-boundaries.csv', header = 0, usecols = [0,1,2,3])

    # add leading zeros to county and create FIPS code from state and county codes
    boundaries['FIPS'] = np.nan
    for i in boundaries.index:
        county = boundaries.loc[i, 'COUNTYFP']
        zeros = 3 - len(str(county))
        boundaries.loc[i, 'FIPS'] = boundaries.loc[i,'STATEFP'].astype(str) + '0'*zeros + boundaries.loc[i,'COUNTYFP'].astype(str)

    boundaries['LAT'] = np.nan
    boundaries['LONG'] = np.nan
    for i in boundaries.index:
        boundaries.loc[i,'LAT'], boundaries.loc[i,'LONG'] = boundaries.loc[i,'Geo Point'].split(', ')
    
    boundaries['LAT'] = boundaries['LAT'].astype(float)
    boundaries['LONG'] = boundaries['LONG'].astype(float)
    # create the combined df
    df['Geo_FIPS'] = df['Geo_FIPS'].astype(int)
    boundaries['FIPS'] = boundaries['FIPS'].astype(int)
    df_merge = pd.merge(df, boundaries, left_on = 'Geo_FIPS', right_on = 'FIPS', how = 'inner')

    # fit the degrees of freedom

    # Create design matrix and target variable
    # cr() vs bs()?
    LAT_spl = dmatrix('cr(LAT, df = 4)', df_merge, return_type="dataframe").drop('Intercept', axis=1)
    LONG_spl = dmatrix('cr(LONG, df = 4)', df_merge, return_type="dataframe").drop('Intercept', axis=1)
    X = pd.concat([LAT_spl, LONG_spl], axis=1)
    y = df_merge['yeses']

    gam = LinearRegression(fit_intercept=True)
    gam.fit(X, y)

    print(gam.intercept_)
    print(gam.coef_)
    # what is the conclusion of this?

def main():
    brands = brand_df()
    yeses = yes_df()
    nos = no_df()
    marketprofile = marketprofile_df()
    brands2 = convert_fips2(brands)
    atlas = atlas_df()
    yeses2 = convert_fips(yeses, 'yeses')
    nos2 = convert_fips(nos, 'nos')
    # print(yeses2[yeses2['STATE'] == 'FL'])
    pref_df = pref_clean(marketprofile, atlas, yeses2, nos2)

    # df1, df2 = explore(pref_df) # number of Yes survey responses has a stronger linear relationship
    # map(yeses2[yeses2['STATE'] == 'FL'])
    # svr(df1)
    gam(pref_df)

    


main()