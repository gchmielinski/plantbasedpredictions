# from statistics import LinearRegression
import pandas as pd
import numpy as np

# regression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from statsmodels.othermod.betareg import BetaModel
# from statsmodels.genmod import families

# correlation map
import seaborn as sns
import matplotlib.pyplot as plt


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
    # plot the correlation matrix as a heatmap
    sns.heatmap(df1.corr(method = "spearman"), annot=True, cmap='coolwarm')
    plt.show()


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

    # preliminary analysis
    explore(df)

    # export to 


main()