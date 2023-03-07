import pandas as pd
import numpy as np

crops = pd.read_csv('crops.csv', skiprows= 8, index_col=0, skipfooter=2)
veggies = pd.read_csv('veggies.csv', skiprows = 8, index_col=0, skipfooter=2)

# print(crops)
# print(veggies)

prices = crops.merge(veggies, on = 'State') # combine vegetable and crop data into one table
prices = prices.fillna(0) # update NaNs to 0s 

# create a dictionary mapping states to regions
regions = {
    'Washington': 'Pacific', 'Oregon': 'Pacific', 'California': 'Pacific', 'Nevada': 'Mountain',
    'Idaho': 'Mountain', 'Montana': 'Mountain', 'Wyoming': 'Mountain', 'Utah': 'Mountain',
    'Colorado': 'Mountain', 'Alaska': 'Pacific', 'Hawaii': 'Pacific', 'Maine': 'New England',
    'Vermont': 'New England', 'New York': 'Middle Atlantic', 'New Hampshire': 'New England',
    'Massachusetts': 'New England', 'Rhode Island': 'New England', 'Connecticut': 'New England',
    'New Jersey': 'Middle Atlantic', 'Pennsylvania': 'Middle Atlantic', 'North Dakota': 'West North Central',
    'South Dakota': 'West North Central', 'Nebraska': 'West North Central', 'Kansas': 'West North Central',
    'Minnesota': 'West North Central', 'Iowa': 'West North Central', 'Missouri': 'West North Central', 'Wisconsin': 'East North Central',
    'Illinois': 'East North Central', 'Michigan': 'East North Central', 'Indiana': 'East North Central', 'Ohio': 'East North Central',
    'West Virginia': 'South Atlantic', 'District of Columbia': 'South Atlantic', 'Maryland': 'South Atlantic',
    'Virginia': 'South Atlantic', 'Kentucky': 'East South Central', 'Tennessee': 'East South Central', 'North Carolina': 'South Atlantic',
    'Mississippi': 'East South Central', 'Arkansas': 'West South Central', 'Louisiana': 'West South Central', 'Alabama': 'East South Central',
    'Georgia': 'South Atlantic', 'South Carolina': 'South Atlantic', 'Florida': 'South Atlantic', 'Delaware': 'South Atlantic',
    'Arizona': 'Mountain', 'New Mexico': 'Mountain', 'Oklahoma': 'West South Central',
    'Texas': 'West South Central'
    }

prices['Region'] = np.nan
for state in prices.index:
    # print(state)
    prices.loc[state, 'Region'] = regions[state]

prices_by_region = prices.groupby('Region').mean()
print(prices_by_region)

# calculate the averages per region -- have zeros for null rows
# don't want to penalize 