import pandas as pd

crops = pd.read_csv('crops.csv', skiprows= 8, index_col=0)
veggies = pd.read_csv('veggies.csv', skiprows = 8, index_col=0)

# print(crops)
# print(veggies)

prices = crops.merge(veggies, on = 'State')
print(prices)
# merge into one table
# create a dictionary mapping states to regions
# calculate the averages per region -- have zeros for null rows