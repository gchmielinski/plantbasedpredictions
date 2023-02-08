import requests 
import pandas as pd

# API request from FoodData Central

response = requests.get('https://api.nal.usda.gov/fdc/v1/foods/search?query=plant%20based&dataType=Branded&pageSize=25&api_key=yrdv403X541C9U4wfmmC4uLXccwXei1sSbxwH6pE')

result = response.json()

# df of all foods that match query
result_df = pd.DataFrame(result['foods'])
# print(result_df)

# nutrition information for a specific food from the query
nutrient_dict = result_df.loc[1, 'foodNutrients']
nutrient_df = pd.DataFrame(nutrient_dict)
# print(result_df.loc[0, 'ingredients'])

# import protein primer table
protein_primer = pd.read_csv('protein_primer.csv', index_col=0)
protein_primer_short = protein_primer.loc[['Wheat', 'Rice','Potato','Pea'],:]
# print(protein_primer_short)

#import emissions excel
emissions = pd.read_csv('emission2.csv', index_col=0)
# print(emissions)

# potatoes, rice, wheat, peas 
emissions_short = emissions.loc[['Wheat & Rye (Bread)', 'Rice','Potatoes','Peas'],:]
emissions_short2 = emissions_short.rename(index={'Wheat & Rye (Bread)': 'Wheat', 'Potatoes':'Potato','Peas':'Pea'})
# print(emissions_short)

# merge emissions and primer tables

result = pd.concat([protein_primer_short, emissions_short2], axis=1)
print(result)

# for later attempt to load Top Plant-Based Brands data
# response = requests.get('https://api.nal.usda.gov/fdc/v1/foods/search?query=%20&dataType=Branded&pageSize=25&pageNumber=2&sortBy=dataType.keyword&sortOrder=asc&brandOwner=Beyond%20Meat%2C%20Inc.&api_key=yrdv403X541C9U4wfmmC4uLXccwXei1sSbxwH6pE')