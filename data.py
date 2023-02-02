import requests 
import pandas as pd

# API request from FoodData Central

response = requests.get('https://api.nal.usda.gov/fdc/v1/foods/search?query=plant%20based&dataType=Branded&pageSize=25&api_key=yrdv403X541C9U4wfmmC4uLXccwXei1sSbxwH6pE')
result = response.json()

# df of all foods that match query
result_df = pd.DataFrame(result['foods'])

# nutrition information for a specific food from the query
nutrient_dict = result_df.loc[1, 'foodNutrients']
nutrient_df = pd.DataFrame(nutrient_dict)

print(result_df.loc[0, 'ingredients'])