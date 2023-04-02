# packages
import pandas as pd

# load original simmons crosstab download without rows 0-6, 8-32, last 4 rows
df = pd.read_csv('demographicXmeatAlts.csv', header = 7,skiprows = range(7,28), skipfooter = 4, usecols = [0, 1, 2, 4, 6], engine = 'python')

# rename columns
cols = df.columns
df.rename(columns = {cols[0]:'demoVars', cols[1]:'measure', cols[2]:'total', cols[3]:'yesMeatAlt', cols[4]:'noMeatAlt'}, inplace = True)
# print(df.head())

# drop all but the 'Sample' rows
survey = df[df['measure'] == 'Sample']
# drop the 'measure' column
survey = survey.drop(columns = 'measure')

# split the demographic variables into individual columns
survey[['income', 'gender', 'age', 'raceX', 'race']] = survey.demoVars.str.split(" AND ", expand=True)
# drop the raceX column because it is a result of needing to split on "AND", drop 'demoVars' since no longer needed
survey = survey.drop(columns = ['raceX', 'demoVars'])

# format demographic variables
survey['income'] = survey['income'].str.replace('\( \(INDIVIDUAL EMPLOYMENT INCOME_', '')
survey['income'] = survey['income'].str.replace(')', '')
survey['age'] = survey['age'].str.replace('\(AGE_','')
survey['age'] = survey['age'].str.replace(')', '')
survey['gender'] = survey['gender'].str.replace('\(GENDER_', '')
survey['gender'] = survey['gender'].str.replace(')', '')
survey['race'] = survey['race'].str.replace('\(RACE ', '')
survey['race'] = survey['race'].str.replace('HISPANIC/LATINO/SPANISH ORIGIN_', '')
survey['race'] = survey['race'].str.replace(')', '')

# convert survey response values to numeric columns
survey[["total", 'yesMeatAlt', 'noMeatAlt']] = survey[["total", 'yesMeatAlt', 'noMeatAlt']].apply(pd.to_numeric)
# drop where there are no respondents for a given demographic profile
survey = survey[survey['total'] > 0]

# df = pd.DataFrame(columns = ['Name', 'Scores', 'Questions'])
# print(survey.head())
survey2 = pd.DataFrame(columns = ['yesMeatAlt', 'noMeatAlt', 'income', 'gender', 'age', 'race'])

# {'Name' : 'Anna', 'Scores' : 97, 'Questions' : 2200}, 
                # ignore_index = True
for row in survey.index:
    for response in range(survey.loc[row, 'yesMeatAlt']):
        new_row = pd.Series({'yesMeatAlt':1, 'noMeatAlt':0, 'income':survey.loc[row, 'income'], 
        'gender':survey.loc[row, 'gender'], 'age':survey.loc[row, 'age'], 'race':survey.loc[row, 'race']})
        survey2 = pd.concat([survey2, new_row.to_frame().T], ignore_index = True)
    for response in range(survey.loc[row, 'noMeatAlt']):
        new_row = pd.Series({'yesMeatAlt':0, 'noMeatAlt':1, 'income':survey.loc[row, 'income'], 
        'gender':survey.loc[row, 'gender'], 'age':survey.loc[row, 'age'], 'race':survey.loc[row, 'race']})
        survey2 = pd.concat([survey2, new_row.to_frame().T], ignore_index = True)

survey2.to_csv('simmonsdata.csv', index = False)