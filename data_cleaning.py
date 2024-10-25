import pandas as pd

# Clean output data files
dat = pd.read_csv('./results/rota_strains_infected_all_1_0.1_1_1_1_0_0.5_1.csv')
print(dat.head())

# Look at all years
dat['Strain3'] = 'Other'
dat.loc[dat['Strain'] == 'G1P8A1B1', 'Strain3'] = 'G1P8'
dat.loc[dat['Strain'] == 'G2P4A1B1', 'Strain3'] = 'G2P4'
dat.loc[dat['Strain'] == 'G9P8A1B1', 'Strain3'] = 'G9P8'

dat['Year'] = dat['CollectionTime'].apply(lambda x: int(x))
YearlyGenoDat = dat.groupby(['Year', 'Strain3']).agg(Geno_cases=('id', 'nunique')).reset_index()
YearlyCases = dat.groupby('Year').agg(All_cases=('id', 'nunique'), mean_pop=('PopulationSize', 'mean')).reset_index()
CasesGeno = pd.merge(YearlyGenoDat, YearlyCases, on='Year')
CasesGeno['geno_prop'] = CasesGeno['Geno_cases'] / CasesGeno['All_cases']

# Subset to years 1-9 for now
initial8 = dat[(dat['CollectionTime'] < 9) & (dat['CollectionTime'] > 1)]
print(initial8['Strain'].value_counts())
initial8['Strain3'] = 'Other'
initial8.loc[initial8['Strain'] == 'G1P8A1B1', 'Strain3'] = 'G1P8'
initial8.loc[initial8['Strain'] == 'G2P4A1B1', 'Strain3'] = 'G2P4'
initial8.loc[initial8['Strain'] == 'G9P8A1B1', 'Strain3'] = 'G9P8'

GenoDist = initial8['Strain3'].value_counts().reset_index()
GenoDist.columns = ['Strain', 'Frequency']
total = GenoDist['Frequency'].sum()
GenoDist['Proportion'] = GenoDist['Frequency'] / total

# Working out the case age distribution
# First making new age bins
initial8['AgeCat'] = None
initial8.loc[initial8['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
initial8.loc[initial8['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
initial8.loc[initial8['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
initial8.loc[initial8['Age'] == '60+', 'AgeCat'] = '>=5 y'

# Now, getting cases by age bin
cases_summary = initial8.groupby(['AgeCat', 'CollectionTime']).agg(Cases_age=('id', 'nunique')).reset_index()
print(cases_summary.head())

# Now, getting total population by time point
pop_pooled = initial8.groupby(['CollectionTime', 'AgeCat']).apply(lambda x: x.head(1)).reset_index(drop=True)
pop_pooled2 = pop_pooled[['CollectionTime', 'AgeCat', 'PopulationSize']]

# Multiplying by fraction of the population in each age bin
pop_pooled2['Pop_Age'] = None
pop_pooled2.loc[pop_pooled2['AgeCat'] == '<1 y', 'Pop_Age'] = pop_pooled2['PopulationSize'] * 0.025
pop_pooled2.loc[pop_pooled2['AgeCat'] == '1-2 y', 'Pop_Age'] = pop_pooled2['PopulationSize'] * 0.025
pop_pooled2.loc[pop_pooled2['AgeCat'] == '2-5 y', 'Pop_Age'] = pop_pooled2['PopulationSize'] * 0.075
pop_pooled2.loc[pop_pooled2['AgeCat'] == '>=5 y', 'Pop_Age'] = pop_pooled2['PopulationSize'] * 0.875

# Merge incidence with the age and time appropriate denominator to get the ratio
AgeIncidence = pd.merge(cases_summary, pop_pooled2, on=['AgeCat', 'CollectionTime'])
AgeIncidence['IR_100k'] = (AgeIncidence['Cases_age'] / AgeIncidence['Pop_Age']) * 100000

# Get an average to calibrate to average pre-vaccine period
Incidence_dist = AgeIncidence.groupby('AgeCat').agg(meanIR=('IR_100k', 'mean')).reset_index()

print('Results:')
print(Incidence_dist)
print(GenoDist)