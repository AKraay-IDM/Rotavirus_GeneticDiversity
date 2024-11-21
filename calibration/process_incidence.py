"""
Process incidence from the model and data
"""

import sciris as sc
import pandas as pd
import numpy as np
import warnings

# Set current folder
thisdir = sc.thispath(__file__)

# Disable annoying warnings
warnings.simplefilter("ignore", FutureWarning)
pd.options.mode.chained_assignment = None
pd.set_option('future.no_silent_downcasting', True)

def process_data(filename=None, sheet=None):
    """
    Extract and process experimental data
    """
    if filename is None:
        filename = thisdir / 'CalibrationDatafile_prevax 3.xlsx'
    if sheet is None:
        sheet = 'Matlab_ageincidence'
    data = sc.dataframe.read_excel(filename, sheet_name=sheet)

    # Standardize data format
    age_mapping = {
        '[0, 1)': 0,
        '[1, 2)': 1,
        '[2, 5)': 2,
        '[5, 125)': 5,
    }
    ages = data['Age'].replace(age_mapping)
    df = sc.dataframe(dict(ages=ages, inci=data['Cases per 100k']))
    df = df.sort_values(by='ages').reset_index(drop=True)
    return df


def process_model(dat=None, popsize=None, verbose=False):
    """
    Extract and process data from the model
    """

    FUDGE = 1.0 # Adjust the fudge factor -- for testing only!!!!

    # Load the data
    if dat is None:
        dat = pd.read_csv(thisdir / '../results/rota_strains_infected_all_1_0.1_2_1_1_0_0.5_1.csv')

    # Look at all years
    dat['Strain3'] = 'Other'
    dat.loc[dat['Strain'] == 'G1P8A1B1', 'Strain3'] = 'G1P8'
    dat.loc[dat['Strain'] == 'G2P4A1B1', 'Strain3'] = 'G2P4'
    dat.loc[dat['Strain'] == 'G9P8A1B1', 'Strain3'] = 'G9P8'

    dat['Year'] = np.floor(dat['CollectionTime']).astype(int)
    YearlyGenoDat = dat.groupby(['Year', 'Strain3']).agg(Geno_cases=('id', 'nunique')).reset_index()
    YearlyCases = dat.groupby('Year').agg(All_cases=('id', 'nunique'), mean_pop=('PopulationSize', 'mean')).reset_index()
    CasesGeno = pd.merge(YearlyGenoDat, YearlyCases, on='Year')
    CasesGeno['geno_prop'] = CasesGeno['Geno_cases'] / CasesGeno['All_cases']

    # Subset to years 1-9 for now
    initial8 = dat[(dat['CollectionTime'] < 9) & (dat['CollectionTime'] > 1)]

    if verbose: print(initial8['Strain'].value_counts())
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
    initial8['AgeCat'] = np.nan
    initial8.loc[initial8['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
    initial8.loc[initial8['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
    initial8.loc[initial8['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
    initial8.loc[initial8['Age'] == '60+', 'AgeCat'] = '>=5 y'

    # Now, getting cases by age bin
    cases_summary = initial8.groupby(['AgeCat', 'CollectionTime']).agg(Cases_age=('id', 'nunique')).reset_index()
    if verbose: print(cases_summary.head())

    # Now, getting total population by time point
    # It's already pooled, so just need one observation per time point and then multiply by age bin
    # Doing one per age bin so that the frame is set up to easily multiply
    pop_pooled = initial8.groupby(['CollectionTime', 'AgeCat']).apply(lambda x: x.head(1)).reset_index(drop=True)
    pop_pooled2 = pop_pooled[['CollectionTime', 'AgeCat', 'PopulationSize']]

    # Multiplying by fraction of the population in each age bin
    pop_pooled2['Pop_Age'] = np.nan
    pop_pooled2.loc[pop_pooled2['AgeCat'] == '<1 y', 'Pop_Age'] = pop_pooled2['PopulationSize'] * 0.025
    pop_pooled2.loc[pop_pooled2['AgeCat'] == '1-2 y', 'Pop_Age'] = pop_pooled2['PopulationSize'] * 0.025
    pop_pooled2.loc[pop_pooled2['AgeCat'] == '2-5 y', 'Pop_Age'] = pop_pooled2['PopulationSize'] * 0.075
    pop_pooled2.loc[pop_pooled2['AgeCat'] == '>=5 y', 'Pop_Age'] = pop_pooled2['PopulationSize'] * 0.875

    # Merge incidence with the age and time appropriate denominator to get the ratio
    AgeIncidence = pd.merge(cases_summary, pop_pooled2, on=['AgeCat', 'CollectionTime'])
    AgeIncidence['IR_100k'] = (AgeIncidence['Cases_age'] / AgeIncidence['Pop_Age']) * 100000 / FUDGE

    # Get an average to calibrate to average pre-vaccine period
    Inci_dist = AgeIncidence.groupby('AgeCat').agg(meanIR=('IR_100k', 'mean')).reset_index()

    # Standardize data format
    age_mapping = {
        '<1 y': 0,
        '1-2 y': 1,
        '2-5 y': 2,
        '>=5 y': 5,
    }
    ages = Inci_dist['AgeCat'].replace(age_mapping)
    df = sc.dataframe(dict(ages=ages, inci=Inci_dist['meanIR']))
    df = df.sort_values(by='ages').reset_index(drop=True)
    return df

if __name__ == '__main__':

    m_inci = process_model()
    d_inci = process_data()