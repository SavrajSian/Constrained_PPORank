import pandas as pd
import numpy as np

############################################################################################################
drugs = pd.read_csv('GDSC_ALL_alldrugs/Drugs/Drug_list_265.csv')

nontoxic_ids = 'GDSC_ALL_alldrugs/gdsc_drugMedianGE0.txt'
nontoxic_ids = list(pd.read_csv(nontoxic_ids, header=None)[0].values.astype(int))

nontoxic_drugs = drugs[drugs['drug_id'].isin(nontoxic_ids)]
nontoxic_drugs.to_csv('GDSC_ALL_alldrugs/nontoxic_drug_list.csv', index=False)

############################################################################################################

#get drugs in ADRCS and nontoxic
ADRCS_path = 'ADRCS/'
ADRCS_drugs = pd.read_csv(ADRCS_path + 'Drug_information.csv')

#concat nontoxic_drugs name and synonyms
nontoxic_all_names = nontoxic_drugs.copy()
nontoxic_all_names = nontoxic_all_names.drop(columns=['Targets', 'Target pathway', 'Sample Size', 'Count'])
nontoxic_all_names['Name'] = nontoxic_all_names['Name'].fillna('')
nontoxic_all_names['Synonyms'] = nontoxic_all_names['Synonyms'].fillna('')
nontoxic_all_names.loc[:, 'combined'] = nontoxic_all_names['Name'].str.cat(nontoxic_all_names['Synonyms'], sep=', ')

#concat ADRCS_drugs name and synonyms
ADRCS_all_names = ADRCS_drugs.copy()
ADRCS_all_names['DRUG_NAME'] = ADRCS_all_names['DRUG_NAME'].fillna('')
ADRCS_all_names['DRUG_SYNONYMS'] = ADRCS_all_names['DRUG_SYNONYMS'].fillna('')
ADRCS_all_names.loc[:, 'combined'] = ADRCS_all_names['DRUG_NAME'].str.cat(ADRCS_all_names['DRUG_SYNONYMS'], sep=', ')

#see what drugs are in both

names1 = set(nontoxic_all_names['combined'].str.split(',').sum())
names2 = set(ADRCS_all_names['combined'].str.split(',').sum())

common_names = names1.intersection(names2)

ADRCS_common = ADRCS_all_names[ADRCS_all_names['combined'].str.split(',').apply(lambda x: any([i in common_names for i in x]))]

drugs_in_common = nontoxic_all_names[nontoxic_all_names['combined'].str.split(',').apply(lambda x: any([i in common_names for i in x]))]
drugs_in_common = drugs_in_common.drop_duplicates(subset='combined')

#append DRUG_ID from ADRCS_common to drugs_in_common
name_to_id = ADRCS_all_names.set_index('DRUG_NAME')['DRUG_ID'].to_dict()
drugs_in_common['BADD_ID'] = drugs_in_common['Name'].map(name_to_id)


############################################################################################################

#load drug-adr interaction data
#all_interactions = pd.read_csv('Drug_ADR_interactions.txt', sep='\t')
#interactions_in_common = all_interactions[all_interactions['DRUG_ID'].isin(drugs_in_common['BADD_ID'])]
#quantitative version - more useful

all_interactions_quantitative = pd.read_csv('ADRCS/Drug_ADR_relations_quantification.txt', sep='\t')
interactions_in_common_quantitative = all_interactions_quantitative[all_interactions_quantitative['DRUG_ID'].isin(drugs_in_common['BADD_ID'])]
#get number of interactions for each drug
drug_id_counts = interactions_in_common_quantitative['DRUG_ID'].value_counts()
#print('interactions per drug:', drug_id_counts)
#count number of ADR terms
adr_counts = interactions_in_common_quantitative['ADR_TERM'].value_counts()
#print('times each ADR term appears:', adr_counts)
#unique ADR terms
unique_adrs = interactions_in_common_quantitative['ADR_TERM'].unique()
#print('unique ADR terms:', len(unique_adrs))

#count severity of ADRs
severity_counts = interactions_in_common_quantitative['ADR_Severity_Grade_FAERS'].value_counts()
#print(severity_counts)

#get number of severity grades for each drug
drug_severity_counts = interactions_in_common_quantitative.pivot_table(index='DRUG_ID', columns='ADR_Severity_Grade_FAERS', aggfunc='size', fill_value=0)
#print(drug_severity_counts)

#append onto drugs_in_common
drugs_with_severities = drugs_in_common.copy()
drugs_with_severities = drugs_in_common.merge(drug_severity_counts, left_on='BADD_ID', right_index=True, how='left')
drugs_with_severities = drugs_with_severities.dropna(axis = 0, how='any', subset=['Mild', 'Moderate', 'Severe', 'Lifethreatening', 'Death'])

#save to csv
drugs_with_severities.to_csv('GDSC_ALL/drugs_with_severities.csv', index=False)

#save drug ids as txt
#use this file in the prepare.py file - replace gdsc_drugMedianGE0.txt with this file so only these are included
drug_ids = drugs_with_severities['drug_id'].values
np.savetxt('GDSC_ALL/drugs_with_severities_ids.txt', drug_ids, fmt='%d')


############################################################################################################

# scoring system for severities
# point system for each severity grade
# then sum for each drug
# normalise by number of interactions

severity_multiplier = np.array([1, 2, 3, 5, 8])

severity_scores = drugs_with_severities[['drug_id', 'Mild', 'Moderate', 'Severe', 'Lifethreatening', 'Death']].copy()
severity_scores = severity_scores.replace('None', 0)
severity_scores = severity_scores.astype(int)
severity_scores.iloc[:, 1:] = severity_scores.iloc[:, 1:].multiply(severity_multiplier, axis=1)
# total severity score
severity_scores['total'] = severity_scores.iloc[:, 1:].sum(axis=1)
# normalise by number of interactions
interactions_per_drug = drugs_with_severities[['Mild', 'Moderate', 'Severe', 'Lifethreatening', 'Death']].sum(axis=1)
severity_scores['normalised'] = severity_scores['total'] / interactions_per_drug

severity_scores.to_csv('GDSC_ALL/severity_scores.csv', index=False)



