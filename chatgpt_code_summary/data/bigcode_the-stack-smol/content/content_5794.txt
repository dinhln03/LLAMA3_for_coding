import pandas as pd
import pprint


all_client_diagnoses = pd.read_csv('2021_encounters_with_diagnoses.csv')
print(all_client_diagnoses.columns)
nora_clients = all_client_diagnoses.drop_duplicates('Pid').drop(columns=['Date Of Service', 'Encounter', 'Age', 'Service Code'])

nora_gender = nora_clients[nora_clients.Facility == 'Northern Ohio Recovery Association'].groupby('Gender').count()

lorain_gender = nora_clients[nora_clients.Facility == 'Lorain'].groupby('Gender').count()
print('------------------------------------')
print('NORA All Client Gender Breakdown')
print('-------------------------------------')
pprint.pprint(nora_gender)
print('------------------------------------')
print('Lorain All Client Gender Breakdown')
print('-------------------------------------')
pprint.pprint(lorain_gender)
print('------------------------------------')