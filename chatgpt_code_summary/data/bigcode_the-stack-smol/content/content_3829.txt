####################################################################################################
"""
adres_dataset.py

This module implements several classes to perform dataset-specific downloading, saving and
data-transformation operations.

Written by Swaan Dekkers & Thomas Jongstra
"""
####################################################################################################

#############
## Imports ##
#############

from pathlib import Path
import pandas.io.sql as sqlio
import pandas as pd
import numpy as np
import requests
import psycopg2
import time
import os
import re

# Import own modules.
import datasets, clean

# Define HOME and DATA_PATH on a global level.
HOME = Path.home()  # Home path for old VAO.
# USERNAME = os.path.basename(HOME)
# HOME = os.path.join('/data', USERNAME)  # Set home for new VAO.
DATA_PATH = os.path.join(HOME, 'Documents/woonfraude/data/')


########################
## AdresDataset class ##
########################

class AdresDataset(datasets.MyDataset):
    """Create a dataset for the adres data."""

    # Set the class attributes.
    name = 'adres'
    table_name = 'import_adres'
    id_column = 'adres_id'


    def extract_leegstand(self):
        """Create a column indicating leegstand (no inhabitants on the address)."""
        self.data['leegstand'] = ~self.data.inwnrs.notnull()
        self.version += '_leegstand'
        self.save()


    def enrich_with_woning_id(self):
        """Add woning ids to the adres dataframe."""
        adres_periodes = datasets.download_dataset('bwv_adres_periodes', 'bwv_adres_periodes')
        self.data = self.data.merge(adres_periodes[['ads_id', 'wng_id']], how='left', left_on='adres_id', right_on='ads_id')
        self.version += '_woningId'
        self.save()


    def prepare_bag(self, bag):
        # To int
        bag['huisnummer_nummeraanduiding'] = bag['huisnummer_nummeraanduiding'].astype(int)
        bag['huisnummer_nummeraanduiding'] = bag['huisnummer_nummeraanduiding'].replace(0, -1)

        # Fillna and replace ''
        bag['huisletter_nummeraanduiding'] = bag['huisletter_nummeraanduiding'].replace('', 'None')

        # bag['_openbare_ruimte_naam@bag'] = bag['_openbare_ruimte_naam@bag'].fillna('None')
        bag['_openbare_ruimte_naam_nummeraanduiding'] = bag['_openbare_ruimte_naam_nummeraanduiding'].replace('', 'None')

        # bag['_huisnummer_toevoeging@bag'] = bag['_huisnummer_toevoeging@bag'].fillna('None')
        bag['huisnummer_toevoeging_nummeraanduiding'] = bag['huisnummer_toevoeging_nummeraanduiding'].replace('', 'None')
        return bag


    def prepare_adres(self, adres):
        # To int
        adres['hsnr'] = adres['hsnr'].astype(int)
        adres['hsnr'] = adres['hsnr'].replace(0, -1)

        return adres


    def replace_string_nan_adres(self, adres):
        adres['hsnr'] = adres['hsnr'].replace(-1, np.nan)
        adres['sttnaam'] = adres['sttnaam'].replace('None', np.nan)
        adres['hsltr'] = adres['hsltr'].replace('None', np.nan)
        adres['toev'] = adres['toev'].replace('None', np.nan)
        adres['huisnummer_nummeraanduiding'] = adres['huisnummer_nummeraanduiding'].replace(-1, np.nan)
        adres['huisletter_nummeraanduiding'] = adres['huisletter_nummeraanduiding'].replace('None', np.nan)
        adres['_openbare_ruimte_naam_nummeraanduiding'] = adres['_openbare_ruimte_naam_nummeraanduiding'].replace('None', np.nan)
        adres['huisnummer_toevoeging_nummeraanduiding'] = adres['huisnummer_toevoeging_nummeraanduiding'].replace('None', np.nan)
        return adres


    def match_bwv_bag(self, adres, bag):
        # Merge dataframes on adres dataframe.
        new_df = pd.merge(adres, bag,  how='left', left_on=['sttnaam','hsnr'], right_on = ['_openbare_ruimte_naam_nummeraanduiding', 'huisnummer_nummeraanduiding'])

        # Find id's that have a direct match and that have multiple matches.
        g = new_df.groupby('adres_id')
        df_direct = g.filter(lambda x: len(x) == 1)
        df_multiple = g.filter(lambda x: len(x) > 1)

        # Make multiplematch more specific to construct perfect match.
        df_multiple = df_multiple[(df_multiple['hsltr'] == df_multiple['huisletter_nummeraanduiding']) & (df_multiple['toev'] == df_multiple['huisnummer_toevoeging_nummeraanduiding'])]

        # Concat df_direct and df_multiple.
        df_result = pd.concat([df_direct, df_multiple])

        # Because of the seperation of an object, there are two matching objects. Keep the oldest object with definif point.
        df_result = df_result.sort_values(['adres_id', 'status_coordinaat_code'])
        df_result = df_result.drop_duplicates(subset='adres_id', keep='first')

        # Add adresses without match.
        final_df = pd.merge(adres, df_result,  how='left', on='adres_id', suffixes=('', '_y'))
        final_df.drop(list(final_df.filter(regex='_y$')), axis=1, inplace=True)

        # Set the name of the final adres dataframe again.
        final_df.name = 'adres'

        return final_df


    def impute_values_for_bagless_addresses(self, adres):
        """Impute values for adresses where no BAG-match could be found."""
        clean.impute_missing_values(adres)
        # clean.impute_missing_values_mode(adres, ['status_coordinaat_code@bag'])
        adres.fillna(value={'huisnummer_nummeraanduiding': 0,
                            'huisletter_nummeraanduiding': 'None',
                            '_openbare_ruimte_naam_nummeraanduiding': 'None',
                            'huisnummer_toevoeging_nummeraanduiding': 'None',
                            'type_woonobject_omschrijving': 'None',
                            'eigendomsverhouding_id': 'None',
                            'financieringswijze_id': -1,
                            'gebruik_id': -1,
                            'reden_opvoer_id': -1,
                            'status_id_verblijfsobject': -1,
                            'toegang_id': 'None'}, inplace=True)
        return adres


    def enrich_with_bag(self, bag):
        """Enrich the adres data with information from the BAG data. Uses the bag dataframe as input."""
        bag = self.prepare_bag(bag)
        self.data = self.prepare_adres(self.data)
        self.data = self.match_bwv_bag(self.data, bag)
        self.data = self.replace_string_nan_adres(self.data)
        self.data = self.impute_values_for_bagless_addresses(self.data)
        self.version += '_bag'
        self.save()
        print("The adres dataset is now enriched with BAG data.")


    def enrich_with_personen_features(self, personen):
        """Add aggregated features relating to persons to the address dataframe. Uses the personen dataframe as input."""

        # Create simple handle to the adres data.
        adres = self.data

        # Compute age of people in years (float)
        today = pd.to_datetime('today')
        # Set all dates within range allowed by Pandas (584 years?)
        personen['geboortedatum'] = pd.to_datetime(personen['geboortedatum'], errors='coerce')


        # Get the most frequent birthdate (mode).
        geboortedatum_mode = personen['geboortedatum'].mode()[0]
        # Compute the age (result is a TimeDelta).
        personen['leeftijd'] = today - personen['geboortedatum']
        # Convert the age to an approximation in years ("smearin out" the leap years).
        personen['leeftijd'] = personen['leeftijd'].apply(lambda x: x.days / 365.25)

        # Find the matching address ids between the adres df and the personen df.
        adres_ids = adres.adres_id
        personen_adres_ids = personen.ads_id_wa
        intersect = set(adres_ids).intersection(set(personen_adres_ids))

        # Iterate over all matching address ids and find all people at each address.
        inhabitant_locs = {}
        print("Now looping over all address ids that have a link with one or more inhabitants...")
        for i, adres_id in enumerate(intersect):
            if i % 1000 == 0:
                print(i)
            inhabitant_locs[adres_id] = personen_adres_ids[personen_adres_ids == adres_id]

        # Create a new column in the dataframe showing the amount of people at each address.
        # TODO: this step currently takes a few minutes to complete, should still be optimized.
        adres['aantal_personen'] = 0
        adres['aantal_vertrokken_personen'] = -1
        adres['aantal_overleden_personen'] = -1
        adres['aantal_niet_uitgeschrevenen'] = -1
        adres['leegstand'] = True
        adres['leeftijd_jongste_persoon'] = -1.
        adres['leeftijd_oudste_persoon'] = -1.
        adres['aantal_kinderen'] = 0
        adres['percentage_kinderen'] = -1.
        adres['aantal_mannen'] = 0
        adres['percentage_mannen'] = -1.
        adres['gemiddelde_leeftijd'] = -1.
        adres['stdev_leeftijd'] = -1.
        adres['aantal_achternamen'] = 0
        adres['percentage_achternamen'] = -1.
        for i in range(1,8):
            adres[f'gezinsverhouding_{i}'] = 0
            adres[f'percentage_gezinsverhouding_{i}'] = 0.
        print("Now looping over all rows in the adres dataframe in order to add person information...")
        for i in adres.index:
            if i % 1000 == 0:
                print(i)
            row = adres.iloc[i]
            adres_id = row['adres_id']
            try:
                # Get the inhabitants for the current address.
                inhab_locs = inhabitant_locs[adres_id].keys()
                inhab = personen.loc[inhab_locs]

                # Check whether any registered inhabitants have left Amsterdam or have passed away.
                aantal_vertrokken_personen = sum(inhab["vertrekdatum_adam"].notnull())
                aantal_overleden_personen = sum(inhab["overlijdensdatum"].notnull())
                aantal_niet_uitgeschrevenen = len(inhab[inhab["vertrekdatum_adam"].notnull() | inhab["overlijdensdatum"].notnull()])
                adres['aantal_vertrokken_personen'] = aantal_vertrokken_personen
                adres['aantal_overleden_personen'] = aantal_overleden_personen
                adres['aantal_niet_uitgeschrevenen'] = aantal_niet_uitgeschrevenen
                # If there are more inhabitants than people that are incorrectly still registered, then there is no 'leegstand'.
                if len(inhab) > aantal_niet_uitgeschrevenen:
                    adres['leegstand'] = False

                # Totaal aantal personen (int).
                aantal_personen = len(inhab)
                adres.at[i, 'aantal_personen'] = aantal_personen

                # Leeftijd jongste persoon (float).
                leeftijd_jongste_persoon = min(inhab['leeftijd'])
                adres.at[i, 'leeftijd_jongste_persoon'] = leeftijd_jongste_persoon

                # Leeftijd oudste persoon (float).
                leeftijd_oudste_persoon = max(inhab['leeftijd'])
                adres.at[i, 'leeftijd_oudste_persoon'] = leeftijd_oudste_persoon

                # Aantal kinderen ingeschreven op adres (int/float).
                aantal_kinderen = sum(inhab['leeftijd'] < 18)
                adres.at[i, 'aantal_kinderen'] = aantal_kinderen
                adres.at[i, 'percentage_kinderen'] = aantal_kinderen / aantal_personen

                # Aantal mannen (int/float).
                aantal_mannen = sum(inhab.geslacht == 'M')
                adres.at[i, 'aantal_mannen'] = aantal_mannen
                adres.at[i, 'percentage_mannen'] = aantal_mannen / aantal_personen

                # Gemiddelde leeftijd (float).
                gemiddelde_leeftijd = inhab.leeftijd.mean()
                adres.at[i, 'gemiddelde_leeftijd'] = gemiddelde_leeftijd

                # Standardeviatie van leeftijd (float). Set to 0 when the sample size is 1.
                stdev_leeftijd = inhab.leeftijd.std()
                adres.at[i, 'stdev_leeftijd'] = stdev_leeftijd if aantal_personen > 1 else 0

                # Aantal verschillende achternamen (int/float).
                aantal_achternamen = inhab.naam.nunique()
                adres.at[i, 'aantal_achternamen'] = aantal_achternamen
                adres.at[i, 'percentage_achternamen'] = aantal_achternamen / aantal_personen

                # Gezinsverhouding (frequency count per klasse) (int/float).
                gezinsverhouding = inhab.gezinsverhouding.value_counts()
                for key in gezinsverhouding.keys():
                    val = gezinsverhouding[key]
                    adres.at[i, f'gezinsverhouding_{key}'] = val
                    adres.at[i, f'percentage_gezinsverhouding_{key}'] = val / aantal_personen

            except (KeyError, ValueError) as e:
                pass

        print("...done!")

        self.data = adres
        self.version += '_personen'
        self.save()
        print("The adres dataset is now enriched with personen data.")


    def add_hotline_features(self, hotline):
        """Add the hotline features to the adres dataframe."""
        # Create a temporary merged df using the adres and hotline dataframes.
        merge = self.data.merge(hotline, on='wng_id', how='left')
        # Create a group for each adres_id
        adres_groups = merge.groupby(by='adres_id')
        # Count the number of hotline meldingen per group/adres_id.
        # 'id' should be the primary key of hotline df, so it is usable for hotline entry counting.
        hotline_counts = adres_groups['id'].agg(['count'])
        # Rename column
        hotline_counts.columns = ['aantal_hotline_meldingen']
        # Enrich the 'adres' dataframe with the computed hotline counts.
        self.data = self.data.merge(hotline_counts, on='adres_id', how='left')
        self.version += '_hotline'
        self.save()
        print("The adres dataset is now enriched with hotline data.")