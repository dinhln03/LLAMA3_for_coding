import csv
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
survey_path = os.path.join(dir_path, '../data/test_two_entries.csv')

NUM_QUESTIONS = 8
RESPONSE_PERSON = ['pat', 'jeremy', 'zach']
TASTE_PROFILE_TYPES = ['deliciousness', 'heaviness', 'reliability', 'frequency', 'between']

i = 0
person_responses = []

with open(survey_path) as f:
    data = csv.reader(f, delimiter=',', quotechar='|')
    for row in data:
        if i == 1:
            sando_type_row = row
        if i > 1:
            person_responses.append(row)
        i += 1

num_sando_types = int(
    (len(sando_type_row) - 3)
    / NUM_QUESTIONS
)

end_index = 2 + num_sando_types
sando_types = sando_type_row[2:end_index]

global_taste_profile = {}

j = 0
for response in person_responses:
    taste_profile = {}
    name = RESPONSE_PERSON[j]

    ## Loop through deliciousness, heaviness, etc.
    ## Pull out deliciousness, etc. scores and store in taste_profile[type]
    for data_type in TASTE_PROFILE_TYPES:
        start_index = 2 + (1 + TASTE_PROFILE_TYPES.index(data_type)) * num_sando_types
        end_index = start_index + num_sando_types
        raw_profile = np.array(response[start_index:end_index])
        if data_type in ['deliciousness', 'heaviness', 'reliability']:
            float_profile = raw_profile.astype(np.float) * 0.01
            taste_profile[data_type] = float_profile
        else:
            int_profile = raw_profile.astype(np.int)
            taste_profile[data_type] = int_profile

    profile_csv_path = os.path.join(dir_path, '../data/users/profiles', (name + '.csv'))

    with open(profile_csv_path, 'w') as f:
        profile_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['sando_type']
        for data_type in TASTE_PROFILE_TYPES:
            header.append(data_type)
        profile_writer.writerow(header)

        ## Loop through sando types and dump to CSV
        for sando in sando_types:
            sando_index = sando_types.index(sando)
            sando_row = [sando]
            for data_type in TASTE_PROFILE_TYPES:
                sando_row.append(taste_profile[data_type][sando_index])
            profile_writer.writerow(sando_row)
