# Load both the 2016 and 2017 sheets by name
all_survey_data = pd.read_excel("fcc_survey.xlsx", sheet_name = ['2016', '2017'])

# View the data type of all_survey_data
print(type(all_survey_data))

'''
<script.py> output:
    <class 'collections.OrderedDict'>
'''


# Load all sheets in the Excel file
all_survey_data = pd.read_excel("fcc_survey.xlsx", sheet_name = [0, '2017'])

# View the sheet names in all_survey_data
print(all_survey_data.keys())

'''
<script.py> output:
    odict_keys([0, '2017'])
'''


# Load all sheets in the Excel file
all_survey_data = pd.read_excel("fcc_survey.xlsx",
                                sheet_name = None)

# View the sheet names in all_survey_data
print(all_survey_data.keys())

'''
<script.py> output:
    odict_keys(['2016', '2017'])
'''


# Notice that if you load a sheet by its index position, the resulting data frame's name is also the index number, not the sheet name.