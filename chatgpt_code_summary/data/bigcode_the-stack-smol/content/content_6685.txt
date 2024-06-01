
# coding: utf-8

# # Load and preprocess 2012 data
# 
# We will, over time, look over other years. Our current goal is to explore the features of a single year.
# 
# ---

# In[1]:

get_ipython().magic('pylab --no-import-all inline')
import pandas as pd


# ## Load the data.
# 
# ---
# 
# If this fails, be sure that you've saved your own data in the prescribed location, then retry.

# In[2]:

file = "../data/interim/2012data.dta"
df_rawest = pd.read_stata(file)


# In[7]:

df_rawest.weight_full.isnull()


# In[8]:

good_columns = [#'campfin_limcorp', # "Should gov be able to limit corporate contributions"
    'pid_x',  # Your own party identification
    
    'abortpre_4point',  # Abortion
    'trad_adjust',  # Moral Relativism
    'trad_lifestyle',  # "Newer" lifetyles
    'trad_tolerant',  # Moral tolerance
    'trad_famval',  # Traditional Families
    'gayrt_discstd_x',  # Gay Job Discrimination
    'gayrt_milstd_x',  # Gay Military Service
    
    'inspre_self',  # National health insurance
    'guarpr_self',  # Guaranteed Job
    'spsrvpr_ssself',  # Services/Spending
    
    'aa_work_x',  # Affirmative Action  ( Should this be aapost_hire_x? )
    'resent_workway', 
    'resent_slavery', 
    'resent_deserve',
    'resent_try',
]

df_raw = df_rawest[good_columns]


# ## Clean the data
# ---

# In[9]:

def convert_to_int(s):
    """Turn ANES data entry into an integer.
    
    >>> convert_to_int("1. Govt should provide many fewer services")
    1
    >>> convert_to_int("2")
    2
    """
    try:
        return int(s.partition('.')[0])
    except ValueError:
        warnings.warn("Couldn't convert: "+s)
        return np.nan
    except AttributeError:
        return s

def negative_to_nan(value):
    """Convert negative values to missing.
    
    ANES codes various non-answers as negative numbers.
    For instance, if a question does not pertain to the 
    respondent.
    """
    return value if value >= 0 else np.nan

def lib1_cons2_neutral3(x):
    """Rearrange questions where 3 is neutral."""
    return -3 + x if x != 1 else x

def liblow_conshigh(x):
    """Reorder questions where the liberal response is low."""
    return -x

def dem_edu_special_treatment(x):
    """Eliminate negative numbers and {95. Other}"""
    return np.nan if x == 95 or x <0 else x

df = df_raw.applymap(convert_to_int)
df = df.applymap(negative_to_nan)

df.abortpre_4point = df.abortpre_4point.apply(lambda x: np.nan if x not in {1, 2, 3, 4} else -x)

df.loc[:, 'trad_lifestyle'] = df.trad_lifestyle.apply(lambda x: -x)  # 1: moral relativism, 5: no relativism
df.loc[:, 'trad_famval'] = df.trad_famval.apply(lambda x: -x)  # Tolerance. 1: tolerance, 7: not

df.loc[:, 'spsrvpr_ssself'] = df.spsrvpr_ssself.apply(lambda x: -x)

df.loc[:, 'resent_workway'] = df.resent_workway.apply(lambda x: -x)
df.loc[:, 'resent_try'] = df.resent_try.apply(lambda x: -x)


df.rename(inplace=True, columns=dict(zip(
    good_columns,
    ["PartyID",
    
    "Abortion",
    "MoralRelativism",
    "NewerLifestyles",
    "MoralTolerance",
    "TraditionalFamilies",
    "GayJobDiscrimination",
    "GayMilitaryService",

    "NationalHealthInsurance",
    "StandardOfLiving",
    "ServicesVsSpending",

    "AffirmativeAction",
    "RacialWorkWayUp",
    "RacialGenerational",
    "RacialDeserve",
    "RacialTryHarder",
    ]
)))


# In[10]:

print("Variables now available: df")


# In[11]:

df_rawest.pid_x.value_counts()


# In[12]:

df.PartyID.value_counts()


# In[13]:

df.describe()


# In[14]:

df.head()


# In[21]:

df.to_csv("../data/processed/2012.csv")


# In[15]:

df_rawest.weight_full.to_csv("../data/processed/2012_weights.csv")


# In[16]:

df_rawest.shapee


# In[ ]:



