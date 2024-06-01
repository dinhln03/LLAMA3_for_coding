
import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-sep-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_blending_1_my_rank_004-2o-lightgbm-colsample_81830_my_ranking_81790_0926_1918.csv"
SUBMISSION_MESSAGE = '"004-2o-lightgbm-colsample-tps-sep-2021 +  stacking_lgb_xbg_cat_imputer_no_imputer"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
