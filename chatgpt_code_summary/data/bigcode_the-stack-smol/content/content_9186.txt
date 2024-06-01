import glob
print(glob.glob("./src/ibmaemagic/sdk/*"))

# import sys
# sys.path.append("./src/ibmaemagic/magic/")
# from analytic_magic_client import AnalyticMagicClient
# import analytic_engine_client.AnalyticEngineClient

# from ibmaemagic.magic.analytic_magic_client import AnalyticMagicClient
# from ibmaemagic.sdk.analytic_engine_client import AnalyticEngineClient
# from ibmaemagic import AnalyticEngineClient


import sys
sys.path.append("./src/ibmaemagic/sdk/")
from analytic_engine_client import AnalyticEngineClient