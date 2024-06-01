import sys
import re
import pandas as pd

network_filename = sys.argv[1]
m = re.match("networks/(?P<dataset>.*?)_similarity", network_filename)
dataset = m.groupdict()['dataset']


G=nx.read_gml(network_filename)
labels=pd.read_csv(f"munged_data/{dataset}/labels.csv", index_col=0)
metadata = pd.read_csv(f"data/intermediate/{dataset}/metadata.csv", index_col=0)
features = pd.read_csv(f"data/intermediate/{dataset}/features.csv", index_col=0)

train = pd.read_csv(f"data/intermediate/{dataset}/train.csv", header = None)[0].values
testing = pd.Series({i:(i in test) for i in labels.index})
labels = labels.mask(testing, other=0)

propagator,nodes=make_propagator(G)
df,df_time=propagate(propagator, nodes, moas)
df.to_csv(f"predictions/{dataset}/predicted_by_propagation.csv")

