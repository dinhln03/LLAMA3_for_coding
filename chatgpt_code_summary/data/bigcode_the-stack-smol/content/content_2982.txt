import pandas as pd
import plotly.express as px

df = pd.read_csv('data/query_result.csv')
max_df = df.groupby(by='topic_id').max().reset_index()
df = df[df['topic_id'].isin(max_df[max_df['recall'] > 0]['topic_id'].to_list())]
for t in df['topic_id'].unique().tolist():
    temp_df = df[df['topic_id'] == t]
    fig = px.box(df, x="topic_id", y="recall")
    fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
    fig.show()
