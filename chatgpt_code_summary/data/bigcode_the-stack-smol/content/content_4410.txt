import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from pymongo import MongoClient
import json
import os
 
client = MongoClient(os.environ.get("DATABASE"))
db = client.politics.brexit


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

data = list(db.find({}, {"_id": 0})) 

data_df = pd.json_normalize(data).set_index('timestamp')
top_columns = data_df.sum().sort_values(ascending=False)
top_10 = top_columns[0:10].index.tolist()
top10df = data_df[top_10].fillna(0).astype(int)
df = top10df[-12:]

cols = list(df.columns)

# Set up plot
fig = go.Figure()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(children=[
    html.H1(
        children='#brexit',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Top Keywords used with the #brexit hashtag in the last 12 hours', style={
        'textAlign': 'center',
        'color': colors['text']
    }),


    dcc.Graph(
        id='test-plot',
        figure={
            'data': [
                go.Scatter(
                    x=df.index,
                    y=df[i],
                    name=i.replace('words.', ''),
                    line=dict(shape='spline', width=2),
                    opacity=0.8
                ) for i in cols[0:10]
            ],
            'layout': go.Layout(
                xaxis={'title': 'Time'},
                yaxis={'title': 'Frequency'},
                margin={'l': 40, 'b': 80, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        },
    ),
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # in milliseconds
        n_intervals=0
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
