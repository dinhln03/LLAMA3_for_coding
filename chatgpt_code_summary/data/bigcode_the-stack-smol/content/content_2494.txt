# coding: utf-8
__author__ = "Jerry He"
import dash_bootstrap_components as dbc
from dash import dcc, no_update
from dash_extensions.enrich import Dash, Output, Input, State, html
import flask
from flask import jsonify
from flask_cors import CORS
from dash import dash_table
import dash_ace

server = flask.Flask(__name__)
CORS(server)
from dash_extensions.enrich import DashProxy,ServersideOutput, TriggerTransform, MultiplexerTransform, ServersideOutputTransform, NoOutputTransform

app = DashProxy(__name__, 
    server=server,
    transforms=[
        ServersideOutputTransform(),  # enable use of ServersideOutput objects
    ],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

server = app.server

import pandas as pd
def row_tf(row):
    keep = ['title', 'userid']
    newrow  = {k:row[k] for k in keep}
    newrow['name'] = newrow['title'].split("-")[0].strip()
    return newrow

def df_transform(df):
    return pd.DataFrame([row_tf(row) for _,row in df.iterrows()])


app.layout = html.Div(
    [
      dcc.Store(id="querystr"),
      dcc.Store(id="store"),
      dcc.Store(id="all-df"),
      dcc.Interval(interval=1800, id="query_sto"),
    dbc.Card([
        dbc.CardImg(src="assets/brick_header.jpg"),
        dbc.CardBody([
        dbc.Tabs(
            [
                dbc.Tab([
    html.Hr(),
    dash_ace.DashAceEditor(
            id='query-input',
            value=r"SELECT * FROM my_music_collection WHERE artist like '%Jr%' LIMIT 8",
            theme='github',
            mode='sql',
            tabSize=2,
            height="35px",
            enableBasicAutocompletion=True,
            enableLiveAutocompletion=True,
            autocompleter='/autocompleter?prefix=',
            placeholder='SQL code ...'
        ),
    dbc.Button("Query", color="secondary", className="me-1",
                 id='query-button'),
    html.Hr(),
    html.Div(id="query-output")

    ],label="SQL", tab_id="tab-1"),
                dbc.Tab(label="History", tab_id="tab-2"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
        html.Div(id="tab-content"),
    ])

    ])
    ]
)

import json
app.clientside_callback("""
function(n_intervals, data) {
       var existing_data;
       if(data) {
          existing_data = JSON.parse(data)
        }
       var editor = ace.edit("query-input")
       if(!existing_data || existing_data['querystr'] != editor.getValue().trim()) {
               return JSON.stringify({
                   'querystr':editor.getValue().trim(), 
                    'time':(new Date()).toISOString()
               })
       }
}
""".strip(),
Output("querystr", "data"), Input("query_sto",'n_intervals'), State("querystr", "data"))

from sqlalchemy import create_engine
engine = create_engine('postgresql://localhost:5432/jerry') # change this to your SQL endpoint/auth

import logging
import dateutil.parser
@app.callback(ServersideOutput("store", "data"), Input('query-button', 'n_clicks'),State("querystr", "data"), memoize=True)
def query(n_clicks, query_data):
    if query_data is None:
        return no_update
    qdata = json.loads(query_data)
    try:
        dat = pd.read_sql(qdata["querystr"].replace("%", "%%"), con=engine)
        return dat
    except:
        logging.exception("SQL query failed\n")

from datetime import datetime

@app.callback(Output("query-output", "children"), ServersideOutput("all-df", "data"), Input("store", "data"), State("all-df", "data"))
def render_query_res_table(data, all_df):
    df = df_transform(data)
    df = df[sorted(df.columns.tolist())]
    if all_df is None:
        all_df = [{'df':df, 'time':datetime.now()}]
    else:
        all_df.append({'df':df, 'time':datetime.now()})
    return [dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
         style_header={
        'backgroundColor': 'grey',
        'fontWeight': 'bold'
         },
    )],all_df


@app.callback(Output("tab-content", "children"), [Input("tabs", "active_tab"), State("all-df", "data")])
def switch_tab(at, all_df):
    if at == "tab-1":
        return []
    elif at == "tab-2":
        return dbc.Accordion(
        [
           dbc.AccordionItem([
                dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in query_hist['df'].columns],
                        data=query_hist['df'].to_dict('records'),
                         style_header={
                        'backgroundColor': 'grey',
                        'fontWeight': 'bold'
                                    },
                )
            ], title = query_hist['time'].strftime("%H:%M:%S")) for query_hist in all_df
        ])
    return html.P("This shouldn't ever be displayed...")

@server.route('/autocompleter', methods=['GET'])
def autocompleter():
    return jsonify([{"name": "Completed", "value": "Completed", "score": 100, "meta": "test"}])

app.run_server(host="127.0.0.1", debug=True, port=8080)