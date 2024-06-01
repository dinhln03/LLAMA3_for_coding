from fastapi import APIRouter, HTTPException
import pandas as pd
import plotly.express as px
import json
from dotenv import load_dotenv
import os
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.types import Integer, Float, Text, String, DateTime
from fastapi.encoders import jsonable_encoder
from os.path import join as join_path

router = APIRouter()

@router.post('/wage_trade_transport_viz/')
async def wage_trade_transport_viz(user_queried_citystates: list):
    """
    ### Path Parameter (POST from front-end)
    list: A list of city-states the user queried in this format: ["Albany, NY", "San Francisco, CA", "Chicago, IL"]

    ### Response
    JSON string of all figures to render with [react-plotly.js](https://plotly.com/javascript/react/)
    """
    def create_db_uri():

        # give full path to .env
        env_path = r'.env'
        # LOAD environment variables
        load_dotenv(dotenv_path=env_path, verbose=True)
        # GET .env vars
        DB_FLAVOR = os.getenv("DB_FLAVOR")
        DB_PYTHON_LIBRARY = os.getenv("DB_PYTHON_LIBRARY")
        DB_HOST = os.getenv("DB_HOST")
        DB_NAME = os.getenv("DB_NAME")
        DB_USER = os.getenv("DB_USER")
        DB_PASS = os.getenv("DB_PASS")
        DB_PORT = os.getenv("DB_PORT")
        DB_URI = DB_FLAVOR + "+" + DB_PYTHON_LIBRARY + "://" + DB_USER + ":" + DB_PASS + "@" + DB_HOST + ":" + DB_PORT + "/" + DB_NAME
        return DB_URI
    
    DB_URI = create_db_uri()

    # CONNECTION Engine with SQLAlchemy
    engine = create_engine(DB_URI, echo=True)

    def cc_json():
        '''
        Opens county_city.json file, converts to .json object and returns it
        '''
        with open(join_path('app', 'db', 'city-county.json')) as f:
            data_to_encode = json.load(f)

        encoded_json = jsonable_encoder(data_to_encode)

        county_city_json = json.dumps(encoded_json)

        return county_city_json
    cc = cc_json()
    cc = json.loads(cc)
    # city_states_list = ["New York City, NY", "San Francisco, CA", "Chicago, IL"]
    def get_county_from_city(city_states_list):
        county_list = []
        i = 0
        for i in range(len(city_states_list)):
            county_list.append(cc[city_states_list[i]])
            i += 1
        return county_list
    county_list = get_county_from_city(user_queried_citystates)

    def sql_query(county_list):
        '''
        Create a SQL query to grab only the user queried cities' data from the covid table in the DB.
        Output: subset grouped DF by month and city with only queried cities
        '''
        
        # get length of list of queried cities
        list_length = len(county_list)

        # Create Boolean Statements to Avoid Errors with output
        if list_length == 1:
            county1 = county_list[0]
            query1 = 'SELECT * FROM jobs WHERE county_state IN (%(county1)s)'
            subsetJ = pd.read_sql(sql = query1, columns = "county_state", params={"county1":county1}, con=engine, parse_dates=['created_at', 'updated_at'])
        elif list_length == 2:
            county1 = county_list[0]
            county2 = county_list[1]
            query2 = 'SELECT * FROM jobs WHERE county_state IN (%(county1)s, %(county2)s)'
            subsetJ = pd.read_sql(sql = query2, columns = "county_state", params={"county1":county1, "county2":county2}, con=engine, parse_dates=['created_at', 'updated_at'])
        elif list_length == 3:
            county1 = county_list[0]
            county2 = county_list[1]
            county3 = county_list[2]
            query3 = 'SELECT * FROM jobs WHERE "county_state" IN (%(county1)s, %(county2)s, %(county3)s)'
            subsetJ = pd.read_sql(sql = query3, columns = "county_state", params={"county1":county1, "county2":county2, "county3":county3}, con=engine, parse_dates=['created_at', 'updated_at'])
        else:
            raise Exception("Please pass a list of 1-3 City-States")

        return subsetJ
    subsetJ = sql_query(county_list)
    
    industry_list = ['Goods-producing', 'Natural resources and mining', 'Construction', 'Manufacturing', 'Service-providing', 'Trade, transportation, and utilities', 'Information', 'Financial activities', 'Professional and business services', 'Education and health services', 'Leisure and hospitality', 'Other services', 'Unclassified']
    def create_wage_plots(df, industry_list, industry_name):
        subsetJ['County, State'] = subsetJ['county_state']
        subsetJ['date'] = pd.PeriodIndex(year=subsetJ['Year'], quarter=subsetJ['Qtr']).to_timestamp()
        industry = subsetJ[subsetJ['Industry']==industry_name]
        industry = industry.sort_values('date')
        fig = px.line(industry, x='date', y='Average Weekly Wage', labels={'Average Weekly Wage': 'Average Weekly Wage ($)', 'date': 'Date'}, color='County, State', title=f"{industry_name}: Average Weekly Wage").for_each_trace(lambda t: t.update(name=t.name.split("=")[-1]))
        fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                    xaxis = dict(
                      tickmode = 'array',
                      tick0 = 1,
                      dtick = 1,
                      tickvals = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
                      ticktext = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
                  ))   # legend above graph top right
        fig.write_image("fig1.png")
        jobs_json = fig.to_json()    # save figure to JSON object to pass to WEB
        return jobs_json
    wage_json = create_wage_plots(subsetJ, industry_list, industry_list[5])

    return wage_json
