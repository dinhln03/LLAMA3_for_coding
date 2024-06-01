import sys
import math
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")
from sdc_etl_libs.sdc_dataframe.Dataframe import *
import pandas as pd
import numpy as np
import json
import pytest


def test_generate_insert_query_ddl(mocker):
    test_schema = """
      {
    "namespace": "TimeControl",
    "type": "object",
    "name": "languages",
    "country_code": "USA",
    "data_sink": {"type":"snowflake", "database": "HRIS_DATA", "table_name": "LANGUAGES", "schema": "TIMECONTROL"},
    "data_source": {"type": "api", "base_url": "https://smiledirectclub.timecontrol.net/api/v1"},
    "fields": [
      {"name":"_METADATA","type":{"type":"string","logical_type":"json"}},
      {"name":"KEY","type":{"type":"int"},"sf_merge_key": true},
      {"name":"NAME","type":{"type":"string"}},
      {"name":"DESCRIPTION","type":{"type":"string"}},
      {"name":"CULTURE","type":{"type":"string"}},
      {"name":"_SF_INSERTEDDATETIME","type":{"type":"string","logical_type":"datetime", "add_column": true }}
    ]
  }"""

    test_data = """
    [{"_metadata": {"links": [{"id": "9",
        "rel": "self",
        "href": "/api/v1/languages/9",
        "code": "Ceština"}]},
     "Key": 9,
     "Name": "Ceština",
     "Description": "Czech",
     "Culture": "cs"},
     {"_metadata": {"links": [{"id": "10",
        "rel": "self",
        "href": "/api/v1/languages/10",
        "code": "This"}]},
     "Key": 9,
     "Name": "This",
     "Description": "Is",
     "Culture": "ze"}]
        """

    df = Dataframe(SDCDFTypes.PANDAS, test_schema)
    df.load_data(json.loads(test_data))
    query = df.generate_insert_query_ddl(df.df)

    assert query == '("CULTURE", "DESCRIPTION", "KEY", "NAME", "_METADATA", "_SF_INSERTEDDATETIME") select Column1 as "CULTURE", Column2 as "DESCRIPTION", Column3 as "KEY", Column4 as "NAME", PARSE_JSON(Column5) as "_METADATA", Column6 as "_SF_INSERTEDDATETIME" from values '

def test_generate_insert_query_values(mocker):
    test_schema = """
      {
    "namespace": "TimeControl",
    "type": "object",
    "name": "languages",
    "country_code": "USA",
    "data_sink": {"type":"snowflake", "database": "HRIS_DATA", "table_name": "LANGUAGES", "schema": "TIMECONTROL"},
    "data_source": {"type": "api", "base_url": "https://smiledirectclub.timecontrol.net/api/v1"},
    "fields": [
      {"name":"_METADATA","type":{"type":"string","logical_type":"json"}},
      {"name":"KEY","type":{"type":"int"},"sf_merge_key": true},
      {"name":"NAME","type":{"type":"string"}},
      {"name":"DESCRIPTION","type":{"type":"string"}},
      {"name":"CULTURE","type":{"type":"string"}}
    ]
  }"""

    test_data = """
    [{"_metadata": {"links": [{"id": "9",
        "rel": "self",
        "href": "/api/v1/languages/9",
        "code": "Ceština"}]},
     "Key": 9,
     "Name": "Ceština",
     "Description": "Czech",
     "Culture": "cs"},
     {"_metadata": {"links": [{"id": "10",
        "rel": "self",
        "href": "/api/v1/languages/10",
        "code": "This"}]},
     "Key": 9,
     "Name": "This",
     "Description": "Is",
     "Culture": "ze"}]
        """

    df = Dataframe(SDCDFTypes.PANDAS, test_schema)
    df.load_data(json.loads(test_data))
    query = df.generate_insert_query_values(df.df)

    assert query == "('cs', 'Czech', '9', 'Ceština', '{'links': [{'id': '9', 'rel': 'self', 'href': '/api/v1/languages/9', 'code': 'Ceština'}]}'), ('ze', 'Is', '9', 'This', '{'links': [{'id': '10', 'rel': 'self', 'href': '/api/v1/languages/10', 'code': 'This'}]}'), "

def test_convert_columns_to_json(mocker):
    test_schema = """
        {
      "namespace": "TimeControl",
      "type": "object",
      "name": "languages",
      "country_code": "USA",
      "data_sink": {"type":"snowflake", "database": "HRIS_DATA", 
      "table_name": "LANGUAGES", "schema": "TIMECONTROL"},
      "data_source": {"type": "api", "base_url": 
      "https://smiledirectclub.timecontrol.net/api/v1"},
      "fields": [
        {"name":"_METADATA","type":{"type":"string","logical_type":"json"}},
        {"name":"KEY","type":{"type":"int"},"sf_merge_key": true},
        {"name":"NAME","type":{"type":"string"}},
        {"name":"DESCRIPTION","type":{"type":"string"}},
        {"name":"CULTURE","type":{"type":"string"}}
      ]
    }"""

    test_data = """
      [{"_metadata": {"links": [{"id": "9",
          "rel": "self",
          "href": "/api/v1/languages/9",
          "code": "Ceština"}]},
       "Key": 9,
       "Name": "Ceština",
       "Description": "Czech",
       "Culture": "cs"},
       {"_metadata": {"links": [{"id": "10",
          "rel": "self",
          "href": "/api/v1/languages/10",
          "code": "This"}]},
       "Key": 9,
       "Name": "This",
       "Description": "Is",
       "Culture": "ze"}]
          """

    df = Dataframe(SDCDFTypes.PANDAS, test_schema)
    df.load_data(json.loads(test_data))

    data_before = df.df["_METADATA"][0]

    df.convert_columns_to_json()

    data_after = df.df["_METADATA"][0]

    pytest.assume(data_before == "{'links': [{'id': '9', 'rel': 'self', 'href': '/api/v1/languages/9', 'code': 'Ceština'}]}")
    pytest.assume(data_after == '{"links": [{"id": "9", "rel": "self", "href": "/api/v1/languages/9", "code": "Ce\\u0161tina"}]}')