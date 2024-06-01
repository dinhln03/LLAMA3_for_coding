from __future__ import absolute_import

#SymPy is a non-commercial alternative to Mathematica and Maple
# SymPy can map variable to a value or a matrix.
# SymPy's Symbolic Statistical Modelling uses scintific computing.
import sys
import numpy as np
import sympy as sp
import pandas as pd
from pathlib import Path
from .tokens import *
from .equation import *

class Equations(Equation):
    def __init__(self):
        path = Path(__file__).parent
        self.filepath = path.joinpath("fixtures","equations.xlsx")
        self.equations_sheet = "equations"
        self.column_mapping_sheet = "col_var_mapping"     
        self.data_sheet = "values"  
        self.mappings = None
        self.df = None
        self.equations_df = pd.DataFrame()
        self.equations = dict()
        self.lhs = None
        self.values =  dict()
    
    def upload_data_equations(self, filepath, equations_sheet, data_sheet, column_mapping_sheet=""):
        if not self.validate_file_inputs(filepath, equations_sheet, data_sheet):
            return False
        self.filepath = filepath
        self.equations_df = pd.read_excel(self.filepath, sheet_name=equations_sheet, mangle_dupe_cols=True)
        self.df = pd.read_excel(self.filepath, sheet_name=data_sheet, mangle_dupe_cols=True)
        if column_mapping_sheet:
            self.mappings = pd.read_excel(self.filepath, sheet_name=column_mapping_sheet, mangle_dupe_cols=True)

    def validate_file_inputs(self, filepath, equations_sheet, data_sheet):
        if not filepath or not equations_sheet or not data_sheet:
            raise Exception("Empty upload data inputs. Please provide valid inputs to file upload.")
        else:
            return True
        return False

    def process_equations(self):
        self.lhs = self.equations_df['name']
        eq_list =  self.equations_df['equation']
        self.equations = dict()
        for variable, equation in zip(self.lhs, eq_list):
            self.equations[variable] = Equation(equation, self.df)
            self.equations[variable].set_symbols(self.mappings)
            self.values[variable] = self.equations[variable].evaluate(self.values)
        result_df = pd.DataFrame.from_dict(self.values)
        result_df.to_csv("results.csv", index=False)
        return self.values
