"""
This file is part of the FJournal Project.
Copyright © 2019-2020, Daniele Penazzo. All Rights Reserved.
The use of this code is governed by the MIT license attached.
See the LICENSE file for the full license.

Created on: 2020-07-10

Author: Penaz
"""
from tkinter import ttk
import tkinter as tk
from models import Meal


class AddMealPopup(ttk.Frame):
    """
    Defines a popup for adding meals
    """

    def __init__(self, master=None, session=None):
        """
        Constructor of the class
        """
        super().__init__(master)
        self.master = master
        self.grid(row=0, column=0)
        self.session = session
        self.mealname = tk.StringVar()
        self.create_widgets()

    def create_widgets(self):
        """
        Creates the widgets for the popup
        """
        self.meallbl = ttk.Label(self, text="Meal Name")
        self.meallbl.grid(row=0, column=0)
        self.mealinput = ttk.Entry(self, textvariable=self.mealname)
        self.mealinput.grid(row=0, column=1)
        self.addbtn = ttk.Button(self,
                                 text="Confirm",
                                 command=self.add_meal)
        self.addbtn.grid(row=1, column=0, columnspan=2)

    def add_meal(self):
        """
        Opens the Add Meal popup
        """
        meal = Meal(name=self.mealname.get())
        self.session.add(meal)
        self.session.commit()
        self.master.destroy()
