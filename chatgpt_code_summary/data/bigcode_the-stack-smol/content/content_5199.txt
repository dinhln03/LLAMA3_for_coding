import tkinter as tk
from tkinter import ttk
import json

from dashboard.entities.InputField import InputField
from dashboard.entities.StatusField import StatusField

class Devices(ttk.Frame):
    """
    Devices Frame for Settings
    """
    def __init__(self, parent, settings):
        """
        Constructs a WarningPopUp
        :param parent: Parent Frame
        :param settings:   settings class 
        """
        self.settings = settings
        ttk.Frame.__init__(self, parent, relief="raised", borderwidth=2)

        self.content = ttk.Frame(self, borderwidth=2)
        self.content.pack(expand=True, fill=tk.X, side='top', anchor='n')
        self.devices = []

        label1 = tk.Label(self.content, text="Apparaten", font=("Verdana", 14), relief="groove")
        label1.pack(expand=True, fill=tk.X, side='top')

        self.render_devices()

    def render_devices(self):
        # Removed current sidebar buttons
        for frame in self.devices:
            frame.pack_forget()

        # Add sidebar buttons based on json
        self.settings.load_devices()
        for serial_number, data in self.settings.devices.items():
            self.build_device(serial_number, data)

    def build_device(self, serial_number, data):
        button = ttk.Button(self.content, text=data["Name"], width=15,
                            command=lambda: self.settings.show_view(serial_number, self))
        button.pack(fill=tk.X, pady=2)

        self.devices.append(button)