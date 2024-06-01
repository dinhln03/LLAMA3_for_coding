#   Copyright (c) 2019 Bernd Wiesner. bernduwiesner@yahoo.co.uk
#   All rights reseArgumentS_Resultsed
#
""" Display the command line options in a window

"""

from argparse import Namespace
from typing import Tuple, Union

import PySimpleGUI as sg

import constants as C
from gui_utility import popup_window

ArgumentsResults = Tuple[Union[str, None], Namespace]


def arguments_window(args: Namespace) -> ArgumentsResults:
    """Window interface

    :param args: the arguments passed from the command line
    :return: Tuple[Union[str, None], Namespace] - The new arguments
    """
    filename: str = C.SAVE_FILE_DIR + args.lottery_type + C.SAVE_FILE_TYPE
    layout = [
        [
            sg.Text(text="Lottery type:"),
            sg.InputCombo(
                values=tuple(C.LOTTERY_TYPES),
                default_value=args.lottery_type,
                readonly=True,
                enable_events=True,
                size=(10, 1),
                tooltip="Choose a lottery type",
                key=C.ELEMENT_NAMES["LOTTO"],
            ),
            sg.Frame(
                layout=[
                    [
                        sg.Text(text="Number of lines"),
                        sg.InputText(
                            default_text=args.number_of_lines,
                            enable_events=True,
                            size=(3, 1),
                            justification="right",
                            key=C.ELEMENT_NAMES["COUNT"],
                        ),
                    ]
                ],
                title="",
                tooltip="Choose the number of lines to generate",
                relief=sg.RELIEF_FLAT,
                key=C.ELEMENT_NAMES["LINES"],
            ),
        ],
        [
            sg.Frame(
                layout=[
                    [
                        sg.Radio(
                            text="Save",
                            group_id="R",
                            default=not args.no_save,
                            tooltip="Save the generated numbers",
                            enable_events=True,
                            key=C.ELEMENT_NAMES["SAVE"],
                        ),
                        sg.Radio(
                            text="Do NOT save",
                            group_id="R",
                            default=args.no_save,
                            tooltip="Do not save the generated numbers",
                            enable_events=True,
                            key=C.ELEMENT_NAMES["NOSAVE"],
                        ),
                        sg.Radio(
                            text="Delete",
                            group_id="R",
                            default=args.delete,
                            enable_events=True,
                            tooltip="Delete a saved file",
                            key=C.ELEMENT_NAMES["DELETE"],
                        ),
                        sg.Radio(
                            text="Show",
                            group_id="R",
                            default=args.print,
                            tooltip="Display a previously saved file",
                            enable_events=True,
                            key=C.ELEMENT_NAMES["SHOW"],
                        ),
                    ]
                ],
                title="Saved file options",
                relief=sg.RELIEF_SOLID,
                size=(0, 40),
            )
        ],
        [
            sg.Text(
                text="File name: " + filename,
                key=C.ELEMENT_NAMES["FILENAME"],
                size=(50, 2),
                tooltip="The name of the file to save or to display",
                justification="left",
            )
        ],
        [
            sg.OK(key="OK", focus=True),
            sg.Quit(key="Cancel", tooltip="Do nothing and quit"),
        ],
    ]

    window = sg.Window(
        title="Lottery number Generator Arguments",
        layout=layout,
        text_justification=C.GUI_JUSTIFY,
        font=(C.GUI_FONT_NAME, C.GUI_FONT_SIZE),
    )

    while True:
        event, values = window.Read()
        if event == C.ELEMENT_NAMES["DELETE"]:
            window.Element(key="OK").Update("Delete Saved File")
            window.Element(key=C.ELEMENT_NAMES["LINES"]).Update(visible=False)
            window.Element(key=C.ELEMENT_NAMES["FILENAME"]).Update(
                "File to delete: " + filename
            )
        elif event == C.ELEMENT_NAMES["SHOW"]:
            window.Element(key="OK").Update("Show Saved File")
            window.Element(key=C.ELEMENT_NAMES["LINES"]).Update(visible=False)
            window.Element(key=C.ELEMENT_NAMES["FILENAME"]).Update(
                "File to display: " + filename
            )
        elif event in (C.ELEMENT_NAMES["NOSAVE"], C.ELEMENT_NAMES["SAVE"]):
            window.Element(key="OK").Update("Generate Numbers")
            window.Element(key=C.ELEMENT_NAMES["LINES"]).Update(visible=True)
            if event == C.ELEMENT_NAMES["NOSAVE"]:
                window.Element(key=C.ELEMENT_NAMES["FILENAME"]).Update(
                    "File will not be saved"
                )
            elif event == C.ELEMENT_NAMES["SAVE"]:
                window.Element(key=C.ELEMENT_NAMES["FILENAME"]).Update(
                    "Will be saved as: " + filename
                )
        if event == C.ELEMENT_NAMES["LOTTO"]:
            filename = (
                C.SAVE_FILE_DIR + values[C.ELEMENT_NAMES["LOTTO"]] + C.SAVE_FILE_TYPE
            )
            window.Element(key=C.ELEMENT_NAMES["FILENAME"]).Update(
                "File name: " + filename
            )
        elif event == C.ELEMENT_NAMES["COUNT"]:
            if values[C.ELEMENT_NAMES["COUNT"]].isnumeric():
                temp = int(values[C.ELEMENT_NAMES["COUNT"]])
            else:
                temp = False
            if temp < C.MIN_LINES or temp > C.MAX_LINES:
                elem = window.Element(key=C.ELEMENT_NAMES["COUNT"])
                elem.Update(C.DEFAULT_LINES)
                msg = "number of lines must be in the range 1-100"
                popup_window(text=msg)
        elif event == "OK" or event == "Cancel" or event is None:
            break

    if event != "Cancel" and event is not None:
        args.lottery_type = values[C.ELEMENT_NAMES["LOTTO"]]  # str
        args.number_of_lines = int(values[C.ELEMENT_NAMES["COUNT"]])  # int
        args.delete = values[C.ELEMENT_NAMES["DELETE"]]  # bool
        args.print = values[C.ELEMENT_NAMES["SHOW"]]  # bool
        args.no_save = values[C.ELEMENT_NAMES["NOSAVE"]]  # bool

    window.Close()
    return event, args
