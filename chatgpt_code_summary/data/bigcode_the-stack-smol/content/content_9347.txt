import re


def validate_sheet_name(sheet_name):
    if len(sheet_name) > 33:
        return False
    return not re.search("[%#&/\*\?\\\]", sheet_name)


def sanitise_sheet_name(sheet_name):

    replaced_with_empty = ["%", "#", "&", "*", "?"]
    replaced_with_hyphens = ["/", "\\"]
    for ch in replaced_with_empty:
        sheet_name = sheet_name.replace(ch, "")
    for ch in replaced_with_hyphens:
        sheet_name = sheet_name.replace(ch, "-")
    return sheet_name[0:33]
