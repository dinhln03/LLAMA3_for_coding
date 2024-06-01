import csv


def save2csv(dst_fh, row):
    """
    Appends a list with data to a dst_fh csv
    args:
        dst_fh: str, output file
        row: list, list of values to write in a row
    """

    with open(dst_fh, "a", encoding="utf-8") as csvfile:
        out = csv.writer(
            csvfile,
            delimiter=",",
            lineterminator="\n",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )
        try:
            out.writerow(row)
        except UnicodeEncodeError:
            pass
