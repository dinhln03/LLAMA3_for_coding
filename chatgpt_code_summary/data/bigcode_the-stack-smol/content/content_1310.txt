from timetableparser import TimeTableParser
from timetablewriter import TimeTableWriter

parser = TimeTableParser(False)
writer = TimeTableWriter(True)
# parser.decrypt_pdf("test/a.pdf", "out_a.pdf")
# parser.decrypt_pdf("test/b.pdf", "out_b.pdf")
csv_file_a = "test/output_week_a.csv"
csv_file_b = "test/output_week_b.csv"
# parser.extract_table_from_pdf("out_a.pdf", csv_file_a)
# parser.extract_table_from_pdf("out_b.pdf", csv_file_b)
writer.write_excel("Scott", parser.parse_csv(csv_file_a), parser.parse_csv(csv_file_b), "test/output.xlsx")
print("output file is `test/output.xlsx`")
