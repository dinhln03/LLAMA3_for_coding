import os
os.chdir("./export")

from reader.csv_mod import CsvReader
from reader.sarif_mod import SarifReader
from reader.server_mod import RestfulReader
from export.export import Exporter

def generate(args):
    project_name = args.name
    
    sarif_list = args.sarif
    if sarif_list == None:
        sarif_list = []
    
    json_list = args.json
    if json_list == None:
        json_list = []

    csv_list = args.csv
    if csv_list == None:
        csv_list = []

    proj_data = []

    sarif_reader = SarifReader()
    for f in sarif_list:
        sarif_reader.read(f)
    sarif_data = sarif_reader.get_data()
    proj_data.extend(sarif_data['data'])


    csv_reader = CsvReader()
    for f in csv_list:
        csv_reader.read(f)
    csv_data = csv_reader.get_data()
    proj_data.extend(csv_data['data'])

    restful_reader = RestfulReader()
    for rid in json_list:
        restful_reader.read(rid)
    restful_data = restful_reader.get_data()
    proj_data.extend(restful_data['data'])

    reporter = Exporter()
    reporter.setData(proj_data)
    return reporter.build(project_name)

#r = SarifReader()
#r.read('/home/heersin/blackhole/codeql/result.sarif')

#print(os.getcwd())

#project_name = "socat"
#pdf_factory = Exporter()
#pdf_factory.setData(r.get_data())
#pdf_factory.build(project_name)