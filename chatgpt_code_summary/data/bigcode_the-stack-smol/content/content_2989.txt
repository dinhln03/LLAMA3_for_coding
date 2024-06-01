import cannibalize
import xlsxwriter
import sys
import os

desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
usage = "Kit Cannibaliztion\n" \
        "usage: analyze.py kit_number serial1 serial2 serial3 ..."

if len(sys.argv) < 2:
    print(usage)
else:
    KIT = sys.argv[1]
    SERIALS = [str(i) for i in sys.argv[2:]]

    FILE_NAME = '{}\\cannibalization_report_{}.xlsx'.format(desktop,KIT)

    kit_assembly_data = cannibalize.create_new_kit_assembly(KIT, SERIALS)

    workbook = xlsxwriter.Workbook(FILE_NAME)

    v_i_data = []


    for r in kit_assembly_data['assembly']:
        v_i_data.append([KIT, r['serial'], r['status'], str(len(r['build']))])


    first_worksheet = workbook.add_worksheet('Report')

    first_worksheet.set_column('A:C', 20)

    first_worksheet.add_table('A1:C{}'.format(str(1 + len(v_i_data))),
                        {'data': v_i_data,
                         'columns': [{'header': 'kit_number'},
                                     {'header': 'serial_number'},
                                     {'header': 'status'},
                                     {'header': 'components_in_kit'}
                                     ]})


    for unique_serial in kit_assembly_data['assembly']:

        worksheet = workbook.add_worksheet('Serial ~ {}'.format(unique_serial['serial']))

        worksheet.set_column('A:B', 20)

        worksheet.write(0, 0, 'Serial ~ {}'.format(unique_serial['serial']))
        worksheet.write(0, 1, 'Status: {}'.format(unique_serial['status'].upper()))

        table_data = []
        for component_information in unique_serial['build']:
            table_data.append([component_information['component'],
                               str(component_information['qty'])])

        worksheet.add_table('A2:B{}'.format(str(1 + len(unique_serial['build']))),
                            {'data': table_data,
                             'columns': [{'header': 'component'},
                                         {'header': 'qty_in_kit'}
                                         ]})

    workbook.close()