from docxtpl import DocxTemplate
import csv
import json
import random
#случайный авто
with open('Car_info.txt') as file:
    car_rand = []
    reader = csv.reader(file)
    for row in file:
        car_rand.append(row)
report_car = car_rand[random.randint(0, len(car_rand)-1)]
car_info = report_car.split()
#О авто
def get_data (Brand, Model, Fuel_cons, Price):
    return {
        'Название': Brand,
        'Модель': Model,
        'Объем': Fuel_cons,
        'Цена': Price
    }
def from_template(Brand, Model, Fuel_cons, Price, template):
    template = DocxTemplate(template)
    data = get_data(Brand, Model, Fuel_cons, Price)
    template.render(data)
    template.save('О_машине.docx')
def report(Brand, Model, Fuel_cons, Price):
    template = 'О_машине.docx'
    document = from_template(Brand, Model, Fuel_cons, Price, template)
report(car_info[0], car_info[1], car_info[2], car_info[3])
#csv файл
car_list=[]
with open('Авто_инфо.txt', 'r') as file:
    for row in file:
        inner_list = [x.strip() for x in row.split(',')]
        car_list.append(inner_list)
print(car_list)
with open('car.csv', 'w') as file:
        writer = csv.writer(file, delimiter = '*')
        writer.writerows(car_list)
#json файл
with open('Авто_json.txt', 'w') as f:
    json.dump(str(car_info), f)