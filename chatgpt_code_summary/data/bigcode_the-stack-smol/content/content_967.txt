#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

if __name__ == '__main__':

    goods = []

    while True:
        command = input(">>> ").lower()

        if command == 'exit':
            break

        elif command == 'add':
            name = input("Название товара: ")
            shop = input("Название магазина: ")
            price = float(input("Стоимость: "))

            good = {
                'name': name,
                'shop': shop,
                'price': price,
            }

            goods.append(good)
            # Отсортировать список в случае необходимости.
            if len(goods) > 1:
                goods.sort(key=lambda item: item.get('shop', ''))

        elif command == 'list':
            line = '+-{}-+-{}-+-{}-+-{}-+'.format(
                '-' * 4,
                '-' * 30,
                '-' * 20,
                '-' * 8
            )
            print(line)
            print(
                '| {:^4} | {:^30} | {:^20} | {:^8} |'.format(
                    "№",
                    "Название",
                    "Магазин",
                    "Цена"
                )
            )
            print(line)

            for idx, good in enumerate(goods, 1):
                print(
                    '| {:>4} | {:<30} | {:<20} | {:>8} |'.format(
                        idx,
                        good.get('name', ''),
                        good.get('shop', ''),
                        good.get('price', 0)
                    )
                )
            print(line)

        elif command.startswith('select '):

            parts = command.split(' ', maxsplit=1)
            shopName = parts[1]
            count = 0

            for good in goods:
                if shopName == good.get('shop', shopName):
                    count += 1
                    print(
                        '{:>4}: {}'.format(count, good.get('name', ''))
                    )

            if count == 0:
                print("Такого магазина не существует либо нет товаров.")

        elif command == 'help':
            print("Список команд:\n")
            print("add - добавить товар;")
            print("list - вывести список товаров;")
            print("select <имя магазина> - запросить товары магазина;")
            print("help - отобразить справку;")
            print("exit - завершить работу с программой.")
        else:
            print(f"Неизвестная команда {command}", file=sys.stderr)