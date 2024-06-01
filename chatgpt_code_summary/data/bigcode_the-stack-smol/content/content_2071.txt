# Qiwi module advanced usage example v1.00
# 17/05/2021
# https://t.me/ssleg  © 2021

import logging

import qiwi_module

# настройка логфлайла test,log, туда будут записываться все ошибки и предупреждения.
lfile = logging.FileHandler('test.log', 'a', 'utf-8')
lfile.setFormatter(logging.Formatter('%(levelname)s %(module)-13s [%(asctime)s] %(message)s'))
# noinspection PyArgumentList
logging.basicConfig(level=logging.INFO, handlers=[lfile])

# простой вариант использования смотрите в файле sample.py

# если у вас настроен свой внешний вид формы платежа, необходимо передать код темы модулю.
# это делается один раз, при его инициализации.
# сам код и настройки формы находятся на странице https://qiwi.com/p2p-admin/transfers/link
theme_code = 'Ivanov-XX-vvv-k_'

# перед любым использованием необходима однократная инициализация модуля.
qiwi_module.init(theme_code)

# создание счета на 1 рубль. При успехе получаете url с формой оплаты для клиента.
# при неуспехе возвращается False с подробной записью в лог.

# идентификаторы счетов придумываете и сохраняете вы сами, они должны быть уникальными всегда.
bill_id = 'bill_2021_00000002'

# по умолчанию счет действителен 15 минут, но вы можете поставить свое время, например сутки и 1 минуту.
valid_hours = 24
valid_minutes = 1

# есть так же поле для комментария, его видит клиент в форме оплаты. например, туда можно записать детали заказа
comment = 'Винт с левой резьбой для Сидорова.'

invoice_url = qiwi_module.create_bill(1.00, bill_id, comment, valid_hours, valid_minutes)
print(invoice_url)

# проверка статуса оплаты.
# возвращает одно из четырех возможных значений, если успешно или False и запись в лог.
# 'WAITING' - cчет выставлен, ожидает оплаты.
# 'PAID' - cчет оплачен.
# 'REJECTED' -	счет отменен с вашей стороны.
# 'EXPIRED' - счет не оплачен и истек срок его действия.
# можно вызывать ежесекундно или реже.
pay_status = qiwi_module.bill_status(bill_id)
print(pay_status)

# отмена счета, если вам это необходимо.
# возврашает 'REJECTED' если успешно, иначе False и запись в лог.
bill_status = qiwi_module.cancel_bill(bill_id)
print(bill_status)
