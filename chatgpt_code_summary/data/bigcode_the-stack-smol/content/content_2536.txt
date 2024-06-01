# -*- coding: utf-8 -*-

# IFD.A-8 :: Версия: 1 :: Проверка ввода невалидного значения в поле "код IATA" для выбора аэропорта
# Шаг 1
def test_check_invalid_value_IATA_to_select_airport(app):
    app.session.enter_login(username="test")
    app.session.enter_password(password="1245")
    app.airport.open_form_add_airport()
    app.airport.enter_IATA_code(iata_cod="QWE")
    app.airport.search_airport_by_parameter()
    app.airport.message_no_airports()
    app.airport.exit_from_the_add_airport_form()
    app.session.logout()

# IFD.A-8 :: Версия: 1 :: Проверка ввода невалидного значения в поле "код IATA" для выбора аэропорта
# Шаг 2
def test_check_invalid_characters_in_IATA_code(app):
    app.session.enter_login(username="test")
    app.session.enter_password(password="1245")
    app.airport.open_form_add_airport()
    app.airport.enter_IATA_code(iata_cod="!№;%:?*")
    app.airport.wait_massege_no_airport()
    app.airport.exit_from_the_add_airport_form()
    app.session.logout()