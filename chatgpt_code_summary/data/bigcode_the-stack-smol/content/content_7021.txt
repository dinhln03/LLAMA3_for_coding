from selenium import webdriver
import time

url = "http://localhost/litecart/admin/"

browser = webdriver.Chrome()
browser.implicitly_wait(1)

without_title = 0

try:
    browser.get(url)

    # логинемся
    login = browser.find_element_by_css_selector("[name='username']")
    login.send_keys("admin")

    password = browser.find_element_by_css_selector("[name='password']")
    password.send_keys("admin")

    button = browser.find_element_by_css_selector("[name='login']")
    button.click()

    time.sleep(1)   # без этого слипа программа перестает работать, очень хотелось бы обсудить этот момент

    # читаем основное меню
    main_menu = browser.find_elements_by_css_selector("#box-apps-menu > li")
    for i in range(len(main_menu)):
        main_menu_temp = browser.find_elements_by_css_selector("#box-apps-menu > li")
        main_menu_temp[i].click()

        # читаем подменю
        sub_menu = browser.find_elements_by_css_selector(".docs > li")

        # условие для пунктов меню, в которых отсутствует подменю
        if len(sub_menu) < 1:
            title = browser.find_element_by_css_selector("#content > h1").text
            if len(title) == 0:
                without_title += 1

        for j in range(len(sub_menu)):
            sub_menu_temp = browser.find_elements_by_css_selector(".docs > li")
            sub_menu_temp[j].click()

            title = browser.find_element_by_css_selector("#content > h1").text
            if len(title) == 0:
                without_title += 1
    if without_title > 0:
        print('BUG!')
    else:
        print('NO BUG')
finally:
    browser.quit()
