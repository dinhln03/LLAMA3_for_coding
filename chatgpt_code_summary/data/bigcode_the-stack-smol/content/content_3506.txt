from openpyxl.workbook import Workbook
from scrapy import Selector
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
from csv import writer

driver_path = 'D:\\Application\\installers\\ChromeDriver\\chromedriver.exe'


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='', encoding="utf-8") as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def parse_house(link):
    driver2 = webdriver.Chrome(executable_path=driver_path)
    driver2.maximize_window()
    driver2.get(link)

    house_info = []

    page_source = driver2.page_source
    response2 = Selector(text=page_source)

    title = response2.css('.kt-page-title h1::text').get()
    address = response2.css('.kt-page-title__subtitle.kt-page-title__subtitle--responsive-sized::text').get()
    area = response2.css('.kt-group-row-item__value::text').get()
    year = response2.css('.kt-group-row-item__value::text')[1].get()
    rooms = response2.css('.kt-group-row-item__value::text')[2].get()
    price = response2.css('.kt-unexpandable-row__value::text').get()
    price_per_meter = response2.css('.kt-unexpandable-row__value::text')[1].get()
    elevator = response2.css('span.kt-group-row-item__value.kt-body.kt-body--stable::text')[0].get()
    parking = response2.css('span.kt-group-row-item__value.kt-body.kt-body--stable::text')[1].get()
    warehouse = response2.css('span.kt-group-row-item__value.kt-body.kt-body--stable::text')[2].get()
    date = response2.css('.time::text').get()

    house_info.append(title)
    house_info.append(address)
    house_info.append(area)
    house_info.append(year)
    house_info.append(rooms)
    house_info.append(price)
    house_info.append(price_per_meter)
    house_info.append(elevator)
    house_info.append(parking)
    house_info.append(warehouse)
    house_info.append(date)

    append_list_as_row('Tehran House Data.csv', house_info)
    driver2.quit()


def parse_neighborhood(link):
    driver1 = webdriver.Chrome(executable_path=driver_path)
    driver1.maximize_window()
    driver1.get(link)

    for i in range(8):
        driver1.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)
        sel = driver1.page_source
        response1 = Selector(text=sel)

        for cards in response1.css('div.post-card-item.kt-col-6.kt-col-xxl-4'):
            link = cards.css('a').attrib['href']
            house_link = "https://divar.ir" + link
            parse_house(house_link)
            time.sleep(1)

    driver1.quit()


def parse():
    driver = webdriver.Chrome(executable_path=driver_path)
    driver.maximize_window()
    driver.get("https://divar.ir/s/tehran/buy-apartment")
    driver.implicitly_wait(5)
    driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/aside/div/div[1]/div[2]/div[1]").click()
    driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/aside/div/div[1]/div[2]/div[2]/div/button").click()
    component = driver.find_element(By.XPATH, "/html/body/div[2]/div/article/div/div/div/div")

    neighborhoods = []
    subtitles = []
    links = []

    for number in range(0, 29280, 650):
        driver.execute_script(f"arguments[0].scrollTop = {number}", component)

        sel = driver.page_source
        response = Selector(text=sel)

        for part in response.css('div.kt-control-row.kt-control-row--large.kt-control-row--clickable'):
            neighborhood = part.css('.kt-control-row__title::text').get()
            neighborhoods.append(neighborhood)
            subtitle = part.css('.kt-base-row__description.kt-body--sm::text').get()
            subtitles.append(subtitle)
            link = part.css('.kt-control-row__title').attrib['href']
            links.append(link)
            print(type(links))
    counter = 1

    set_links = set(links)

    for element in set_links:
        counter += 1
        if counter <= 5:
            continue

        neighborhood_link = "https://divar.ir" + element
        parse_neighborhood(neighborhood_link)


parse()
