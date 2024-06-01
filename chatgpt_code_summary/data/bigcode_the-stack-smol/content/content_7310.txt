#! /usr/bin/env python3.6

from selenium import webdriver
import time

browser = webdriver.Chrome(executable_path='/home/coslate/anaconda3/bin/chromedriver')
#url = 'https://stats.nba.com/leaders'
url = 'http://stats.nba.com/teams/traditional/#!?sort=W_PCT&dir=-1'
browser.get(url)
time.sleep(5)
#browser.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/div[1]/div[1]/div/div/label/select/option[3]').click()
#browser.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/div[1]/div[2]/div/div/label/select/option[2]').click()
#browser.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/nba-stat-table/div[3]/div/div/select/option[1]').click()

#table = browser.find_element_by_class_name('nba-stat-table__overflow')
table = browser.find_elements_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/nba-stat-table/div[2]/div[1]/table/tbody')
line1 = browser.find_element_by_xpath('//tr[@index="0"]')
print(line1.text)
print("All the window handles : ")
print(browser.window_handles)  # 查看所有window handles
print("The current window handle : ")
print(browser.current_window_handle)  # 查看所有window handles

browser.close()

