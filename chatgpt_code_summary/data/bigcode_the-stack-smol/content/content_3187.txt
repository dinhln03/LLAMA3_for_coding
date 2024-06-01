# -*- coding: utf-8 -*-
__author__ = 'abbot'

from selenium import webdriver
from selenium.webdriver import ActionChains

driver = webdriver.PhantomJS(executable_path='/Users/wangbo/Downloads/phantomjs-2.1.1-macosx/bin/phantomjs')

ac = driver.find_element_by_xpath('element')

ActionChains(driver).move_to_element(ac).perform()

ActionChains(driver).move_to_element(ac).click(ac).perform()

