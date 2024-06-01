#!/usr/bin/python
#coding:utf-8

import time
import json
import requests
from selenium import webdriver

filename = 'a.csv'
url = 'http://www.icourse163.org/university/view/all.htm#/'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}

# with open(filename, 'w+') as file:
#     file.write("大学,课程,课程时长,课程负载,内容类型,课程分类\n")
file = open(filename, 'w+')
print("大学,课程,课程时长,课程负载,内容类型,课程分类")
file.write("大学,课程,课程时长,课程负载,内容类型,课程分类\n")

browser  = webdriver.PhantomJS()
browser2 = webdriver.PhantomJS()
browser3 = webdriver.PhantomJS()

browser.get(url)

# 大学
university = browser.find_elements_by_class_name("u-usity")
for i in university:
    university_url = i.get_attribute("href")
    university_name = i.find_element_by_id("").get_attribute("alt")

    browser2.get(university_url)


# 课程
    course = browser2.find_elements_by_class_name("g-cell1")
    for j in course:
        course_url = "http://www.icourse163.org" + j.get_attribute("data-href")
        course_name = j.find_element_by_class_name("card").find_element_by_class_name("f-f0").text

        browser3.get(course_url)
    

# 课程信息
        course_text = browser3.find_elements_by_class_name("block")
        try:           
            k0 = course_text[0].find_element_by_class_name("t2").text
            k1 = course_text[1].find_element_by_class_name("t2").text
            k2 = course_text[2].find_element_by_class_name("t2").text
            k3 = course_text[3].find_element_by_class_name("t2").text
        except Exception as e:
            k3 = k2
            k2 = k1
            k1 = None
            K0 = None
        finally:
            print("%s,%s,%s,%s,%s,%s" % (university_name,course_name,k0,k1,k2,k3))        
            file.write("%s,%s,%s,%s,%s,%s\n" % (university_name,course_name,k0,k1,k2,k3))

        # with open(filename, 'a+') as file:
        #     file.write("%s,%s,%s,%s,%s,%s\n" % (university_name,course_name,k0,k1,k2,k3))

browser3.close()
browser2.close()
browser.close()
