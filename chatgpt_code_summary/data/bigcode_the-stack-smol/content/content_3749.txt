import requests
import logging
import os
import selenium
import unittest
import time
import requests, re
from django.core.management.base import BaseCommand
from search.models import Product, Category, DetailProduct
from django.db import IntegrityError 
from django.core.exceptions import MultipleObjectsReturned
from logging.handlers import RotatingFileHandler
from logging import handlers
from configparser import ConfigParser
from django.test import RequestFactory
from django.contrib.auth.models import User
from django.contrib.auth.tokens import default_token_generator
from django.core import mail
from django.http import request, HttpRequest
from django.utils.http import base36_to_int, int_to_base36
from django.utils.http import urlsafe_base64_encode
from django.db.models.query_utils import Q
from django.utils.encoding import force_bytes
from django.contrib.auth import get_user_model
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


class Command(BaseCommand):
    help = "Tests Selenium"

    def __init__(self):
        if os.environ.get("ENV") == "DEV":
            self.driver = webdriver.Firefox("/Users/david/Projets/selenium driver/")
            self.url = "http://127.0.0.1:8000/"
            self.driver.maximize_window()

        if os.environ.get("ENV") == "TRAVIS":
            self.BROWSERSTACK_URL = 'https://davidbarat1:FxhRcmmHYxhSpVrjeAWu@hub-cloud.browserstack.com/wd/hub'
            self.desired_cap = {
                'os' : 'Windows',
                'os_version' : '10',
                'browser' : 'Chrome',
                'browser_version' : '80',
                'name' : "P8 Test"
                }
            self.driver = webdriver.Remote(
                command_executor=self.BROWSERSTACK_URL,
                desired_capabilities=self.desired_cap)
            self.driver.maximize_window()
            self.url = "http://167.99.212.10/"

        self.search = "Nutella"
        self.user = "test@test.com"
        self.password = "007Test!"
        self.newpassword = "newpassword456"
    
    def handle(self, *args, **options):
        self.testMyProducts()
        self.testMentionsContacts()
        # self.testResetPassword()
        self.tearDown()

    def testResetPassword(self):
        # self.driver.maximize_window()
        self.driver.get(self.url)
        time.sleep(5)
        self.elem = self.driver.find_element_by_id("login")
        self.elem.send_keys(Keys.RETURN)
        time.sleep(5)
        self.elem = self.driver.find_element_by_id("id_username")
        self.elem.send_keys(self.user)
        self.elem = self.driver.find_element_by_id("id_password")
        self.elem.send_keys(self.password)
        self.elem.send_keys(Keys.RETURN)
        time.sleep(3)
        self.elem = self.driver.find_element_by_id("logout")
        self.elem.send_keys(Keys.RETURN)
        time.sleep(3)
        self.elem = self.driver.find_element_by_id("login")
        self.elem.send_keys(Keys.RETURN)
        time.sleep(5)
        self.elem = self.driver.find_element_by_id("resetpassword")
        self.elem.send_keys(Keys.RETURN)
        time.sleep(3)
        self.elem = self.driver.find_element_by_id("id_email")
        self.elem.send_keys(self.user)
        time.sleep(3)
        self.user_filter = User.objects.filter(Q(email=self.user))
        for self.user in self.user_filter:
            print(self.user)
            self.token = default_token_generator.make_token(self.user)
            print(self.token)
            self.uid = urlsafe_base64_encode(force_bytes(self.user.pk))
            print(self.uid)
        self.driver.get(self.url + "reset/%s/%s/" % (self.uid, self.token))
        time.sleep(3)
        self.driver.find_element_by_id("id_new_password1").send_keys(self.newpassword)
        self.driver.find_element_by_id("id_new_password2").send_keys(self.newpassword)
        self.elem = self.driver.find_element_by_id("id_new_password2")
        time.sleep(3)
        self.elem.send_keys(Keys.RETURN)
        time.sleep(3)
        self.driver.quit()

    def testMyProducts(self):
        # self.driver.maximize_window()
        self.driver.get(self.url)
        self.elem = self.driver.find_element_by_id("myproducts")
        self.elem.send_keys(Keys.RETURN)
        time.sleep(5)
        self.elem = self.driver.find_element_by_id("id_username")
        self.elem.send_keys(self.user)
        self.elem = self.driver.find_element_by_id("id_password")
        self.elem.send_keys(self.password)
        self.elem.send_keys(Keys.RETURN)
        time.sleep(5)

    def testMentionsContacts(self):
        # self.driver.maximize_window()
        self.driver.get(self.url)
        self.elem = self.driver.find_element_by_id("mentions")
        self.elem.send_keys(Keys.RETURN)
        time.sleep(5)
        self.elem = self.driver.find_element_by_id("contact")
        self.elem.send_keys(Keys.RETURN)
        time.sleep(5)

    def tearDown(self):
        self.driver.quit()
    