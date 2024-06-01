from selenium import webdriver
from selenium.webdriver.chrome.options import Options

browser = webdriver.Chrome("./driver/chromedriver.exe")

options = Options()
#options.headless = True
browser = webdriver.Chrome(executable_path="./driver/chromedriver.exe", options=options)
browser.get("https://center-pf.kakao.com/")