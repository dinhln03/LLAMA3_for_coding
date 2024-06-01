import sys
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import smtplib
import logging

logger = logging.getLogger("crawler")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('sc_appointment_check.log')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)


def gmail_login():
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.ehlo()
    server.starttls()
    server.login(sys.argv[3], sys.argv[4])
    return server


def verify_gmail():
    try:
        server = gmail_login()
        server.close()
    except StandardError as e:
        print (e)
        logger.error(e)
        exit()


def notify_user(month):
    FROM = sys.argv[3]
    TO = sys.argv[3]
    SUBJECT = "[SC Application] Vacancy found in %s" % month
    TEXT = "Go to below address to catch the slot: https://eappointment.ica.gov.sg/ibook/index.do"

    message = """
              From: %(FROM)s
              To: %(TO)s
              Subject: %(SUBJECT)s

              %(TEXT)
              """ % locals()

    try:
        server = gmail_login()
        server.sendmail(FROM, TO, message)
        server.close()
    except StandardError as r:
        print "failed to send mail %s" % r
        logger.info("failed to send mail %s" % r)
        exit()


def go_to_query_page(driver):
    driver.get("https://eappointment.ica.gov.sg/ibook/index.do")
    driver.switch_to_frame(driver.find_element_by_name("bottomFrame"));
    driver.switch_to_frame(driver.find_element_by_name("mainFrame"));
    driver.find_element_by_name("apptDetails.apptType").send_keys("Singapore Citizen Application")
    driver.find_element_by_name("apptDetails.identifier1").send_keys(sys.argv[1])
    driver.find_element_by_name("apptDetails.identifier2").send_keys(sys.argv[2])
    driver.find_element_by_name("Submit").send_keys(Keys.ENTER)


def contains_released_dates(driver):
    days = driver.find_elements_by_css_selector("td[class^='cal_']")
    return any(day.get_attribute("class") in ("cal_AF", "cal_AD") for day in days)


def get_month(driver):
    year = int(driver.find_element_by_name("calendar.calendarYearStr").get_attribute("value"))
    month = int(driver.find_element_by_name("calendar.calendarMonthStr").get_attribute("value")) + 1
    return "%d%.2d" % (year, month)


def check_free_slots(driver):
    days = driver.find_elements_by_css_selector("td[class^='cal_']")
    current_month = get_month(driver)
    if any("cal_AD" in day.get_attribute("class") for day in days):
        logger.info("Slots found in %s" % current_month)
        notify_user(current_month)


def check():
    driver = webdriver.Chrome()
    go_to_query_page(driver)
    while contains_released_dates(driver):
        check_free_slots(driver)
        driver.execute_script("doNextMth(document.forms[0]);")
    logger.info("Checked until %s, no available slots found." % get_month(driver))
    driver.close()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Please refer to the readme file for proper usage.")
    else:
        verify_gmail()
        retry_interval = sys.argv[5] if len(sys.argv) > 5 else 60
        while True:
            check()
            time.sleep(retry_interval)
