from selenium_test.selenium_utils import *
from file_and_system.windows_os_utils import WindowsOsUtil
from python_common.global_param import GlobalParam
from http_request.request_utils import request_download_file_by_url
import cv2 as cv
import time

WindowsOsUtil.kill_process_by_name('MicrosoftWebDriver.exe')
# mail_lists=['mail.hoperun.com', 'mail.qq.com', 'mail.163.com]
mail_lists = ['mail.163.com']
mail_driver = init_driver('edge', GlobalParam.get_edge_driver_path())
open_browser_multi_tab(mail_driver, mail_lists)
wait_for_page_full_loaded(mail_driver)


def hoperun_login(hoperun_driver, user_name, user_pass):
    hoperun_driver.execute_script("document.getElementById('usernameTip').removeAttribute('readonly');")
    element = find_element_by_id(hoperun_driver, 'usernameTip')
    element.click()
    element = find_element_by_id(hoperun_driver, 'username')
    element.send_keys(user_name)
    element = find_element_by_id(hoperun_driver, 'userType')
    element.click()
    element = find_element_by_id(hoperun_driver, 'userTypePwd')
    element.send_keys(user_pass)
    element = find_element_by_id(hoperun_driver, 'wmSubBtn')
    element.click()


def hoperun_check_mail(hoperun_driver, mail_sender, mail_title):
    wait_for_frame_and_switch_to_frame(hoperun_driver, 'treeBox')
    element = find_element_by_id(hoperun_driver, 'tree_folder_1_span')
    element.click()
    wait_for_page_full_loaded(hoperun_driver)
    wait_for_frame_and_switch_to_frame(hoperun_driver, 'tabsHome')
    wait_for_page_full_loaded(hoperun_driver)
    element = hoperun_driver.find_elements_by_xpath(''.join(('//div[text()="', mail_sender, '"]/../../../..')))
    for e in element:
        if e.find_element_by_xpath('li[2]/div[3]/span').text.__contains__(mail_title):
            e.find_element_by_xpath('li[2]/div[3]/span').click()


def qq_login(qq_driver, user_name, user_pass):
    element = find_element_by_id(qq_driver, 'qqLoginTab')
    element.click()
    qq_driver.switch_to.frame('login_frame')
    element = find_element_by_id(qq_driver, 'u')
    element.click()
    element.send_keys(user_name)
    element = find_element_by_id(qq_driver, 'p')
    element.click()
    element.send_keys(user_pass)
    element = find_element_by_id(qq_driver, 'login_button')
    element.click()
    wait_for_frame_and_switch_to_frame(qq_driver, 'tcaptcha_iframe')
    img_element = find_element_by_id(qq_driver, 'slideBg')
    wait_for_element_appeared(qq_driver, img_element)
    big = img_element.get_attribute('src')
    request_download_file_by_url(big, GlobalParam.get_test_image_path() + 'test_qq_mail_big.png')
    img_element = find_element_by_id(qq_driver, 'slideBlock')
    wait_for_element_appeared(qq_driver, img_element)
    small = img_element.get_attribute('src')
    request_download_file_by_url(small, GlobalParam.get_test_image_path() + 'test_qq_mail_small.png')


def netcase_163_login(netcase_163_driver, user_name, user_pass):
    netcase_login_frame = netcase_163_driver.find_element_by_tag_name('iframe')
    wait_for_frame_and_switch_to_frame(netcase_163_driver, netcase_login_frame)
    wait_for_element_exist(netcase_163_driver, '//input[@name="email"]')
    element = find_element_by_name(netcase_163_driver, 'email')
    element.click()
    element.send_keys(user_name)
    wait_for_element_exist(netcase_163_driver, '//input[@name="password"]')
    element = find_element_by_name(netcase_163_driver, 'password')
    element.click()
    element.send_keys(user_pass)
    element = find_element_by_id(netcase_163_driver, 'dologin')
    element.click()
    # ------------------------security mail captcha not show----------------------
    # wait_for_element_exist(netcase_163_driver,'//div[@class="yidun_panel"]')
    # element = find_element_by_class_name(netcase_163_driver, 'yidun_panel')
    # netcase_163_driver.execute_script("arguments[0].style['display'] = 'block';",element)
    # # element = find_element_by_class_name(netcase_163_driver, 'yidun_bg-img')
    # # netcase_mail_captcha = element.get_attribute('src')
    # # request_download_file_by_url(netcase_mail_captcha, test_image_path+'test_netcase_mail_captcha.png')
    # time.sleep(4)
    # element = find_element_by_class_name(netcase_163_driver, 'yidun_refresh')
    # element.click()
    #
    # element = find_element_by_class_name(netcase_163_driver, 'yidun_tips__point')
    # print(element.location)
    #
    # # element = find_element_by_class_name(netcase_163_driver, 'yidun_tips__point')
    # # print(element.get_attribute("innerHTML"))
    # ------------------------security mail captcha not show----------------------


def netcase_163_check_mail(netcase_163_driver, mail_sender, mail_title):
    wait_for_element_to_be_clickable(netcase_163_driver, '//div[@id="_mail_component_140_140"]/span[@title="收件箱"]')
    time.sleep(2)
    # rF0 kw0 nui-txt-flag0 : not read
    # rF0 nui-txt-flag0 : readed
    # element = netcase_163_driver.find_elements_by_xpath('//div[@class="rF0 nui-txt-flag0"]/div/div[2]/span')
    element = netcase_163_driver.find_elements_by_xpath('//div[@class="rF0 nui-txt-flag0"]')
    for e in element:
        print(e.find_element_by_xpath('.//div/div[2]/span').text)
        # if e.text.__contains__(mail_title):
        #     print(e.text)


def qq_captcha_pass():
    big_image = cv.imread(GlobalParam.get_test_image_path() + 'test_qq_mail_big.png')
    small_image = cv.imread(GlobalParam.get_test_image_path() + 'test_qq_mail_small.png')
    cv.imshow('1', small_image)
    cv.waitKey(0)


def netcase_captcha_pass():
    return ''


# login hoperun mail and check mail
# hoperun_login(mail_driver, 'user', 'password')
# wait_for_page_full_loaded(mail_driver)
# hoperun_check_mail(mail_driver, 'sender', 'title')

netcase_163_login(mail_driver, '****', '****')
wait_for_page_full_loaded(mail_driver)
netcase_163_check_mail(mail_driver, '', '123')

# qq_login(mail_driver, '', '')
# netcase_163_login(mail_driver, '', '')
# captcha_pass()
