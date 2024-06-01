from nose.tools import assert_equals
from framework.pages.loginPage import loginPage
from framework.pages.headerPage import headerPage
from framework.core.webdriverfactory import WebDriverFactory
from framework.core.configuration import webdriver_configuration


class testLogin():

    baseUrl = "http://twiindan.pythonanywhere.com/admin"

    @classmethod
    def setup_class(self):
        wdf = WebDriverFactory(webdriver_configuration)
        self.driver = wdf.getWebDriverInstance()
        self.login_page = loginPage(self.driver)

    def setup(self):
        self.login_page.navigate()

    def test_correct_login(self):
        self.login_page.locate_elements()
        self.login_page.fillUsername('user1')
        self.login_page.fillPassword('selenium')
        self.login_page.submitClick()



    @classmethod
    def teardown_class(self):
        self.driver.quit()


