class AdminCatalogHelper:

    def __init__(self, app):
        self.app = app

    def go_though_each_product_and_print_browser_log(self):
        for i in range(len(self.app.wd.find_elements_by_css_selector('.dataTable td:nth-of-type(3) a[href*="&product_id="]'))):
            self.app.wd.find_elements_by_css_selector('.dataTable td:nth-of-type(3) a[href*="&product_id="]')[i].click()
            [print(log) for log in self.app.wd.get_log("browser")]
            self.app.wait_for_element_to_be_visible('#tab-general')
            self.app.wd.find_element_by_css_selector('button[name="cancel"]').click()
