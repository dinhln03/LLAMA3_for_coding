from client.util.html.tooling.base.HTMLElement import HTMLElement


class ScriptElement(HTMLElement):
    def __init__(self, src):
        super().__init__('script')
        self.set_attribute('src', src)
