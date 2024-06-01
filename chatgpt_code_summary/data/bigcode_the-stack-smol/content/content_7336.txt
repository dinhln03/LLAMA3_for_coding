from kivy.app import App
from kivy.uix.label import Label


class ChildApp(App):
    def build(self):
        return Label(text='Child')

if __name__ == '__main__':
    ChildApp().run()
