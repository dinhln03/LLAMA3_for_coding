from mycroft import MycroftSkill, intent_file_handler
import subprocess

class Fortune(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self)

    @intent_file_handler('fortune.intent')
    def handle_fortune(self, message):
        result = subprocess.run("fortune", capture_output=True, text=True)
        self.speak_dialog(result.stdout)


def create_skill():
    return Fortune()

