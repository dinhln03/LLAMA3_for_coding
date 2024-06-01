import re

from mail_app.mail import Mail
from mail_app.mail_processors.abstract_processor import AbstractProcessor
from mail_app.processed_mail import ProcessedMail

class PasswordProcessor(AbstractProcessor):

    general_keywords = ["password (reset|request|update|updated)", "(new|reset|change|updated|changed your) password",
                        "address verification", "(confirm|confirm your|activate your) (registration|account)"]

    def __init__(self):
        super().__init__()
        self.category = "Password"

    def process(self, mail):
        if self.__general_conditions(mail):
            return ProcessedMail(mail.user_id, mail.message_id, mail.from_, self.category, mail.body, mail.time, mail.attachments)

    ############################################ Conditions ############################################

    def __general_conditions(self, mail: Mail):
        return (any(re.search(keyword, mail.subject.lower()) for keyword in self.general_keywords) or
            any(re.search(keyword, mail.body.lower()) for keyword in self.general_keywords) or
            any(re.search(keyword, name.lower()) for name, _ in mail.attachments.items() for keyword in self.general_keywords))