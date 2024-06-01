""" A QuoteController Module """

from masonite.controllers import Controller
from masonite.request import Request
from app.Quote import Quote

class QuoteController(Controller):
    def __init__(self, request: Request):
        self.request = request

    def show(self):
        id = self.request.param("id")
        return Quote.find(id)

    def index(self):
        return Quote.all()

    def create(self):
        subject = self.request.input("subject")
        quote = Quote.create({"subject": subject})
        return quote

    def update(self):
        subject = self.request.input("subject")
        id = self.request.param("id")
        Quote.where("id", id).update({"subject": subject})
        return Quote.where("id", id).get()

    def destroy(self):
        id = self.request.param("id")
        quote = Quote.where("id", id).get()
        Quote.where("id", id).delete()
        return quote
