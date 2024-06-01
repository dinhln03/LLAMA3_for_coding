import requests
from logging import getLogger
from bs4 import BeautifulSoup
from requests import Session
from typing import List
from urllib.parse import quote_plus
from .book import Book
from ..base import HEADERS


class BahanAjar:
    def __init__(self, email: str, password: str, login: bool = True):
        self.session: Session = Session()
        self.session.headers.update(HEADERS)
        self.email = email
        self.password = password
        self._my_books: List[Book] = []
        self.logger = getLogger(self.__class__.__name__)
        if login and self.login():
            self.logger.debug("Berhasil login ke bahan ajar")

    def login(self, email: str = None, password: str = None) -> bool:
        try:
            email = email if email else self.email
            password = password if password else self.password
            url = f"http://bahanajar.ut.ac.id/Homes/login_frame/{email}/{password}//////?service="
            res = self.session.post(url)
            return res.ok
        except Exception as E:
            self.logger.exception(E)
        return False

    @property
    def my_books(self) -> List[Book]:
        if self._my_books:
            return self._my_books
        url = "http://bahanajar.ut.ac.id/Homes/my_books"
        res = self.session.get(url)
        if not res.ok or "No books are available" in res.text:
            return []
        soup: BeautifulSoup = BeautifulSoup(res.text, "lxml")
        soup = soup.find("div", id="bookHolder").find_all(
            "div", class_="publib_bkthumb"
        )
        if not len(soup) > 0:
            return []
        else:
            self._my_books = [Book.from_bkthumb(bktm) for bktm in soup]
        return self._my_books

    @my_books.deleter
    def my_books(self):
        del self._my_books

    @staticmethod
    def search(query: str, start: int = 0) -> List[Book]:
        url = f"http://bahanajar.ut.ac.id/ebookstore/ajax_load_search_books/0/{quote_plus(query)}"
        res = requests.get(url)
        if not res.ok:
            return []
        soup: BeautifulSoup = BeautifulSoup(res.text, "lxml")
        soup = soup.find("div", class_="book_stnd").find_all("div", class_="newb_bg")
        if not len(soup) > 0:
            return []
        return [Book.from_newb_bg(newb_bg) for newb_bg in soup]
