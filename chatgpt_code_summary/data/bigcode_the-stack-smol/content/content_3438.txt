import pytest
from datetime import datetime, timedelta
import pytz
from bs4 import BeautifulSoup
from src.events import Events
from src.users import Users
from src.user import USER_ACCESS_MANAGER
from src.stores import MemoryStore
from src.email_generators import EventLocationChangedEmail


def test_event_location_changed_email():
    store = MemoryStore()
    events = Events(store)
    users = Users(store)
    start = datetime.now(pytz.timezone("America/New_York"))
    dur = timedelta(hours=1)
    end = start + dur
    u = users.add("test@test.com", 'name', 'alias', 'psw', 8)
    e = events.add('test', 'test', 30, start, dur, 'test', 'test',
                   'test@test.com', 'test', u)
    email = EventLocationChangedEmail(e, e, '', root='./src')
    html = email.generate(u)
    soup = BeautifulSoup(html, 'html.parser')
    assert html
    assert type(html) == str
    assert bool(soup.find())
    assert soup.find("div", {"class": "user"}).string.strip() == 'name'
    assert soup.find("a", {"class": "event-link"}).string.strip() == 'test'
    assert soup.find("td", {"class": "event-location-text"}).string.strip() == 'test'
    assert soup.find("div", {"class": "event-description"}).string.strip() == 'test'
