# -*- coding: utf-8 -*-

"""Top-level package for PrecisionMapper."""

import requests
from requests import ConnectionError
from datetime import datetime
from bs4 import BeautifulSoup

__author__ = """Thibault Ducret"""
__email__ = 'hello@tducret.com'
__version__ = '0.0.2'

_DEFAULT_BEAUTIFULSOUP_PARSER = "html.parser"
_SIGNIN_URL = "https://www.precisionmapper.com/users/sign_in"
_SURVEYS_URL = "https://www.precisionmapper.com/surveys"
_SHARED_SURVEYS_URL = "https://www.precisionmapper.com/shared_surveys"

_DEFAULT_DATE = "2000-01-01T00:00:00.000Z"
_RFC3339_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'
_SHORT_DATE_FORMAT = '%d/%m/%Y %H:%M'

_AUTHENTICITY_TOKEN_SELECTOR = 'meta["name"="csrf-token"]'
_SURVEYS_SELECTOR = "#surveysList .tableCellWrap"
_SURVEY_NAME_SELECTOR = "div.surveyName > a['href']"
_SURVEY_LOCATION_SELECTOR = "div.cellWrap.locationWrapper > span"
_SURVEY_DATE_SELECTOR = "div.cellWrap.surveyDateRow .date"
_SURVEY_IMG_NB_AND_SIZE_SELECTOR = "div.surveyotherDetails > span"
_SURVEY_SENSOR_SELECTOR = ".surveySensorWrap"
_SURVEY_URL_SELECTOR = "div.surveyName > a['href']"


class Client(object):
    """ Do the requests with the servers """
    def __init__(self):
        self.session = requests.session()
        self.headers = {
                        'authority': 'www.precisionmapper.com',
                        'origin': 'https://www.precisionmapper.com',
                        'user-Agent': 'Mozilla/5.0 (Macintosh; \
Intel Mac OS X 10_13_5) AppleWebKit/537.36 (KHTML, like Gecko) \
Chrome/67.0.3396.99 Safari/537.36',
                        'referer': 'https://www.precisionmapper.com\
/users/sign_in',
                       }

    def _get(self, url, expected_status_code=200):
        ret = self.session.get(url=url, headers=self.headers)
        if (ret.status_code != expected_status_code):
            raise ConnectionError(
                'Status code {status} for url {url}\n{content}'.format(
                    status=ret.status_code, url=url, content=ret.text))
        return ret

    def _post(self, url, post_data, expected_status_code=200,
              allow_redirects=True):
        ret = self.session.post(url=url,
                                headers=self.headers,
                                data=post_data,
                                allow_redirects=allow_redirects)
        if (ret.status_code != expected_status_code):
            raise ConnectionError(
                'Status code {status} for url {url}\n{content}'.format(
                    status=ret.status_code, url=url, content=ret.text))
        return ret


class Survey(object):
    """ Class for a drone survey (mission) """
    def __init__(
            self, id, name, url, date, sensor="", location="",
            image_nb=0, size="0 MB", thumbnail="", altitude_in_m=0,
            resolution_in_cm=0, area_in_ha=0, drone_platform=""):

        if type(id) != int:
            raise TypeError("id must be an int, not a "+str(type(id)))
        self.id = id

        if type(image_nb) != int:
            raise TypeError("image_nb must be an int, not a " +
                            str(type(image_nb)))
        self.image_nb = image_nb

        self.date = date
        try:
            self.date_obj = _rfc_date_str_to_datetime_object(self.date)
        except:
            raise TypeError("date must respect the format \
                         YYYY-MM-DDTHH:MM:SS.sssZ, received : "+date)

        self.name = name
        self.drone_platform = drone_platform
        self.sensor = sensor
        self.location = location
        self.date_str = _datetime_object_to_short_date_str(self.date_obj)

        self.size = size
        self.thumbnail = thumbnail
        self.altitude_in_m = altitude_in_m
        self.resolution_in_cm = resolution_in_cm
        self.area_in_ha = area_in_ha

    def __str__(self):
        return('[{name}] ({location} - {date}) : {image_nb} images, \
{size}, sensor : {sensor}, id : {id}'.format(
            name=self.name,
            location=self.location,
            date=self.date_str,
            image_nb=self.image_nb,
            size=self.size,
            sensor=self.sensor,
            id=self.id))

    def __repr__(self):
        return("Survey(id={}, name={})".format(self.id, self.name))


class PrecisionMapper(object):
    """ Class for the communications with precisionmapper.com """
    def __init__(self, login, password):
        self.login = login
        self.password = password
        self.client = Client()

    def __str__(self):
        return(repr(self))

    def __repr__(self):
        return("PrecisionMapper(login={})".format(self.login))

    def get_authenticity_token(self, url=_SIGNIN_URL):
        """ Returns an authenticity_token, mandatory for signing in """
        res = self.client._get(url=url, expected_status_code=200)
        soup = BeautifulSoup(res.text, _DEFAULT_BEAUTIFULSOUP_PARSER)
        selection = soup.select(_AUTHENTICITY_TOKEN_SELECTOR)
        try:
            authenticity_token = selection[0].get("content")
        except:
            raise ValueError(
                "authenticity_token not found in {} with {}\n{}".format(
                 _SIGNIN_URL, _AUTHENTICITY_TOKEN_SELECTOR, res.text))
        return authenticity_token

    def sign_in(self):
        authenticity_token = self.get_authenticity_token()
        post_data = {"utf8": "✓",
                     "authenticity_token": authenticity_token,
                     "return": "",
                     "login[username]": self.login,
                     "login[password]": self.password,
                     "commit": "Log In"}
        res = self.client._post(url=_SIGNIN_URL, post_data=post_data,
                                expected_status_code=302,
                                allow_redirects=False)
        # "allow_redirects = False" because we don't want to load the
        # <survey> page right now => better performance
        return res

    def get_surveys(self, url=_SURVEYS_URL):
        """ Function to get the surveys for the account """
        res = self.client._get(url=url, expected_status_code=200)
        soup = BeautifulSoup(res.text, _DEFAULT_BEAUTIFULSOUP_PARSER)
        surveys_soup = soup.select(_SURVEYS_SELECTOR)
        survey_list = []
        for survey_soup in surveys_soup:
            survey_name = _css_select(survey_soup, _SURVEY_NAME_SELECTOR)

            try:
                url = survey_soup.select(_SURVEY_URL_SELECTOR)[0]["href"]
            except:
                raise ValueError("Cannot get URL for the survey \
with css selector {}".format(_SURVEY_URL_SELECTOR))

            try:
                id = int(url.split("survey_id=")[1].split("&")[0])
            except:
                raise ValueError("Cannot extract id from URL {}".format(
                    url))

            survey_location = _css_select(survey_soup,
                                          _SURVEY_LOCATION_SELECTOR)
            try:
                survey_epoch = int(survey_soup.select(
                    _SURVEY_DATE_SELECTOR)[0]["epoch"])
                survey_date_obj = datetime.fromtimestamp(survey_epoch)
                survey_date = _datetime_object_to_rfc_date_str(survey_date_obj)
            except:
                raise ValueError("Cannot get date for the survey \
with css selector {}".format(_SURVEY_DATE_SELECTOR))

            survey_img_nb_and_size = survey_soup.select(
                _SURVEY_IMG_NB_AND_SIZE_SELECTOR)

            try:
                survey_img_nb = survey_img_nb_and_size[0].text
                survey_img_nb = int(survey_img_nb.split(" ")[0])
            except:
                raise ValueError("Cannot get or convert image number, \
survey_img_nb_and_size = {}".format(survey_img_nb_and_size))
            try:
                survey_size = survey_img_nb_and_size[1].text
            except:
                raise ValueError("Cannot get survey size, \
survey_img_nb_and_size = {}".format(survey_img_nb_and_size))

            sensor = _css_select(survey_soup, _SURVEY_SENSOR_SELECTOR)

            survey = Survey(
                id=id, name=survey_name, url=url,
                date=survey_date, location=survey_location,
                image_nb=survey_img_nb, size=survey_size, sensor=sensor)
            survey_list.append(survey)
        return survey_list

    def get_shared_surveys(self, url=_SHARED_SURVEYS_URL):
        return self.get_surveys(url=url)


def _css_select(soup, css_selector):
        """ Returns the content of the element pointed by the CSS selector,
        or an empty string if not found """
        selection = soup.select(css_selector)
        if len(selection) > 0:
            if hasattr(selection[0], 'text'):
                retour = selection[0].text.strip()
            else:
                retour = ""
        else:
            retour = ""
        return retour


def _datetime_object_to_rfc_date_str(datetime_obj):
    """ Returns a date string to the RFC 3339 standard """
    return datetime_obj.strftime(_RFC3339_DATE_FORMAT)


def _rfc_date_str_to_datetime_object(rfc_date_str):
    """ Returns a date string to the RFC 3339 standard """
    return datetime.strptime(rfc_date_str, _RFC3339_DATE_FORMAT)


def _datetime_object_to_short_date_str(datetime_obj):
    """ Returns a short date string """
    return datetime_obj.strftime(_SHORT_DATE_FORMAT)
