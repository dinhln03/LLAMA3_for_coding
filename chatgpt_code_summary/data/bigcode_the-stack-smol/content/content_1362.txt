import pytest
import requests

from directory_constants import expertise, sectors

from django.shortcuts import Http404
from django.urls import reverse

from core import helpers
import core.tests.helpers


@pytest.mark.parametrize('status_code,exception', (
    (400, requests.exceptions.HTTPError),
    (404, Http404),
    (500, requests.exceptions.HTTPError),
))
def test_handle_cms_response_error(status_code, exception):
    response = core.tests.helpers.create_response(status_code=status_code)
    with pytest.raises(exception):
        helpers.handle_cms_response(response)


def test_handle_cms_response_ok():
    response = core.tests.helpers.create_response(
        status_code=200, json_payload={'field': 'value'}
    )

    assert helpers.handle_cms_response(response) == {'field': 'value'}


@pytest.mark.parametrize('path,expect_code', (
    ('/', None),
    ('?language=pt', 'pt'),
    ('/?language=ar', 'ar'),
    ('/industries?language=es', 'es'),
    ('/industries/?language=zh-hans', 'zh-hans'),
    ('/industries/aerospace?language=de', 'de'),
    ('/industries/automotive/?language=fr', 'fr'),
    ('?lang=fr', 'fr'),
    ('?language=de&lang=de', 'de'),
    ('?lang=pt&language=es', 'es')
))
def test_get_language_from_querystring(path, expect_code, rf):
    url = reverse('index')
    request = rf.get(url + path)
    language_code = helpers.get_language_from_querystring(request)
    assert language_code == expect_code


def test_company_parser_serialize_for_template(retrieve_profile_data):
    company = helpers.CompanyParser(retrieve_profile_data)

    assert company.serialize_for_template() == {
        'address': '123 Fake Street, Fakeville, London, E14 6XK',
        'address_line_1': '123 Fake Street',
        'address_line_2': 'Fakeville',
        'country': 'GB',
        'date_of_creation': '02 March 2015',
        'description': 'Ecommerce website',
        'email_address': 'test@example.com',
        'email_full_name': 'Jeremy',
        'employees': '501-1,000',
        'expertise_countries': '',
        'expertise_industries': '',
        'expertise_languages': '',
        'expertise_products_services': {},
        'expertise_regions': '',
        'facebook_url': 'http://www.facebook.com',
        'has_expertise': False,
        'keywords': 'word1, word2',
        'linkedin_url': 'http://www.linkedin.com',
        'locality': 'London',
        'logo': 'nice.jpg',
        'mobile_number': '07506043448',
        'modified': '2016-11-23T11:21:10.977518Z',
        'name': 'Great company',
        'number': '01234567',
        'po_box': 'abc',
        'postal_code': 'E14 6XK',
        'postal_full_name': 'Jeremy',
        'sectors': 'Security',
        'slug': 'great-company',
        'summary': 'this is a short summary',
        'supplier_case_studies': [],
        'twitter_url': 'http://www.twitter.com',
        'verified_with_code': True,
        'website': 'http://example.com',
        'company_type': 'COMPANIES_HOUSE',
        'is_published_investment_support_directory': True,
        'is_published_find_a_supplier': True,
        'is_in_companies_house': True
    }


def test_company_parser_serialize_for_template_empty():
    company = helpers.CompanyParser({})

    assert company.serialize_for_template() == {}


def test_get_results_from_search_response_xss(retrieve_profile_data):
    response = core.tests.helpers.create_response(json_payload={
        'hits': {
            'total': 1,
            'hits': [
                {
                    '_source': retrieve_profile_data,
                    'highlight': {
                        'description': [
                            '<a onmouseover=javascript:func()>stuff</a>',
                            'to the max <em>wolf</em>.'
                        ]
                    }

                }
            ]
        }
    })

    formatted = helpers.get_results_from_search_response(response)

    assert formatted['results'][0]['highlight'] == (
        '&lt;a onmouseover=javascript:func()&gt;stuff&lt;/a&gt;...to the max '
        '<em>wolf</em>.'
    )


def test_get_filters_labels():
    filters = {
        'expertise_languages': ['aa'],
        'q': 'foo',
        'page': 5,
        'expertise_regions': ['NORTH_EAST'],
        'expertise_products_services_financial': [expertise.FINANCIAL[1]],
        'industries': [sectors.AEROSPACE, sectors.ADVANCED_MANUFACTURING],
        'expertise_products_services_human_resources': [
            'Employment and talent research'
        ],
    }

    expected = [
        'Afar',
        'North East',
        'Insurance',
        'Aerospace',
        'Advanced manufacturing',
        'Employment and talent research',
    ]

    assert helpers.get_filters_labels(filters) == expected
