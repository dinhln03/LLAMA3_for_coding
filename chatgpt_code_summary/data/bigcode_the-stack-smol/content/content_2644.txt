import pytest

from contoso import get_company_name, get_company_address

def test_get_company_name():
    assert get_company_name() == "Contoso"

def test_get_company_address():
    assert get_company_address() == "Contosostrasse 1, Zurich, Switzerland"