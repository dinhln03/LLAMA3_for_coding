from typing import Dict

import pytest

from {{cookiecutter.project_slug}}.tests.assertions import assert_field_error
from {{cookiecutter.project_slug}}.users.api.serializers import RegisterSerializer

pytestmark = pytest.mark.django_db


@pytest.fixture
def user_json(user_json: Dict) -> Dict:
    user_json["password1"] = user_json["password"]
    user_json["password2"] = user_json["password"]

    for field in list(user_json):
        if field not in ["name", "email", "password1", "password2"]:
            user_json.pop(field)

    return user_json


class TestRegisterSerializer:
    def test_get_cleaned_data_returns_dict_with_correct_fields(
        self, user_json: Dict
    ) -> None:
        serializer = RegisterSerializer(data=user_json)

        assert serializer.is_valid()
        cleaned_data = serializer.get_cleaned_data()
        assert len(cleaned_data) == 3
        for field in ["name", "password1", "email"]:
            assert cleaned_data[field] == user_json[field]

    def test_get_cleaned_data_returns_empty_string_for_name_when_name_not_provided(
        self, user_json: Dict
    ) -> None:
        user_json.pop("name")
        serializer = RegisterSerializer(data=user_json)

        assert serializer.is_valid()
        cleaned_data = serializer.get_cleaned_data()
        assert cleaned_data["name"] == ""

    @pytest.mark.parametrize(
        "field",
        ["email", "password1", "password2"],
        ids=["email", "password1", "password2"],
    )
    def test_fields_are_required(self, user_json: Dict, field: str) -> None:
        user_json.pop(field)
        serializer = RegisterSerializer(data=user_json)

        assert_field_error(serializer, field)
