from behave import *
import requests

from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from host.models import Event


use_step_matcher("re")


# @given("that I am a registered host of privilege walk events and want to create questions and answer choices for the event")
# def step_impl(context):
#     context.username = "12thMan"
#     context.password = "SomePassword123"
#     context.first_name = "12th"
#     context.last_name = "Man"
#     context.email = "twelve@testtamu.edu"

#     usr = User.objects.create_user(
#         context.username,
#         context.email,
#         context.password
#     )
#     usr.first_name = context.first_name
#     usr.last_name = context.last_name
#     usr.save()

#     registered_user = User.objects.filter(username="12thMan")

#     assert len(registered_user) == 1

#     user_auth_token, _ = Token.objects.get_or_create(user=usr)
#     context.key = user_auth_token.key

#     data = {
#         "name": "New year event"
#     }
#     headers = {
#         'Authorization':'Token '+ context.key
#     }

#     resp = requests.post(context.test.live_server_url + "/host/events/create/", data, headers=headers)
#     context.event_api_response_data = resp.json()
#     context.eventId = context.event_api_response_data["id"]


# @when("I make an API call to create questions API with my correct username, questions, answer choices and correct eventid")
# def step_impl(context):
#     data = {
#         "event_id": context.eventId,
#         "title": "The question's title goes here",
#         "choices": [
#             {
#                 "description": "Pizza",
#                 "value": 1
#             },
#             {
#                 "description": "Ice Cream",
#                 "value": 2
#             },
#             {
#                 "description": "Salt Water",
#                 "value": -1
#             }
#         ]
#     }
#     headers = {
#         'Authorization':'Token '+ context.key
#     }

#     resp = requests.post(context.test.live_server_url + "/host/qa/create/", data, headers=headers)
#     assert resp.status_code >= 200 and resp.status_code < 300

#     context.api_response_data = resp.json()

# @then("I expect the response that gives the status and id of the created question")
# def step_impl(context):
#     assert context.api_response_data["status"] == "created"
#     assert context.api_response_data["id"] != ""



# @given("that I am a registered host of privilege walk and wants to create questions but with wrong eventid")
# def step_impl(context):
#     context.username = "12thMan"
#     context.password = "SomePassword123"
#     context.first_name = "12th"
#     context.last_name = "Man"
#     context.email = "twelve@testtamu.edu"

#     usr = User.objects.create_user(
#         context.username,
#         context.email,
#         context.password
#     )
#     usr.first_name = context.first_name
#     usr.last_name = context.last_name
#     usr.save()

#     registered_user = User.objects.filter(username="12thMan")

#     assert len(registered_user) == 1

#     user_auth_token, _ = Token.objects.get_or_create(user=usr)
#     context.key = user_auth_token.key

#     data = {
#         "name": "New year event"
#     }
#     headers = {
#         'Authorization':'Token '+ context.key
#     }

#     resp = requests.post(context.test.live_server_url + "/host/events/create/", data, headers=headers)
#     context.event_api_response_data = resp.json()
#     context.eventId = context.event_api_response_data["id"]


# @when("I make an API call to create questions API with my username, questions, answer choices and wrong event id")
# def step_impl(context):
#     data = {
#         "event_id": 12,
#         "title": "Are you under 20?",
#         "choices": [
#             {
#                 "description": "Yes",
#                 "value": "1"
#             },
#             {
#                 "description": "No",
#                 "value": "-1"
#             }
#         ]
#     }
#     headers = {
#         'Authorization':'Token '+ context.key
#     }

#     resp = requests.post(context.test.live_server_url + "/host/qa/create/", data, headers=headers)
#     assert resp.status_code >= 500

#     context.api_response_data = resp.json()


# @then("I expect the response that says questions cannot be created as event id doesn't exist")
# def step_impl(context):
#     pass


# @given("that I am a registered host of privilege walk and wants to create questions but without giving eventid")
# def step_impl(context):
#     context.username = "12thMan"


# @when("I make an API call to create questions API with my username, questions, answer choices and without event id")
# def step_impl(context):
#     data = {
#         "title": "Are you under 20?",
#         "choices": [
#             {
#                 "description": "Yes",
#                 "value": "1"
#             },
#             {
#                 "description": "No",
#                 "value": "-1"
#             }
#         ]
#     }
#     headers = {
#         'Authorization':'Token '+ context.key
#     }
#     resp = requests.post(context.test.live_server_url + "/host/qa/create/", data, headers=headers)
#     assert resp.status_code >= 500

#     context.api_response_data = resp.json()


# @then("I expect the response that says questions cannot be created as event id is missing")
# def step_impl(context):
#     pass


@given("that I am a registered host of privilege walk events and want to create questions but forgets to give username")
def step_impl(context):
    context.username = "11thMan"


@when("I make an API call to create questions API with missing username in request")
def step_impl(context):
    data = {
        "title": "Are you under 20?",
        "choices": [
            {
                "description": "Yes",
                "value": "1"
            },
            {
                "description": "No",
                "value": "-1"
            }
        ]
    }

    resp = requests.post(context.test.live_server_url + "/host/events/create/", data)
    assert resp.status_code >= 400 and resp.status_code < 500

    context.api_response_data = resp.json()


@then("I expect the response that says questions cannot be created and username is required in request")
def step_impl(context):
    assert context.api_response_data["detail"] == "Authentication credentials were not provided."

