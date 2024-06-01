from rest_framework.test import APIRequestFactory
from rest_framework import status
from django.test import TestCase
from django.urls import reverse
from ..models import User
from ..serializer import UserSerializer
from ..views import UserViewSet
import ipapi


class UsersApiRootTestCase(TestCase):

    def test_api_root_should_reply_200(self):
        """ GET /api/v1/ should return an hyperlink to the users view and return a successful status 200 OK.
        """
        request = APIRequestFactory().get("/api/v1/")
        user_list_view = UserViewSet.as_view({"get": "list"})
        response = user_list_view(request)
        self.assertEqual(status.HTTP_200_OK, response.status_code)


class UsersApiTestCase(TestCase):
    """ Factorize the tests setup to use a pool of existing users. """

    def setUp(self):
        self.factory = APIRequestFactory()
        self.users = [
            User.objects.create(
                first_name="Riri", last_name="Duck", email="riri.duck@ricardo.ch", password="dummy"),
            User.objects.create(
                first_name="Fifi", last_name="Duck", email="fifi.duck@ricardo.ch", password="dummy"),
            User.objects.create(
                first_name="Loulou", last_name="Duck", email="loulou.duck@ricardo.ch", password="dummy")
            ]


class GetAllUsersTest(UsersApiTestCase):
    """ Test GET /api/v1/users """


    def test_list_all_users_should_retrieve_all_users_and_reply_200(self):
        """ GET /api/v1/users should return all the users (or empty if no users found)
            and return a successful status 200 OK.
        """
        users = User.objects.all().order_by("id")
        request = self.factory.get(reverse("v1:user-list"))
        serializer = UserSerializer(users, many=True, context={'request': request})
        
        user_list_view = UserViewSet.as_view({"get": "list"})
        response = user_list_view(request)

        self.assertEqual(len(self.users), len(response.data["results"]))
        self.assertEqual(serializer.data, response.data["results"])
        self.assertEqual(status.HTTP_200_OK, response.status_code)


class GetSingleUserTest(UsersApiTestCase):
    """ Test GET /api/v1/users/:id """


    def test_get_user_when_id_valid_should_retrieve_user_and_reply_200(self):
        riri = User.objects.create(
            first_name="Riri", last_name="Duck", email="riri.duck@ricardo.ch", password="dummy")
        user = User.objects.get(pk=riri.pk)
        request = self.factory.get(reverse("v1:user-detail", kwargs={"pk": riri.pk}))
        serializer = UserSerializer(user, context={'request': request})
        
        user_detail_view = UserViewSet.as_view({"get": "retrieve"})
        response = user_detail_view(request, pk=riri.pk)

        self.assertEqual(serializer.data, response.data)
        self.assertEqual(status.HTTP_200_OK, response.status_code)


    def test_get_user_when_id_invalid_should_reply_404(self):
        request = self.factory.get(reverse("v1:user-detail", kwargs={"pk": 100}))
        user_detail_view = UserViewSet.as_view({"get": "retrieve"})
        response = user_detail_view(request, pk=100)
        self.assertEqual(status.HTTP_404_NOT_FOUND, response.status_code)


class CreateNewUserTest(UsersApiTestCase):
    """ Test POST /api/v1/users
        Override 'REMOTE_ADDR' to set IP address to Switzerland or another country for testing purpose.
    """


    def test_post_user_when_from_Switzerland_and_data_valid_should_create_user_and_reply_201(self):
        initial_users_count = len(self.users)
        valid_data = {
            "first_name": "Casper",
            "last_name": "Canterville",
            "email": "c@sper.com",
            "password": "dummy",
        }
        request = self.factory.post(
            reverse("v1:user-list"),
            data=valid_data,
            REMOTE_ADDR='2.16.92.0'
            )
        user_detail_view = UserViewSet.as_view({"post": "create"})
        response = user_detail_view(request)

        self.assertEqual(status.HTTP_201_CREATED, response.status_code)
        new_users_count = User.objects.count()
        self.assertEqual(initial_users_count+1, new_users_count)


    def test_post_user_when_id_invalid_should_not_create_user_and_reply_400(self):
        initial_users_count = len(self.users)
        invalid_data = {
            "first_name": "Casper",
            "last_name": "Canterville",
            "email": "",
            "password": "dummy",
        }
        request = self.factory.post(
            reverse("v1:user-list"),
            data=invalid_data,
            REMOTE_ADDR='2.16.92.0'
            )
        user_detail_view = UserViewSet.as_view({"post": "create"})
        response = user_detail_view(request)

        self.assertEqual(status.HTTP_400_BAD_REQUEST, response.status_code)
        users_count = User.objects.count()
        self.assertEqual(initial_users_count, users_count)


    def test_post_user_when_data_valid_but_email_already_used_should_not_create_user_and_reply_400(self):
        initial_users_count = len(self.users)
        valid_data_with_used_email = {
            "first_name": "Casper",
            "last_name": "Canterville",
            "email": "riri.duck@ricardo.ch",
            "password": "dummy",
        }
        request = self.factory.post(
            reverse("v1:user-list"),
            data=valid_data_with_used_email,
            REMOTE_ADDR='2.16.92.0'
            )
        user_detail_view = UserViewSet.as_view({"post": "create"})
        response = user_detail_view(request)

        self.assertEqual(status.HTTP_400_BAD_REQUEST, response.status_code)
        new_users_count = User.objects.count()
        self.assertEqual(initial_users_count, new_users_count)


    def test_post_user_when_IP_not_in_Switzerland_should_not_create_user_and_reply_403(self):
        initial_users_count = len(self.users)
        valid_data = {
            "first_name": "Casper",
            "last_name": "Canterville",
            "email": "c@sper.com",
            "password": "dummy",
        }
        request = self.factory.post(
            reverse("v1:user-list"),
            data=valid_data,
            REMOTE_ADDR='2.16.8.0' # Spain
            )

        user_detail_view = UserViewSet.as_view({"post": "create"})
        response = user_detail_view(request)

        self.assertEqual(status.HTTP_403_FORBIDDEN, response.status_code)
        self.assertTrue(len(response.data['detail']) > 0)
        users_count = User.objects.count()
        self.assertEqual(initial_users_count, users_count)


class UpdateSinglUserTest(UsersApiTestCase):
    """ Test PUT|PATCH /api/v1/user/:id """


    def test_patch_user_when_id_valid_should_patch_user_and_reply_200(self):
        riri = User.objects.create(
            first_name="Riri", last_name="Duck", email="riri.duck@ricardo.ch", password="dummy")
        
        request = self.factory.patch(
            reverse("v1:user-detail", kwargs={"pk": riri.pk}),
            data={"email": "riri@ricardo.ch"}
            )
        user_detail_view = UserViewSet.as_view({"patch": "partial_update"})
        response = user_detail_view(request, pk=riri.pk)

        self.assertEqual(status.HTTP_200_OK, response.status_code)


    def test_patch_user_when_id_invalid_should_not_patch_user_and_reply_404(self):
        riri = User.objects.create(
            first_name="Riri", last_name="Duck", email="riri.duck@ricardo.ch", password="dummy")
        
        request = self.factory.patch(
            reverse("v1:user-detail", kwargs={"pk": 100}),
            data={"email": "riri@ricardo.ch"}
            )
        user_detail_view = UserViewSet.as_view({"patch": "partial_update"})
        response = user_detail_view(request, pk=100)

        self.assertEqual(status.HTTP_404_NOT_FOUND, response.status_code)


    def test_put_when_invalid_data_should_not_update_user_and_reply_400(self):
        riri = User.objects.create(
            first_name="Riri", last_name="Duck", email="riri.duck@ricardo.ch", password="dummy")
        invalid_payload = {
            "first_name": "",
            "last_name": "Duck",
            "email": "riri.duck@ricardo.ch"
        }
        request = self.factory.put(
            reverse("v1:user-detail", kwargs={"pk": riri.pk}),
            data=invalid_payload
            )
        user_detail_view = UserViewSet.as_view({"put": "update"})
        response = user_detail_view(request, pk=riri.pk)
        self.assertEqual(status.HTTP_400_BAD_REQUEST, response.status_code)


class DeleteSinglePuppyTest(UsersApiTestCase):
    """ Test DELETE /api/v1/user/:id """


    def test_delete_user_when_id_valid_should_delete_user_and_reply_204(self):
        initial_users_count = len(self.users)
        user_to_delete = self.users[0]
        
        request = self.factory.delete(reverse("v1:user-detail", kwargs={"pk": user_to_delete.pk}))
        user_detail_view = UserViewSet.as_view({"delete": "destroy"})
        response = user_detail_view(request, pk=user_to_delete.pk)
        
        self.assertEqual(status.HTTP_204_NO_CONTENT, response.status_code)
        new_users_count = User.objects.count()
        self.assertEqual(initial_users_count-1, new_users_count)


    def test_delete_user_when_id_invalid_should_reply_404(self):
        request = self.factory.delete(reverse("v1:user-detail", kwargs={"pk": 100}))
        user_detail_view = UserViewSet.as_view({"delete": "destroy"})
        response = user_detail_view(request, pk=100)
        self.assertEqual(status.HTTP_404_NOT_FOUND, response.status_code)

