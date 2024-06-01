from flask_testing import TestCase
from flask import url_for
from core import app, db
import unittest
from core.models import FeatureRequest, Client, ProductArea
import datetime


class BaseTest(TestCase):
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    TESTING = True

    def create_app(self):
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
        return app

    def setUp(self):
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()


class HomepageTest(BaseTest):
    def test_homepage(self):
        "Make sure that homepage works fine"
        response = self.client.get(url_for("home_view"))
        assert b"Add a feature request:" in response.data
        assert b"List feature requests:" in response.data


class ListpageTest(BaseTest):
    def test_empty_listpage(self):
        "Make sure that empty list page works fine"
        response = self.client.get(url_for("home_view"))
        response = self.client.get(url_for("feature_requests_view"))
        assert b"No feature requests found." in response.data

    def test_non_empty_listpage(self):
        "Also that it can display multiple entries"
        fr = FeatureRequest(
            title="Title",
            description="Desc",
            client=None,
            client_priority=1,
            target_date=datetime.date(2018, 1, 1),
            product_area=None,
        )
        db.session.add(fr)
        fr2 = FeatureRequest(
            title="Title",
            description="Desc",
            client=None,
            client_priority=1,
            target_date=datetime.date(2018, 1, 1),
            product_area=None,
        )
        db.session.add(fr2)
        db.session.commit()
        response = self.client.get(url_for("feature_requests_view"))
        assert response.data.count(b"Update") == 2
        assert response.data.count(b"Delete") == 2
        assert (
            url_for("feature_requests_update", feature_request_id=1).encode()
            in response.data
        )
        assert (
            url_for("feature_requests_delete", feature_request_id=1).encode()
            in response.data
        )


class AddOtherObjectsMixin:
    "A reusable mixin that adds a client and a product area to the db"

    def add_other_objects(self):
        self.cl = Client("C1")
        db.session.add(self.cl)
        self.pa = ProductArea("PA1")
        db.session.add(self.pa)
        db.session.commit()


class CreatepageTest(AddOtherObjectsMixin, BaseTest):
    def test_createpage(self):
        "Make sure that the create page works"
        response = self.client.get(url_for("feature_requests_create"))
        assert b"Add Feature Request" in response.data
        assert b"<form method='POST'>" in response.data
        assert b"form-group has-error" not in response.data

    def test_createpage_error(self):
        "The create page should return with error when post data is missing"
        response = self.client.post(
            url_for("feature_requests_create"),
            data=dict(
                title="Title",
                description="Desc",
                client=None,
                client_priority=1,
                target_date=datetime.date(2018, 1, 1),
                product_area=None,
            ),
        )

        assert b"form-group has-error" in response.data
        assert b"<form method='POST'>" in response.data
        assert response.status == "200 OK"

    def test_createpage_success(self):
        "The create page should return a 302 FOUND redirect when an entry is submitted"
        client = Client("C1")
        db.session.add(client)
        product_area = ProductArea("PA1")
        db.session.add(product_area)
        db.session.commit()
        response = self.client.post(
            url_for("feature_requests_create"),
            data=dict(
                title="Title",
                description="Desc",
                client=client.id,
                client_priority=1,
                target_date=datetime.date(2018, 1, 1),
                product_area=product_area.id,
            ),
        )
        assert response.status == "302 FOUND"

    def test_createpage_success_flash(self):
        """The create page should display the proper flash message when an object is
        created"""
        self.add_other_objects()
        response = self.client.post(
            url_for("feature_requests_create"),
            data=dict(
                title="Title",
                description="Desc",
                client=self.cl.id,
                client_priority=1,
                target_date=datetime.date(2018, 1, 1),
                product_area=self.pa.id,
            ),
            follow_redirects=True,
        )
        assert response.status == "200 OK"
        assert b"Feature request created!" in response.data
        assert response.data.count(b"Update") == 1
        assert response.data.count(b"Delete") == 1
        assert self.cl.name.encode() in response.data
        assert self.pa.name.encode() in response.data

    def test_createpage_change_priorities(self):
        """The create page should change the priorities of the other objects when a
        new one has the same priority and client"""
        self.add_other_objects()
        fr = FeatureRequest(
            title="Title",
            description="Desc",
            client=self.cl,
            client_priority=1,
            target_date=datetime.date(2018, 1, 1),
            product_area=self.pa,
        )
        db.session.add(fr)
        db.session.commit()
        assert FeatureRequest.query.filter_by(id=fr.id).first().client_priority == 1
        response = self.client.post(
            url_for("feature_requests_create"),
            data=dict(
                title="Title",
                description="Desc",
                client=self.cl.id,
                client_priority=1,
                target_date=datetime.date(2018, 1, 1),
                product_area=self.pa.id,
            ),
            follow_redirects=True,
        )
        assert response.status == "200 OK"
        assert FeatureRequest.query.filter_by(id=fr.id).first().client_priority == 2


class UpdatepageTest(AddOtherObjectsMixin, BaseTest):
    def add_feature_request(self):
        "A reusable method for this class"
        self.fr = FeatureRequest(
            title="Title",
            description="Desc",
            client=None,
            client_priority=1,
            target_date=datetime.date(2018, 1, 1),
            product_area=None,
        )
        db.session.add(self.fr)
        db.session.commit()

    def test_updatepage_not_found(self):
        "Make sure that the update page returs 404 when the obj is not found"
        response = self.client.get(
            url_for("feature_requests_update", feature_request_id=1232)
        )
        assert response.status == "404 NOT FOUND"

    def test_updatepage_ok(self):
        "Make sure that the update page is displayed properly along with the object"
        self.add_feature_request()
        response = self.client.get(
            url_for("feature_requests_update", feature_request_id=self.fr.id)
        )
        assert "Edit Feature Request: {0}".format(self.fr.id).encode() in response.data
        assert b"<form method='POST'>" in response.data
        assert b"form-group has-error" not in response.data
        assert self.fr.title.encode() in response.data
        assert self.fr.description.encode() in response.data

    def test_updatepage_error(self):
        "The createpage should return an error when data is missing"
        self.add_feature_request()
        response = self.client.post(
            url_for("feature_requests_update", feature_request_id=self.fr.id),
            data=dict(
                title="Title",
                description="Desc",
                client=None,
                client_priority=1,
                target_date=datetime.date(2018, 1, 1),
                product_area=None,
            ),
        )

        assert b"form-group has-error" in response.data
        assert b"<form method='POST'>" in response.data
        assert response.status == "200 OK"

    def test_createpage_success(self):
        "The createpage should properly update the object"
        self.add_feature_request()
        self.add_other_objects()
        newtitle = "The new title"
        response = self.client.post(
            url_for("feature_requests_update", feature_request_id=self.fr.id),
            data=dict(
                title=newtitle,
                description="Desc",
                client=self.cl.id,
                client_priority=1,
                target_date=datetime.date(2018, 1, 1),
                product_area=self.pa.id,
            ),
        )
        assert response.status == "302 FOUND"
        assert FeatureRequest.query.filter_by(id=self.fr.id).first().title == newtitle

    def test_updatepage_success_flash(self):
        """Make sure that the flash message is displayed correctly and we are
        redirected to the list view"""
        self.add_feature_request()
        self.add_other_objects()
        response = self.client.post(
            url_for("feature_requests_update", feature_request_id=self.fr.id),
            data=dict(
                title="Title",
                description="Desc",
                client=self.cl.id,
                client_priority=1,
                target_date=datetime.date(2018, 1, 1),
                product_area=self.pa.id,
            ),
            follow_redirects=True,
        )
        assert response.status == "200 OK"
        assert b"Feature request updated!" in response.data
        assert response.data.count(b"Update") == 1
        assert response.data.count(b"Delete") == 1
        assert self.cl.name.encode() in response.data
        assert self.pa.name.encode() in response.data

    def test_updatepage_change_priorities(self):
        "The updatepage should also update the client priorities"
        self.add_other_objects()
        fr = FeatureRequest(
            title="Title",
            description="Desc",
            client=self.cl,
            client_priority=1,
            target_date=datetime.date(2018, 1, 1),
            product_area=self.pa,
        )
        db.session.add(fr)
        fr2 = FeatureRequest(
            title="Title",
            description="Desc",
            client=self.cl,
            client_priority=2,
            target_date=datetime.date(2018, 1, 1),
            product_area=self.pa,
        )
        db.session.add(fr2)
        db.session.commit()
        assert FeatureRequest.query.filter_by(id=fr.id).first().client_priority == 1
        assert FeatureRequest.query.filter_by(id=fr2.id).first().client_priority == 2
        response = self.client.post(
            url_for("feature_requests_update", feature_request_id=2),
            data=dict(
                title="Title",
                description="Desc",
                client=self.cl.id,
                client_priority=1,
                target_date=datetime.date(2018, 1, 1),
                product_area=self.pa.id,
            ),
            follow_redirects=True,
        )
        assert response.status == "200 OK"
        assert FeatureRequest.query.filter_by(id=fr.id).first().client_priority == 2
        assert FeatureRequest.query.filter_by(id=fr2.id).first().client_priority == 1


class DeletepageTest(BaseTest):
    def add_feature_request(self):
        "A reusable method for this class"
        self.fr = FeatureRequest(
            title="Title",
            description="Desc",
            client=None,
            client_priority=1,
            target_date=datetime.date(2018, 1, 1),
            product_area=None,
        )
        db.session.add(self.fr)
        db.session.commit()

    def test_deletepdatepage_only_post(self):
        "Make sure that the delete page returns 405 when requested with get"
        response = self.client.get(
            url_for("feature_requests_delete", feature_request_id=1232)
        )
        assert response.status == "405 METHOD NOT ALLOWED"

    def test_deletepdatepage_not_found(self):
        "Make sure that the delete page returs 404 when the obj is not found"
        response = self.client.post(
            url_for("feature_requests_delete", feature_request_id=1232)
        )
        assert response.status == "404 NOT FOUND"

    def test_deletepage_ok(self):
        "Make sure that the delete page deletes the obj"
        self.add_feature_request()
        assert db.session.query(FeatureRequest.query.filter().exists()).scalar() is True
        response = self.client.post(
            url_for("feature_requests_delete", feature_request_id=self.fr.id)
        )
        assert (
            db.session.query(FeatureRequest.query.filter().exists()).scalar() is False
        )
        assert response.status == "302 FOUND"

    def test_deletepage_flash_message(self):
        "Make sure that the delete page shows the proper flash message"
        self.add_feature_request()
        response = self.client.post(
            url_for("feature_requests_delete", feature_request_id=self.fr.id),
            follow_redirects=True,
        )
        assert response.status == "200 OK"
        assert b"Feature request deleted!" in response.data
        assert response.data.count(b"Update") == 0
        assert response.data.count(b"Delete") == 0


if __name__ == "__main__":
    unittest.main()
