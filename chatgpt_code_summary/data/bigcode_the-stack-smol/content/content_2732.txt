# -*- coding: utf-8 -*-

"""
slicr.resources.links
~~~~~~~~~~~~~~~~~~~~~
Slicr link resource.

:copyright: © 2018
"""

from flask import current_app
from flask_restful import Resource
from webargs import fields
from webargs.flaskparser import use_args

from slicr.models import Link, LinkSchema
from slicr.utils import convert_args


link_args = {
    'url': fields.Str(required=True),
    'domain_id': fields.Int(missing=None)
}


# pylint: disable=R0201
class LinkResource(Resource):
    """Link resource."""

    endpoints = ['/links', '/links/<int:link_id>']
    schema = LinkSchema()

    def get(self, link_id):
        """Get link resource.

        .. :quickref: Link collection.

        **Example request**:

        .. sourcecode:: http

            GET /links/1 HTTP/1.1
            Host: example.com
            Accept: application/json, text/javascript

        **Example response**:

        .. sourcecode:: http

            HTTP/1.1 200 OK
            Vary: Accept
            Content-Type: text/javascript

            {
                "data": {
                    "clicks": 0,
                    "created": "2018-08-21T19:13:34.157470+00:00",
                    "short_link": "b",
                    "updated": null,
                    "url": "https://www.google.com"
                },
                "id": 1,
                "type": "links",
                "url": "/links"
            }

        :jsonparam string url: url for which to create short link.
        :reqheader Accept: The response content type depends on
            :mailheader:`Accept` header
        :reqheader Authorization: Optional authentication token.
        :resheader Content-Type: this depends on :mailheader:`Accept`
            header of request
        :statuscode 201: Link created
        """

        link = Link.query.filter_by(id=link_id).first()

        link_data, errors = self.schema.dump(link)

        if errors:
            current_app.logger.warning(errors)

        response_out = {
            'id': link.id,
            'data': link_data,
            'url': '/links',
            'type': 'link'
        }

        return response_out, 200

    @use_args(link_args)
    def post(self, args):
        """Create shortened link.

        .. :quickref: Link collection.

        **Example request**:

        .. sourcecode:: http

            POST /links HTTP/1.1
            Host: example.com
            Accept: application/json, text/javascript

            {
                "url": "https://www.google.com"
            }

        **Example response**:

        .. sourcecode:: http

            HTTP/1.1 201 OK
            Vary: Accept
            Content-Type: text/javascript

            {
                "data": {
                    "clicks": 0,
                    "created": "2018-08-21T19:13:34.157470+00:00",
                    "short_link": "b",
                    "updated": null,
                    "url": "https://www.google.com"
                },
                "id": 1,
                "type": "links",
                "url": "/links"
            }

        :jsonparam string url: url for which to create short link.
        :reqheader Accept: The response content type depends on
            :mailheader:`Accept` header
        :reqheader Authorization: Optional authentication token.
        :resheader Content-Type: this depends on :mailheader:`Accept`
            header of request
        :statuscode 201: Link created
        """

        args = convert_args(args)

        link = Link(
            url=args.url,
            domain_id=args.domain_id,
            salt=int(current_app.config.get('ENCODER_SALT'))
        ).save()

        link_data, errors = self.schema.dump(link)

        if errors:
            current_app.logger.warning(errors)

        response_out = {
            'id': link.id,
            'data': link_data,
            'url': '/links',
            'type': 'link'
        }

        return response_out, 201
