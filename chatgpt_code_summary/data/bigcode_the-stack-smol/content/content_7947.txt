"""
Serializer fields for django_hal

"""


from collections import OrderedDict

from django.utils.http import urlencode
from rest_framework import serializers

from .utils import reverse


class LinksField(serializers.DictField):
    """HAL-style _links field.

    Parameters
    ----------
    *args : tuple
        A tuple representing the relation name, and arguments to
        reverse the url.  Example: `(name, urlpattern, {'pk', 'pk'})`.

        name : str
            The string used to identify the url in the final output.
        urlpattern : str
            A named urlpattern.
        kwargs : dict
          The kwargs to pass (with the urlpattern) to `reverse`.

          This is a dict where the key is the url kwarg, and the value is the
          attribute to lookup on the instance.  So, `{'user', 'pk'}` would
          translate to `{'user': getattr(instance, 'pk')}`.

    Example
    -------

        MySerializer(serializers.Serializer):
            _links = LinksField(
                ('self', 'namespace:view-name', {'pk': 'pk'})
            )

        # Outputs:
        #
        #     {
        #       '_links': {
        #         'self': 'https://.../my-resource/34'
        #       }
        #     }

      A shorthand syntax is available to reduce the repetitiveness of
      `{'pk': 'pk'}`, when both the kwarg and the instance attribute name
      are the same.

          ('ref', 'urlpattern', 'pk')

      is equivalent to

          ('ref', 'urlpattern', {'pk': 'pk'})

      In a full example that looks like:

          MySerializer(serializers.Serializer):
              _links = LinksField(
                  ('self', 'namespace:view-name', 'pk')
              )

          # Outputs:
          #
          #     {
          #         '_links': {
          #             'self': { 'href': 'https://.../my-resource/34' }
          #         }
          #     }

    """

    def __init__(self, *links):
        super(LinksField, self).__init__(read_only=True)
        self.links = links

    def to_representation(self, instance):
        """Return an ordered dictionary of HAL-style links."""
        request = self.context.get('request')
        ret = OrderedDict()
        for link in self.links:
            name = link[0]
            ret[name] = self.to_link(request, instance, *link[1:])
        return ret

    def get_attribute(self, instance, *args, **kwargs):
        """Return the whole instance, instead of looking up an attribute value.

        Implementation note: We do this because `Serializer.to_representation`
        builds the list of serializer fields with something like:

            for field in serializer_fields:
              field.to_representation(field.get_attribute(instance))

        Since we need the instance in `to_representation` so we can query arbitrary
        attributes on it to build urls, we simply have to return the instance here.
        """
        return instance

    def to_link(self, request, instance, urlpattern, kwargs=None,
                query_kwargs=None):
        """Return an absolute url for the given urlpattern."""
        if query_kwargs:
            query_kwargs = {k: getattr(instance, v) for k, v in query_kwargs.items()}
        if not kwargs:
            url = reverse(urlpattern, request=request)
            if not query_kwargs:
                return {'href': url}
            return {'href': '%s?%s' % (url, urlencode(query_kwargs))}

        if isinstance(kwargs, basestring):
            # `(ref, urlpattern, string)` where `string` is equivalent to
            # `{string: string}`
            url = reverse(urlpattern,
                          kwargs={kwargs: getattr(instance, kwargs)},
                          request=request)
            if not query_kwargs:
                return {'href': url}
            return {'href': '%s?%s' % (url, urlencode(query_kwargs))}

        reverse_kwargs = {}
        if kwargs:
            for k, v in kwargs.items():
                reverse_kwargs[k] = getattr(instance, v)
        try:
            url = reverse(urlpattern, kwargs=reverse_kwargs, request=request)
            if not query_kwargs:
                return {'href': url}
            return {'href': '%s?%s' % (url, urlencode(query_kwargs))}
        except NoReverseMatch:
            return None


class QueryField(serializers.HyperlinkedIdentityField):
    """Return the query url that lists related objects in a reverse relation.

    Example
    -------

    .. code:: python

        class Book:
            title = CharField()
            author = ForeignKey(Author)

        class Author:
            name = CharField()

        url('books/query/author/<pk>', ..., name='book-query-by-author')

        class AuthorSerializer:
            name = CharField()
            books = QueryField('book-query-by-author')

        >>> nick = Author(name='Nick').save()
        >>> book1 = Book(title='Part 1', author=nick)
        >>> book2 = Book(title='Part 2', author=nick)
        >>> AuthorSerializer(nick)
        {
            'name': 'Nick',
            'books': '../books/query/author/1',
        }

    Raises
    ------
    django.*.NoReverseMatch
        if the `view_name` and `lookup_field` attributes are not configured to
        correctly match the URL conf.

    """
    lookup_field = 'pk'

    def __init__(self, view_name, url_kwarg=None, query_kwarg=None, **kwargs):
        assert url_kwarg is not None or query_kwarg is not None, 'The `url_kwarg` argument is required.'  # noqa

        kwargs['lookup_field'] = kwargs.get('lookup_field', self.lookup_field)
        self.url_kwarg = url_kwarg
        self.query_kwarg = query_kwarg

        super(QueryField, self).__init__(view_name, **kwargs)

    def get_url(self, obj, view_name, request, response_format):
        lookup_value = getattr(obj, self.lookup_field)

        if self.url_kwarg:
            kwargs = {self.url_kwarg: lookup_value}
            return reverse(view_name,
                           kwargs=kwargs,
                           request=request,
                           format=response_format)

        url = reverse(view_name,
                      request=request,
                      format=response_format)
        query_kwargs = {self.query_kwarg: lookup_value}
        return u'%s?%s' % (url, urlencode(query_kwargs))


