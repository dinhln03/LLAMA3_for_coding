from django import forms
from django.http import QueryDict
from django.forms.formsets import formset_factory
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from datetime import date
import itertools
import re
from fields import SubmitButtonField, SubmitButtonWidget


class Filter(object):

    __metaclass__ = ABCMeta
    _order = itertools.count()

    form_field_class = None
    form_field_widget = None
    filter_state_names = ['%s', ]
    filter_field = ''

    def __init__(self,
                 default=None,
                 required=False,
                 label=None,
                 form_field_class=None,
                 form_field_widget=None,
                 filter_set=False,
                 filter_field=None):
        self.default = default
        self.required = required
        self.label = label
        self.form_field_class = form_field_class or self.form_field_class
        self.form_field_widget = form_field_widget or self.form_field_widget
        self.order = Filter._order.next()
        self.filter_set = filter_set
        self.filter_field = filter_field or self.filter_field

    def get_form_field(self):
        """
        Returns an instance of the form field class, used for constructing the
        filter form for a report.
        """
        return self.form_field_class(required=(self.required and not self.filter_set),
                                     widget=self.form_field_widget,
                                     label=self.label)

    def get_form_class(self, name, index=0, postfix="Form"):
        form_class_name = "%s%s" % (type(self).__name__, postfix)
        form_class_dict = {name: self.get_form_field()}
        return type(form_class_name, (forms.Form,), form_class_dict)

    def clean_data(self, name, raw_data):
        form = self.get_form_class(name)(data=raw_data)
        return form.cleaned_data[name] if form.is_valid() else None

    def get_data(self, name, data):
        """
        To get the data for this filter given the filter sets, we instantiate
        the form with the data, validate it, and return the cleaned data.
        """
        cleaned_data = self.clean_data(name, data)
        return cleaned_data if cleaned_data else self.default

    def get_data_set(self, name, data):
        """
        This horribly ugly little function is in charge of returning a list of
        data entries, given filter states, for a filter set. It does the same
        thing as get_data, but for every item in a filter set, returning the
        results in a list.
        """
        # If we're not really a set, just return a 1-element list with the data
        if not self.filter_set:
            return [self.get_data(name, data)]

        # Get the deletion field name and index
        delete = data.get('delete', None)
        delete_index = None
        if delete:
            n, i = delete.split('.')
            if n == name:
                delete_index = int(i) + 1

        # Zip together all the lists of filter state values. This gives us a
        # list of tuples of filter state fields. Ugly but necessary in case we
        # have a filter which generates a MultiValueField (aka,
        # NumericComparisonFilter). Exclude elements which have been deleted.
        filter_state_names = self.filter_state_names[:]
        filter_state_list = [data.getlist(state_name % name, []) for state_name in filter_state_names]
        filter_states = zip(*filter_state_list)

        # Loop over every filter state tuple, converting it to a mini filter-
        # -state dict. Clean it, and store the cleaned data in a list
        data_set = []
        for i in range(len(filter_states)):

            # If this index is getting deleted, don't add it
            if i == delete_index:
                continue

            # Get the dict of states for this filter set element
            state = filter_states[i]
            filter_dict = {}
            for i in range(0, len(filter_state_names)):
                filter_dict.update({filter_state_names[i] % name: state[i]})

            # Clean and validate the set instance data. If it validates, store
            # it in the state list.
            cleaned_data = self.clean_data(name, filter_dict)
            if cleaned_data:
                data_elem = cleaned_data
                data_set.append(data_elem)

        # Return the list of states
        return data_set

    def get_filter_state_from_data(self, name, data):
        """
        Another nasty little bit. This one (if not overridden) takes some
        data and encodes it, using the filter state names, to be a valid
        filter_state which would return the original data if passed to get_data

        TODO: Make sure this actually works for stuff other than
              NumericComparisonFilter

        TODO: Add good comments :P
        """
        if len(self.filter_state_names) > 1:
            if not (hasattr(data, '__iter__') and len(self.filter_state_names) == len(data)):
                raise Exception()
            state = {}
            for i in range(0, len(data)):
                state.update({self.filter_state_names[i] % name: data[i]})
            return state
        else:
            return {self.filter_state_names[0] % name: data}

    def apply_filter(self, queryset, data):
        filterspec = {self.filter_field: data}
        return queryset.filter(**filterspec)

    def apply_filter_set(self, queryset, data_set):

        # Apply the filter to the queryset based on each entry in the data set
        for data in data_set:
            queryset = self.apply_filter(queryset, data)
        return queryset


class Report(object):

    __metaclass__ = ABCMeta

    headers = None
    footers = None
    title = None

    def __init__(self, filter_states={}):
        """
        filter_state will be a querydict with keys corresponding to the names
        of the filter members on this report object.
        """
        if isinstance(filter_states, QueryDict):
            self.filter_states = filter_states
        else:
            self.filter_states = QueryDict('', mutable=True)
            self.filter_states.update(filter_states)
        self.title = self.title or self.get_title_from_class_name()

    def __getattribute__(self, name):
        """
        When getting a filter attribute, looks for the corresponding filter
        state and returns that instead of the filter object. If none is found,
        looks for the default value on the filter object. If that's not found
        either, then returns none.
        """
        # Perform the normal __getattribute__ call
        attr = object.__getattribute__(self, name)

        # If it's a filter attribute...
        if issubclass(type(attr), Filter):

            # If we have a filter state for this filter, convert it to the type
            # of data for this filter.
            if not attr.filter_set:
                return attr.get_data(name, self.filter_states)
            else:
                return attr.get_data_set(name, self.filter_states)

        # This isn't a filter, just return the attribute
        return attr

    def get_title_from_class_name(self):
        """
        Split the class name into words, delimited by capitals.
        """
        words = re.split(r'([A-Z])', self.__class__.__name__)[1:]
        words = [words[i] + words[i+1] for i in range(0, len(words) - 1, 2)]
        return ' '.join(words)

    def get_filter(self, name):
        """
        Perform the normal __getattribute__ call,
        and return it if it's a filter
        """
        attr = object.__getattribute__(self, name)
        return attr if issubclass(type(attr), Filter) else None

    def get_filters(self):
        """
        Return a list of all the names and attributes on this report instance
        which have a base class of Filter.
        """
        filters = []
        for name in dir(self):
            attr = object.__getattribute__(self, name)
            if issubclass(type(attr), Filter):
                filters.append((name, attr))
        return sorted(filters, key=lambda attr: attr[1].order)

    def get_filter_forms(self):
        for name, attr in self.get_filters():

            # If it is a filter set, loop through the existing list of data
            # in the filter states, if there are any. For each of these, make a
            # sub-form which includes a "delete" checkbox
            if attr.filter_set:

                # Get the new-set element form
                form = attr.get_form_class(name)()
                form.name = name
                yield form

                # Yield all the existing form elements
                data_set = attr.get_data_set(name, self.filter_states)
                for i in range(len(data_set)):
                    data = data_set[i]
                    state = attr.get_filter_state_from_data(name, data)

                    # Generate and yield a form containing the filter's field,
                    # as well as a deleting submit field to mark deletions
                    form = attr.get_form_class(
                        name=name,
                        postfix="FormSetElem"
                        )(data=state)
                    form.delete = {
                        'filter': name,
                        'index': i}

                    form.name = name
                    yield form

            # If it ain't a filter set, just get it's form class and render it
            # with the filter state data
            else:
                form = attr.get_form_class(name)(data=self.filter_states)
                form.name = name
                yield form

    def get_title(self):
        return self.title

    def get_headers(self):
        return self.headers

    def get_footers(self):
        return self.footers

    def apply_filter(self, queryset, name):
        f = self.get_filter(name)

        # If it's not a filterset, just get the regular data and apply it
        if not f.filter_set:
            data = f.get_data(name, self.filter_states)
            if data:
                return f.apply_filter(queryset, data)

        # Otherwise, get the full data set and apply it
        else:
            data_set = f.get_data_set(name, self.filter_states)
            if len(data_set) > 0:
                return f.apply_filter_set(queryset, data_set)

        # If we weren't able to apply the filter, return the raw queryset
        return queryset

    def apply_filters(self, queryset, names=None, excludes=[]):
        for name, f in self.get_filters():

            # Only apply this filter if it's selected
            if name in excludes or (names and name not in names):
                continue

            # Apply this filter
            queryset = self.apply_filter(queryset, name)

        # Return the filtered queryset
        return queryset

    def get_queryset(self):
        return []

    def get_row(self, item):
        """
        This can return a list for simple data that doesn't need special
        template rendering, or a dict for more complex data where individual
        fields will need to be rendered specially.
        """
        return []

    def get_rows(self):
        rows = []
        for item in self.get_queryset():
            row = self.get_row(item)
            if row:
                rows.append(row)
        return rows

    def get_count(self):
        return self.get_queryset().count()

    def get_table(self):
        return [[cell for cell in row] for row in self.get_rows()]

    @staticmethod
    def encode_filter_states(data):
        """
        Converts a normal POST querydict to the filterstate data,
        to be stored in the url
        """
        #data = QueryDict(data.urlencode(), mutable=True)
        return data

    @staticmethod
    def decode_filter_states(data):
        """
        Opposite of encode_filter_states
        """
        return data


class Row(object):
    def __init__(self, list, attrs=None):
        self.list = list
        if attrs:
            for name, value in attrs.iteritems():
                setattr(self, name, value)

    def __iter__(self):
        return self.list.__iter__()
