# coding: utf-8

"""
    Factern API
"""


import pprint
import re  # noqa: F401

import six
import importlib




parent_name = "BaseResponse"
def get_parent():
    # Lazy importing of parent means that loading the classes happens
    # in the correct order.
    if get_parent.cache is None:
        parent_fname = "factern_client.com.factern.model.%s" % re.sub("([a-z])([A-Z])", "\\1_\\2", "BaseResponse").lower()
        parent = importlib.import_module(parent_fname).BaseResponse
        get_parent.cache = parent
    return get_parent.cache
get_parent.cache = None


class CreateEntityResponse(get_parent()):

    @staticmethod
    def get_parent():
        return get_parent()

    @staticmethod
    def compute_parent_updates():
        pass

        get_parent().compute_parent_updates()

        CreateEntityResponse.swagger_types.update(get_parent().swagger_types)
        CreateEntityResponse.attribute_map.update(get_parent().attribute_map)


    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'description': 'str',
        'name': 'str'
    }

    attribute_map = {
        'description': 'description',
        'name': 'name'
    }

    def __init__(self, **kwargs):  # noqa: E501
        """CreateEntityResponse - a model defined in Swagger"""  # noqa: E501
        self.compute_parent_updates()
        for k in kwargs:
            if k not in self.swagger_types:
                raise ValueError("CreateEntityResponse got unexpected argument '%s'" % k)
        get_parent().__init__(self, **kwargs)

        self._description = None
        self._name = None


        if "description" in kwargs:
            self.description = kwargs["description"]
        if "name" in kwargs:
            self.name = kwargs["name"]

    @property
    def description(self):
        """Gets the description of this CreateEntityResponse.  # noqa: E501


        :return: The description of this CreateEntityResponse.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """Sets the description of this CreateEntityResponse.


        :param description: The description of this CreateEntityResponse.  # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def name(self):
        """Gets the name of this CreateEntityResponse.  # noqa: E501


        :return: The name of this CreateEntityResponse.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this CreateEntityResponse.


        :param name: The name of this CreateEntityResponse.  # noqa: E501
        :type: str
        """

        self._name = name

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, CreateEntityResponse):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
