"""
Auto-generated class for JobResult
"""
from .EnumJobResultName import EnumJobResultName
from .EnumJobResultState import EnumJobResultState

from . import client_support


class JobResult(object):
    """
    auto-generated. don't touch.
    """

    @staticmethod
    def create(data, id, level, name, startTime, state, stderr, stdout):
        """
        :type data: str
        :type id: str
        :type level: int
        :type name: EnumJobResultName
        :type startTime: int
        :type state: EnumJobResultState
        :type stderr: str
        :type stdout: str
        :rtype: JobResult
        """

        return JobResult(
            data=data,
            id=id,
            level=level,
            name=name,
            startTime=startTime,
            state=state,
            stderr=stderr,
            stdout=stdout,
        )

    def __init__(self, json=None, **kwargs):
        if json is None and not kwargs:
            raise ValueError('No data or kwargs present')

        class_name = 'JobResult'
        create_error = '{cls}: unable to create {prop} from value: {val}: {err}'
        required_error = '{cls}: missing required property {prop}'

        data = json or kwargs

        property_name = 'data'
        val = data.get(property_name)
        if val is not None:
            datatypes = [str]
            try:
                self.data = client_support.val_factory(val, datatypes)
            except ValueError as err:
                raise ValueError(create_error.format(cls=class_name, prop=property_name, val=val, err=err))
        else:
            raise ValueError(required_error.format(cls=class_name, prop=property_name))

        property_name = 'id'
        val = data.get(property_name)
        if val is not None:
            datatypes = [str]
            try:
                self.id = client_support.val_factory(val, datatypes)
            except ValueError as err:
                raise ValueError(create_error.format(cls=class_name, prop=property_name, val=val, err=err))
        else:
            raise ValueError(required_error.format(cls=class_name, prop=property_name))

        property_name = 'level'
        val = data.get(property_name)
        if val is not None:
            datatypes = [int]
            try:
                self.level = client_support.val_factory(val, datatypes)
            except ValueError as err:
                raise ValueError(create_error.format(cls=class_name, prop=property_name, val=val, err=err))
        else:
            raise ValueError(required_error.format(cls=class_name, prop=property_name))

        property_name = 'name'
        val = data.get(property_name)
        if val is not None:
            datatypes = [EnumJobResultName]
            try:
                self.name = client_support.val_factory(val, datatypes)
            except ValueError as err:
                raise ValueError(create_error.format(cls=class_name, prop=property_name, val=val, err=err))
        else:
            raise ValueError(required_error.format(cls=class_name, prop=property_name))

        property_name = 'startTime'
        val = data.get(property_name)
        if val is not None:
            datatypes = [int]
            try:
                self.startTime = client_support.val_factory(val, datatypes)
            except ValueError as err:
                raise ValueError(create_error.format(cls=class_name, prop=property_name, val=val, err=err))
        else:
            raise ValueError(required_error.format(cls=class_name, prop=property_name))

        property_name = 'state'
        val = data.get(property_name)
        if val is not None:
            datatypes = [EnumJobResultState]
            try:
                self.state = client_support.val_factory(val, datatypes)
            except ValueError as err:
                raise ValueError(create_error.format(cls=class_name, prop=property_name, val=val, err=err))
        else:
            raise ValueError(required_error.format(cls=class_name, prop=property_name))

        property_name = 'stderr'
        val = data.get(property_name)
        if val is not None:
            datatypes = [str]
            try:
                self.stderr = client_support.val_factory(val, datatypes)
            except ValueError as err:
                raise ValueError(create_error.format(cls=class_name, prop=property_name, val=val, err=err))
        else:
            raise ValueError(required_error.format(cls=class_name, prop=property_name))

        property_name = 'stdout'
        val = data.get(property_name)
        if val is not None:
            datatypes = [str]
            try:
                self.stdout = client_support.val_factory(val, datatypes)
            except ValueError as err:
                raise ValueError(create_error.format(cls=class_name, prop=property_name, val=val, err=err))
        else:
            raise ValueError(required_error.format(cls=class_name, prop=property_name))

    def __str__(self):
        return self.as_json(indent=4)

    def as_json(self, indent=0):
        return client_support.to_json(self, indent=indent)

    def as_dict(self):
        return client_support.to_dict(self)
