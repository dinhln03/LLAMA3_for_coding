from copy import deepcopy

from simple_api.django_object.actions import DetailAction, ListAction, CreateAction, UpdateAction, DeleteAction
from simple_api.django_object.datatypes import create_associated_list_type
from simple_api.django_object.filters import generate_filters
from simple_api.django_object.converter import determine_simple_api_fields
from simple_api.django_object.utils import get_pk_field
from simple_api.object.object import Object, ObjectMeta
from simple_api.object.registry import object_storage
from simple_api.django_object.registry import model_django_object_storage
from simple_api.utils import ClassStub


class DjangoObjectMeta(type):
    base_class = "simple_api.django_object.django_object.DjangoObject"

    def __new__(mcs, name, bases, attrs, **kwargs):
        cls = super().__new__(mcs, name, bases, attrs)
        if kwargs.get("skip", False) or object_storage.key_for_class(attrs["__module__"], name) == mcs.base_class:
            return cls

        object_stub = ClassStub(name=cls.__name__, bases=(Object,))

        # set the module of the generated Object class to match the module of the user class
        object_stub.add_attr("__module__", cls.__module__)
        assert cls.model is not None, "`model` must be set."

        # if the class is meant to resolve relations, store it for the particular model
        if cls.class_for_related:
            model_django_object_storage.store(cls.model, cls)

        cls.pk_field_name, cls.pk_field = get_pk_field(cls.model)
        object_stub.add_attr("pk_field", cls.pk_field_name)

        # make sure the primary key is included, otherwise `ModelObjectAction`s would just not work
        if cls.only_fields and cls.pk_field_name not in cls.only_fields:
            cls.only_fields = cls.only_fields + (cls.pk_field_name,)
        elif cls.exclude_fields and cls.pk_field_name in cls.exclude_fields:
            cls.exclude_fields = (f for f in cls.exclude_fields if f != cls.pk_field_name)

        fields, input_fields, output_fields, field_validators = determine_simple_api_fields(
            cls.model,
            cls.only_fields, cls.exclude_fields,
            cls.custom_fields, cls.input_custom_fields, cls.output_custom_fields,
        )

        for f in input_fields:
            assert f not in fields, "Redefinition of `{}` field.".format(f)
        cls.in_fields = {**fields, **input_fields}

        for f in output_fields:
            assert f not in fields, "Redefinition of `{}` field.".format(f)
        cls.out_fields = {**fields, **output_fields}

        object_stub.add_attr("fields", fields)
        object_stub.add_attr("input_fields", input_fields)
        object_stub.add_attr("output_fields", output_fields)

        # create filters and List type for potential listing actions
        cls.filter_type = ObjectMeta("{}Filters".format(cls.__name__), (Object,), {"fields": generate_filters(cls)})
        object_stub.add_attr("filter_type", cls.filter_type)
        create_associated_list_type(cls)

        actions = {}

        if cls.detail_action is not None:
            actions["detail"] = deepcopy(cls.detail_action)
        if cls.list_action is not None:
            actions["list"] = deepcopy(cls.list_action)
        if cls.create_action is not None:
            actions["create"] = deepcopy(cls.create_action)
        if cls.update_action is not None:
            actions["update"] = deepcopy(cls.update_action)
        if cls.delete_action is not None:
            actions["delete"] = deepcopy(cls.delete_action)

        actions.update(cls.custom_actions)

        converted_actions = {}
        for action_name, action in actions.items():
            action.set_parent_class(cls)
            action.set_name(action_name)
            converted_actions[action_name] = action.to_action()

        object_stub.add_attr("actions", converted_actions)

        if cls.field_difficulty_scores is not None:
            object_stub.add_attr("field_difficulty_scores", cls.field_difficulty_scores)

        cls._object = object_stub.build(ObjectMeta)

        return cls


class DjangoObject(metaclass=DjangoObjectMeta):
    model = None
    auto_pk = True
    class_for_related = True

    only_fields = None
    exclude_fields = None

    custom_fields = {}
    input_custom_fields = {}
    output_custom_fields = {}
    field_difficulty_scores = {}

    detail_action = DetailAction()
    list_action = ListAction()
    create_action = CreateAction()
    update_action = UpdateAction()
    delete_action = DeleteAction()
    custom_actions = {}

    @classmethod
    def to_object(cls):
        return cls._object
