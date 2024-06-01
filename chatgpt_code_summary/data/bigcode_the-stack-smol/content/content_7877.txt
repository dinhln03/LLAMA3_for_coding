import functools
import typing

from aws_cdk import core

from cdk_resources.utils import (
    app_context,
    ALLOWED_ENVIRONMENTS,
    get_environment,
)


__all__ = ["ResourceStack", "register_stacks"]


class ResourceStack(core.Stack):
    """ """

    EXISTING_RESOURCES = None
    RESOURCES = None

    def __init__(
        self, scope: core.App, stack_id: str, **kwargs
    ) -> None:
        super().__init__(scope, stack_id, **kwargs)
        # Update Context
        app_context.update(app=scope, current_stack=self)
        if self.is_valid_environment is False:
            raise Exception(
                f"`{get_environment()}` must be a valid environment allowed "
                f"values {ALLOWED_ENVIRONMENTS}"
            )

        # Existing resources
        for resources in self.EXISTING_RESOURCES or []:
            resource_name, Resource, resource_attrs = (
                self.get_resource_name(resources[0]),
                resources[1],
                (resources[2] if len(resources) == 3 else {}),
            )
            setattr(
                self,
                resource_name,
                Resource(
                    scope=self,
                    construct_id=resources[0],
                    **resource_attrs,
                ),
            )

        # Own Resources
        for resources in self.RESOURCES or []:
            resource_name, Resource, resource_attrs = (
                self.get_resource_name(resources[0]),
                resources[1],
                (resources[2] if len(resources) == 3 else {}),
            )
            resource = Resource(scope=self, construct_id=resource_name)
            setattr(self, resource_name, resource)

    @staticmethod
    def get_resource_name(value: typing.Union[str, typing.Callable]) -> str:
        return value() if hasattr(value, "__call__") else value

    @property
    @functools.lru_cache(maxsize=None)
    def is_valid_environment(self) -> bool:
        if len(ALLOWED_ENVIRONMENTS) == 0:
            return True
        environment = get_environment()
        return environment is not None and environment in ALLOWED_ENVIRONMENTS


def register_stacks(
    app: core.App, aws_env: core.Environment, stacks: list
) -> None:
    # Create Stacks
    for stack in stacks:
        stack_id, stack_class, stack_kwargs = (
            stack[0],
            stack[1],
            (stack[2] if len(stack) == 3 else {}),
        )
        stack_class(app, stack_id, env=aws_env, **stack_kwargs)
