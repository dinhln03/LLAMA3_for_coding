# Auto-generated at 2021-09-27T17:12:31.553030+08:00
# from: Justice Iam Service (4.1.0)

# Copyright (c) 2018 - 2021 AccelByte Inc. All Rights Reserved.
# This is licensed software from AccelByte Inc, for limitations
# and restrictions contact your company contract manager.

# pylint: disable=duplicate-code
# pylint: disable=line-too-long
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# pylint: disable=too-many-public-methods
# pylint: disable=too-many-return-statements
# pylint: disable=too-many-statements
# pylint: disable=unused-import

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

from ....core import Model

from ..models.accountcommon_pagination_v3 import AccountcommonPaginationV3
from ..models.model_user_response_v3 import ModelUserResponseV3


class ModelGetUsersResponseWithPaginationV3(Model):
    """Model get users response with pagination V3

    Properties:
        data: (data) REQUIRED List[ModelUserResponseV3]

        paging: (paging) REQUIRED AccountcommonPaginationV3
    """

    # region fields

    data: List[ModelUserResponseV3]                                                                # REQUIRED
    paging: AccountcommonPaginationV3                                                              # REQUIRED

    # endregion fields

    # region with_x methods

    def with_data(self, value: List[ModelUserResponseV3]) -> ModelGetUsersResponseWithPaginationV3:
        self.data = value
        return self

    def with_paging(self, value: AccountcommonPaginationV3) -> ModelGetUsersResponseWithPaginationV3:
        self.paging = value
        return self

    # endregion with_x methods

    # region to methods

    def to_dict(self, include_empty: bool = False) -> dict:
        result = {}
        if hasattr(self, "data") and self.data:
            result["data"] = [i0.to_dict(include_empty=include_empty) for i0 in self.data]
        elif include_empty:
            result["data"] = []
        if hasattr(self, "paging") and self.paging:
            result["paging"] = self.paging.to_dict(include_empty=include_empty)
        elif include_empty:
            result["paging"] = AccountcommonPaginationV3()
        return result

    # endregion to methods

    # region static methods

    @classmethod
    def create(
        cls,
        data: List[ModelUserResponseV3],
        paging: AccountcommonPaginationV3,
    ) -> ModelGetUsersResponseWithPaginationV3:
        instance = cls()
        instance.data = data
        instance.paging = paging
        return instance

    @classmethod
    def create_from_dict(cls, dict_: dict, include_empty: bool = False) -> ModelGetUsersResponseWithPaginationV3:
        instance = cls()
        if not dict_:
            return instance
        if "data" in dict_ and dict_["data"] is not None:
            instance.data = [ModelUserResponseV3.create_from_dict(i0, include_empty=include_empty) for i0 in dict_["data"]]
        elif include_empty:
            instance.data = []
        if "paging" in dict_ and dict_["paging"] is not None:
            instance.paging = AccountcommonPaginationV3.create_from_dict(dict_["paging"], include_empty=include_empty)
        elif include_empty:
            instance.paging = AccountcommonPaginationV3()
        return instance

    @staticmethod
    def get_field_info() -> Dict[str, str]:
        return {
            "data": "data",
            "paging": "paging",
        }

    # endregion static methods
