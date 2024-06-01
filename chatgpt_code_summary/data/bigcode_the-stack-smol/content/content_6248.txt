# SPDX-License-Identifier: Apache-2.0
# Copyright 2021 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import pybind11_generics_tests.cpp as pyg_test
from pybind11_generics_tests.cpp import Animal, ListHolder, get_list

from .util import do_constructor_test, do_doc_test, do_error_test


class Dog(Animal):
    def __init__(self, name):
        Animal.__init__(self, name)

    def go(self, n_times):
        raise NotImplementedError("Not implemented")


class Husky(Dog):
    def __init__(self, name):
        Dog.__init__(self, name)

    def go(self, n_times):
        return "woof " * n_times


class ChildList(pyg_test.TestList):
    def __init__(self, vec1, vec2):
        pyg_test.TestList.__init__(self, vec1)
        self._list2 = vec2

    def get_data(self):
        return self._list2

    def get_data_base(self):
        return pyg_test.TestList.get_data(self)


test_data = [
    (pyg_test.TestList, []),
    (pyg_test.TestList, [1, 3, 5, 7, 6]),
    (pyg_test.TestList, [2, 4, 8]),
    (pyg_test.TestList, [13]),
]

fail_data = [
    (pyg_test.TestList, TypeError, [1, 2, 3.5]),
]

doc_data = [
    (pyg_test.TestList, "List[int]"),
]


@pytest.mark.parametrize(("cls", "data"), test_data)
def test_constructor(cls, data):
    """Check object is constructed properly."""
    do_constructor_test(cls, data)


@pytest.mark.parametrize(("cls", "err", "data"), fail_data)
def test_error(cls, err, data):
    """Check object errors when input has wrong data type."""
    do_error_test(cls, err, data)


@pytest.mark.parametrize(("cls", "type_str"), doc_data)
def test_doc(cls, type_str):
    """Check object has correct doc string."""
    do_doc_test(cls, type_str)


def test_inheritance():
    """Test inheritance behavior."""
    vec1 = [1, 2, 3, 4]
    vec2 = [5, 6, 7]

    obj = ChildList(vec1, vec2)

    assert obj.get_data() == vec2
    assert obj.get_data_base() == vec1
    assert get_list(obj) == vec1

    holder = ListHolder(obj)
    obj_ref = holder.get_obj_ref()
    obj_ptr = holder.get_obj_ptr()
    assert obj_ref is obj
    assert obj_ptr is obj
    assert isinstance(obj_ref, ChildList)


def test_virtual():
    """Test overriding virtual methods from python."""
    prime = Animal("Prime")
    dog = Dog("Doggo")
    lily = Husky("Lily")

    assert prime.go(1) == ""
    assert lily.go(2) == "woof woof "
    assert prime.command(2) == "Prime: "
    assert lily.command(3) == "Lily: woof woof woof "

    with pytest.raises(NotImplementedError):
        dog.go(3)
    with pytest.raises(NotImplementedError):
        dog.command(2)
