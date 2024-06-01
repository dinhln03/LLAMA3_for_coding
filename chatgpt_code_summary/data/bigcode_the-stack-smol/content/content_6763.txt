from datetime import datetime

import ujson
from django.db import transaction
from django.test import TestCase
from math import pi

from convertable_model.models import BarFooFooModel
from convertable_model.models import FooBarModel
from convertable_model.models import FooModel, BarModel
from convertable_model.models import MixedFooBarModel


class JSONConvertibleModelTest(TestCase):

    def tearDown(self) -> None:
        FooModel.objects.all().delete()

    def _test_equality(self, model: any, *, pk: int, result: dict):
        obj = model.objects.get(pk=pk)
        obj_data = ujson.loads(obj.to_json())
        obj_expected_result = self.get_expected_result(result)
        self.assertEqual(obj_data, obj_expected_result)

    @staticmethod
    def get_expected_result(data: iter) -> dict:
        return ujson.loads(ujson.dumps(data))

    @staticmethod
    def normal_foo_results() -> dict:
        return {
            1: {'IntegerField': None,
                'My_name': 'val1',
                'foo3': pi,
                'foo4': datetime.utcnow(),
                'id': 1
                },
            2: {'IntegerField': 10,
                'My_name': 'val2',
                'foo3': pi,
                'foo4': datetime.utcnow(),
                'id': 2
                },
            3: {'IntegerField': 9999,
                'My_name': 'val1',
                'foo3': pi,
                'foo4': datetime.utcnow(),
                'id': 3
                },
        }

    @staticmethod
    def create_foo_objects():
        FooModel.objects.create(foo2='val1', foo3=1.56)
        FooModel.objects.create(foo1=10, foo2='val2', foo3=2.34)
        FooModel.objects.create(foo1=9999, foo2='val1', foo3=7**0.5)

    def test_normal_foo_model(self):
        results = self.normal_foo_results()
        self.create_foo_objects()
        for i in range(1, 4):
            self._test_equality(FooModel, pk=i, result=results[i])

    @staticmethod
    def normal_bar_results() -> dict:
        return {
            1: {'bar1': None,
                'bar2': 'Hello World',
                'bar4': datetime.utcnow()
                },
            2: {'bar1': None,
                'bar2': 'Hello World',
                'bar4': datetime.utcnow()
                },
            3: {'bar1': 2,
                'bar2': 'Hello World',
                'bar4': datetime.utcnow()
                },
        }

    @staticmethod
    def create_bar_objects():
        BarModel.objects.create(bar3=0.1234)
        BarModel.objects.create(bar2='Some random string', bar3=0.1004)
        BarModel.objects.create(bar1=2, bar2='Another random string', bar3=0.44)

    def test_normal_bar_model(self):
        self.create_bar_objects()
        results = self.normal_bar_results()
        for i in range(1, 4):
            self._test_equality(BarModel, pk=i, result=results[i])
        # test json array
        json_array = ujson.loads(BarModel.to_json_array(BarModel.objects.all()))
        self.assertEqual(json_array,
                         self.get_expected_result(results.values()))

    @staticmethod
    def foo_foreignkey_results() -> dict:
        return {
            1: {
                'FooModel': 'val1: 0',
                'foobar2': 'CALL ME SNAKE',
                'foobar3': 24.45,
            },
            2: {
                'FooModel': 'None',
                'foobar2': 'CALL ME TIGER',
                'foobar3': 23.22,
            },
        }

    def test_with_foreignkey(self):
        results = self.foo_foreignkey_results()
        with transaction.atomic():
            self.create_foo_objects()
            FooBarModel.objects.create(foobar1=FooModel.objects.get(pk=1),
                                       foobar2='call me snake',
                                       foobar3=24.45)
            FooBarModel.objects.create(foobar2='call me tiger',
                                       foobar3=23.22)
        for i in range(1, 3):
            self._test_equality(FooBarModel, pk=i, result=results[i])

    def result_many_to_many(self) -> dict:
        return {
            'id': 1,
            'foofoo': [
                self.normal_foo_results()[1],
                self.normal_foo_results()[2]
            ],
            'barbar': [
                self.normal_bar_results()[2],
                self.normal_bar_results()[3],
            ],
            'barfoo': 'BarFooFooModel: 1'
        }

    def test_many_to_many_and_foreignkey(self):
        with transaction.atomic():
            self.create_foo_objects()
            self.create_bar_objects()
            BarFooFooModel.objects.create(barfoo1=BarModel.objects.get(pk=1))

        mixed_obj = MixedFooBarModel.objects.create(barfoo=BarFooFooModel.objects.get(pk=1))
        mixed_obj.foofoo.add(FooModel.objects.get(pk=1))
        mixed_obj.foofoo.add(FooModel.objects.get(pk=2))
        mixed_obj.barbar.add(BarModel.objects.get(pk=2))
        mixed_obj.barbar.add(BarModel.objects.get(pk=3))

        obj = MixedFooBarModel.objects.get(pk=1)
        obj_data = ujson.loads(obj.to_json())
        obj_expected_result = self.get_expected_result(self.result_many_to_many())
        self.assertEqual(obj_data, obj_expected_result)
