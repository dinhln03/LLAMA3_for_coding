# uncompyle6 version 3.7.4
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.9 (tags/v3.7.9:13c94747c7, Aug 17 2020, 18:58:18) [MSC v.1900 64 bit (AMD64)]
# Embedded file name: T:\InGame\Gameplay\Scripts\Server\objects\gardening\gardening_commands.py
# Compiled at: 2017-11-18 00:09:10
# Size of source mod 2**32: 1465 bytes
from objects.components import types
from objects.components.types import GARDENING_COMPONENT
from objects.gardening.gardening_component_fruit import GardeningFruitComponent
import services, sims4.commands

@sims4.commands.Command('gardening.cleanup_gardening_objects')
def cleanup_gardening_objects(_connection=None):
    for obj in services.object_manager().get_all_objects_with_component_gen(GARDENING_COMPONENT):
        gardening_component = obj.get_component(types.GARDENING_COMPONENT)
        if not isinstance(gardening_component, GardeningFruitComponent):
            continue
        if obj.parent is None:
            obj.is_in_inventory() or obj.is_on_active_lot() or sims4.commands.output('Destroyed object {} on open street was found without a parent at position {}, parent_type {}.'.format(obj, obj.position, obj.parent_type), _connection)
            obj.destroy(source=obj, cause='Fruit/Flower with no parent on open street')

    sims4.commands.output('Gardening cleanup complete', _connection)
    return True