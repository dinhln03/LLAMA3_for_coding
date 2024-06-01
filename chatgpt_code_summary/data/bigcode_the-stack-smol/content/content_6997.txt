
from enum import Enum
from sensors.ph_sensor import PhSensor

class SensorType(Enum):
    Undefined = 0,
    TDS = 1,
    Ph = 2,
    WaterLevel = 3,
    Voltage = 4

class TriggerType(Enum):
    Undefined = 0,
    NutrientDose = 1,
    PhDose = 2,
    WaterPumpCutout = 3

class Hydriot():
    sensors = dict()
    triggers = dict()

    def set_sensor(self, sensor_type, sensor):
        self.sensors[sensor_type] = sensor

    def get_sensor(self, sensor_type):
        if sensor_type not in self.sensors:
            return None
        return self.sensors[sensor_type]

    def set_trigger(self, trigger_type, trigger):
        self.triggers[trigger_type] = trigger

    def get_trigger(self, trigger_type):
        if trigger_type not in self.triggers:
            return None
        return self.triggers[trigger_type]

    @property
    def ph_sensor(self):
        return None if SensorType.Ph not in self.sensors else self.sensors[SensorType.Ph]

    @property
    def tds_sensor(self):
        return None if SensorType.TDS not in self.sensors else self.sensors[SensorType.TDS]

    @property
    def water_level_sensor(self):
        return None if SensorType.WaterLevel not in self.sensors else self.sensors[SensorType.WaterLevel]
    
    @property
    def voltage_sensor(self):
        return None if SensorType.Voltage not in self.sensors else self.sensors[SensorType.Voltage]

    @property
    def nutrient_trigger(self):
        return None if TriggerType.NutrientDose not in self.triggers else self.triggers[TriggerType.NutrientDose]
    
    @property
    def ph_trigger(self):
        return None if TriggerType.PhDose not in self.triggers else self.triggers[TriggerType.PhDose]

    @property
    def water_pump_trigger(self):
        return None if TriggerType.WaterPumpCutout not in self.triggers else self.triggers[TriggerType.WaterPumpCutout]