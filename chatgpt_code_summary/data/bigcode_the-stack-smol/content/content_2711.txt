import atexit
from .MecanumRover_MotorDriver import MecanumRover_MotorDriver
import traitlets
from traitlets.config.configurable import Configurable


class Motor(Configurable):

    value = traitlets.Float()

    # config
    alpha = traitlets.Float(default_value=1.0).tag(config=True)
    beta = traitlets.Float(default_value=0.0).tag(config=True)

    def __init__(self, driver, channel, *args, **kwargs):
        super(Motor, self).__init__(*args, **kwargs)  # initializes traitlets

        self._driver = driver
        self._motor = self._driver.getMotor(channel)
        atexit.register(self._release)

    @traitlets.observe('value')
    def _observe_value(self, change):
        self._write_value(change['new'])

    def _write_value(self, value):
        """Sets motor value between [-1, 1]"""
        # ジョイスティック等の値ブレ対策
        if abs(value) <= 0.05:
            value = 0.0

        #モータの目標速度(mm/s)に変換。※最高1300mm/s
        mapped_value = int(1300.0 * (self.alpha * value + self.beta))
        speed = min(max(mapped_value, -1300), 1300)
        self._motor.setSpeed(speed)

    def _release(self):
        """Stops motor by releasing control"""
        self._motor.setSpeed(0)
