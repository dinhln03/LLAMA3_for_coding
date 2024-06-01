import logging
import os
from pathlib import Path
import typing
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

import services.pv_simulator.constants as constants
from services.pv_simulator.main_loop import MainLoop
from services.pv_simulator.mq_receiver import MQReceiver, MQReceiverFactory
from services.pv_simulator.pv_power_value_calculator import PVPowerValueCalculator
from services.pv_simulator.typing_custom_protocols import (
    MQReceiverProtocol,
    PVPowerValueCalculatorProtocol
)
from services.pv_simulator import utils

current_dir_path: Path = Path(__file__).parent.absolute()
load_dotenv(dotenv_path=f"{current_dir_path}/.env")


def get_test_modules_names() -> typing.List[str]:
    from services.pv_simulator.tests.unit import constants_for_tests
    return constants_for_tests.TESTS_MODULES


def get_mq_receiver(callback: typing.Callable[[], bool]) -> MQReceiverProtocol:
    return MQReceiverFactory.get_mq_receiver(constants.MQ_RECEIVER_TYPE, check_if_must_exit=callback)


def get_pv_power_value_calculator() -> PVPowerValueCalculatorProtocol:
    return PVPowerValueCalculator(constants.MINUTES_DATA_SET, constants.PV_POWER_VALUES_DATA_SET)


def main(sys_argv: typing.List[str]) -> None:
    """PV simulator execution entry point.

    Parameters
    ----------
    sys_argv : list
        contains the list of arguments passed to the CLI during its execution. The first argument contains the
        executed script name.
    """

    main_logger: typing.Optional[logging.Logger] = None
    try:
        must_exit_after_24h = os.getenv("MUST_EXIT_AFTER_24H", "0")
        must_exit_after_24h = \
            True if must_exit_after_24h.isdecimal() and int(must_exit_after_24h) == 1 else False

        main_logger = utils.initialize_loggers(current_dir_path)
        main_loop: MainLoop = MainLoop(constants.LOGGER_NAME,
                                       constants.RESULTS_LOGGER_NAME,
                                       current_dir_path,
                                       must_exit_after_24h,
                                       get_mq_receiver,
                                       get_pv_power_value_calculator,
                                       tests_modules_names_provider=get_test_modules_names)

        main_loop.handle_arguments(sys_argv)
    except KeyboardInterrupt:
        if main_logger is not None:
            main_logger.exception("Required to abort:")
        else:
            import traceback
            traceback.print_exc()
    except Exception:
        if main_logger is not None:
            main_logger.exception("Error:")
        else:
            import traceback
            traceback.print_exc()
