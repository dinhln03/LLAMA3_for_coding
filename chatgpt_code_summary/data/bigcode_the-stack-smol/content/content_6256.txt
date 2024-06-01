from rqalpha.interface import AbstractMod
from rqalpha.apis import *
from rqalpha.events import EVENT
from collections import defaultdict
import datetime
import os


class ForceClose(AbstractMod):
    def __init__(self):
        self._log_dir = None
        self._log_file = defaultdict(lambda: None)
        self._force_close_time = []

    def start_up(self, env, mod_config):
        for timespan in mod_config.force_close_time:
            v = timespan.split('-')
            assert len(v) == 2, "%s invalid" % mod_config.force_close_time
            start_time_v = v[0].split(':')
            end_time_v = v[1].split(':')
            assert len(start_time_v) == 2, "%s invalid" % mod_config.force_close_time
            assert len(end_time_v) == 2, "%s invalid" % mod_config.force_close_time
            self._force_close_time.append({'start': {'hour': int(start_time_v[0]), 'minute': int(start_time_v[1])},
                                           'end': {'hour': int(end_time_v[0]), 'minute': int(end_time_v[1])}})
        if "log_dir" in mod_config.keys():
            self._log_dir = mod_config.log_dir
            if os.path.exists(self._log_dir) is False:
                os.makedirs(self._log_dir)

        #        env.event_bus.add_listener(EVENT.BAR, self._check_force_close)
        env.event_bus.prepend_listener(EVENT.BAR, self._check_force_close)

    def tear_down(self, success, exception=None):
        for f in self._log_file.values():
            if f:
                f.close()

    def _check_force_close(self, event):
        contract_list = list(event.bar_dict.keys())
        for contract in contract_list:
            event.bar_dict[contract].force_close = False
        cur_time = event.calendar_dt
        force_close = False
        for ft in self._force_close_time:
            start_time = cur_time.replace(hour=ft['start']['hour'], minute=ft['start']['minute'])
            end_time = cur_time.replace(hour=ft['end']['hour'], minute=ft['end']['minute'])
            if start_time <= cur_time <= end_time:
                force_close = True
                break
        if force_close:
            contract_list = list(event.bar_dict.keys())
            for contract in contract_list:
                long_positions = get_position(contract, POSITION_DIRECTION.LONG)
                short_positions = get_position(contract, POSITION_DIRECTION.SHORT)
                if long_positions.quantity == 0 and short_positions.quantity == 0:
                    continue
                # order_to(contract, 0)
                event.bar_dict[contract].force_close = True
                if not self._log_dir:
                    continue
                if not self._log_file[contract]:
                    path = os.path.join(self._log_dir, contract + '_force_close.csv')
                    self._log_file[contract] = open(path, 'w')
                msg = "%s,%s" % (str(cur_time), "FORCE_CLOSE")
                self._log_file[contract].write(msg + "\n")
            # return True
        return
        # print("call _calc_flow")
        # if event.bar_dict._frequency != "1m":
        #     return
        # if len(self._kline_bar) < self._kline_bar_cnt:
        #     self._kline_bar.append(event.)
