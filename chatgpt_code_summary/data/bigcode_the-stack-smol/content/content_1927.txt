# -*- coding: utf-8 -*-
import os

from O365 import Account, Connection, FileSystemTokenBackend
from datetime import datetime as dt
from datetime import timedelta
from conf.conf import CONFIG as conf
from fritzhome import FritzBox
import logging


class Core:

    @staticmethod
    def get_credentials():
        return conf['OFFICE_CLIENT_ID'], conf['OFFICE_CLIENT_SECRET']

    @staticmethod
    def get_account():
        return Account(credentials=Core.get_credentials())

    @staticmethod
    def get_scopes():
        return ['offline_access',
                'https://graph.microsoft.com/Mail.ReadWrite',
                'https://graph.microsoft.com/Mail.Send',
                'https://graph.microsoft.com/Calendars.Read',
                'https://graph.microsoft.com/Files.ReadWrite',
                'https://graph.microsoft.com/User.Read']

    @staticmethod
    def get_con_obj():
        credentials = (conf['OFFICE_CLIENT_ID'], conf['OFFICE_CLIENT_SECRET'])
        scopes = Core.get_scopes()

        return Connection(credentials, scopes=scopes, token_backend=FileSystemTokenBackend(token_filename='o365_token.txt'))

    def run(self):
        con = Core.get_con_obj()

        if not con.token_backend.check_token():
            logging.error("You have to generate your token file with python -m radiator_fritz_o365_sync.gen_token first!")
            exit(1)

        con.refresh_token()

        heating = self.query_for_heating_periods()

        # Cool down if no heating entries found in calendar
        if len(heating) == 0:
            logging.debug('No heating entry in calendar found. Cooling down all thermostats if they are heating. ')
            self.cool_down_all()

        # For each heating entry in calendar heat up
        subjects = []
        for heat in heating:
            logging.info('Found entry "%s"', heat.subject)
            self.heat_up(heat.subject)
            subjects.append(heat.subject)

        # Cool down thermostats if they are not heated
        self.cool_down_unless(subjects)

        # auto reset
        if len(heating) == 0:
            self.auto_reset()

        # Every night refresh the token and cool down to reset manual changes on thermostats
        if dt.now().time().strftime('%H:%M') == '00:00':
            con.refresh_token()

    """
    Gets all thermostats from fritzbox
    """
    def get_thermostats(self):
        if conf['FRITZ_TLS']:
            fritzbox = FritzBox(conf['FRITZ_IP'], conf['FRITZ_USER'], conf['FRITZ_PW'], use_tls=conf['FRITZ_TLS'], tls_cert_path='conf/fritz.crt')
        else:
            fritzbox = FritzBox(conf['FRITZ_IP'], conf['FRITZ_USER'], conf['FRITZ_PW'])
        fritzbox.login()
        actors = fritzbox.get_actors()
        thermostats = []
        for actor in actors:
            if actor.has_heating_controller:
                thermostats.append(actor)

        return thermostats

    def thermostat_heatup(self, actor):
        if actor.target_temperature == conf['HEATING_LOW_TEMP']:
            logging.info('Heating up %s ...', actor.name)
            actor.set_temperature(conf['HEATING_COMFORT_TEMP'])

    """
    Sets the temperature of thermostats with matching subject or all thermostats to comfort temperature
    """
    def heat_up(self, sub):
        thermostats = self.get_thermostats()
        for thermostat in thermostats:
            if sub == conf['CALENDAR_HEAT_ALL_SUBJECT']:
                self.thermostat_heatup(thermostat)
            else:
                if thermostat.name == sub:
                    self.thermostat_heatup(thermostat)

    """
    Cool down every thermostat which is not in unless list
    """
    def cool_down_unless(self, unless):
        # return if wildcard is found in subjects
        if conf['CALENDAR_HEAT_ALL_SUBJECT'] in unless:
            return

        thermostats = self.get_thermostats()
        for thermostat in thermostats:
            if thermostat.name not in unless:
                self.cool_down(thermostat)

    """
    Sets the temperature of all thermostats to LOW_TEMP if they are currently set to COMFORT_TEMP
    """
    def cool_down_all(self):
        thermostats = self.get_thermostats()
        for thermostat in thermostats:
            self.cool_down(thermostat)
    """
    Sets the temperature of thermostat to low temp if it is on comfort temp
    """
    def cool_down(self, thermostat):
        if thermostat.target_temperature == conf['HEATING_COMFORT_TEMP']:
            logging.info('Cooling down %s ...', thermostat.name)
            thermostat.set_temperature(conf['HEATING_LOW_TEMP'])
    """
    If the temperature has changed manually via app or on the thermostat itself, 
    this method resets the temperature to the HEATING_LOW_TEMP on a given time
    """
    def auto_reset(self):
        if conf['HEATING_AUTO_RESET']:
            current_time = dt.now().time()
            target_time = conf['HEATING_AUTO_RESET_TIME']

            if current_time.strftime('%H:%M') == target_time:
                logging.info('Resetting temperature on all thermostats now!')
                thermostats = self.get_thermostats()
                for thermostat in thermostats:
                    thermostat.set_temperature(conf['HEATING_LOW_TEMP'])

    def query_for_heating_periods(self):
        account = Core.get_account()
        schedule = account.schedule()
        calendar = schedule.get_calendar(calendar_name=conf['CALENDAR_NAME'])

        if calendar is None:
            logging.error("Calendar with name '%s' does not exist!", conf['CALENDAR_NAME'])
            exit(1)

        q = calendar.new_query('start').greater_equal(dt.now())
        q.chain('and').on_attribute('end').less_equal(dt.now() + timedelta(minutes=5))

        return list(calendar.get_events(query=q))

if __name__ == "__main__":
    Core().run()
