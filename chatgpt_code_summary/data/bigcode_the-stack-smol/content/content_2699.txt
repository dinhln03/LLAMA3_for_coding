import logging
import configparser
import os

from utils import bool_query

class BreakRule(object):
    def __init__(self, settings):
        self.settings = settings

        self.rules_record = configparser.ConfigParser()
        self.rules_record.read("{}/tms/breakrules.ini".format(os.getcwd()))

        self.rules = {}
        for rule_id in self.rules_record.sections():
            self.rules[rule_id] = self.rules_record.get(rule_id, "Description")

    def _check_rule_exists(self, rule_id):
        if self.rules.get(rule_id, None) is None:
            logging.warning("Rule {} doesn't exist".format(rule_id))
            return False
        else:
            logging.debug("Rule {} exists".format(rule_id))
            return True
    
    def _update_break_rule(self, rule_id):

        self.settings.set("Settings", "BreakRule", rule_id)
        with open("{}/tms/settings.ini".format(os.getcwd()), 'w') as configfile:
            self.settings.write(configfile)
        logging.info("Break rule changed to rule {}".format(self.settings.get("Settings", "BreakRule")))

    def print_rules(self):
        logging.info("Break Rules: ")
        for rule_id in self.rules:
            logging.info('  [{}] {}'.format(rule_id, self.rules[rule_id]))

    def get_break_rule(self, desired_rule_id=None):
        if not desired_rule_id: desired_rule_id = self.settings.get("Settings", "BreakRule")
        if self._check_rule_exists(desired_rule_id):
            for rule_id in self.rules:
                if rule_id == desired_rule_id:
                    logging.info('  [{}] {}'.format(rule_id, self.rules[desired_rule_id]))

    def cmd_update_break_rule(self):
        self.print_rules()

        selection_query = None
        while selection_query is None:
            logging.info('Please enter the ID of the rule to be used...')
            selection = input()
            try:
                int(selection)
            except ValueError:
                logging.warning('WARNING: Please enter a numeric value corresponding to a rule ID.')
            else:
                if self._check_rule_exists(selection):
                    selection_query = bool_query('Select Rule "{}" for use?'.format(selection, default="y"))
        
        self._update_break_rule(selection)
