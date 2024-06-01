import functools
import jinja2
import json
import threading
from hashlib import sha256
from ..caserunconfiguration import CaseRunConfiguration, ConfigurationsList, CaseRunConfigurationsList
#from ..exceptions import UnknownEventSubTypeExpression

from .functions import dotted_startswith
from .structures.factory import EventStructuresFactory

class Event():
    """
    Base class of event which stores its type, event structures (automatically
    provides converted event structures) and decides which testplans and
    caserunconfigurations will be executed based on the event.

    This base class can be directly used just by providing settings, event type
    and optionally definitions of event_structures which are constructed using
    EventStructuresFactory. Such created event uses default selection of
    testplans with testcases and provides only the provided event structures
    (along with possibly automatically converted event structures).

    When defining new event type, one should create a new child class
    inheriting this class providing additional methods and/or properties.
    """
    def __init__(self, settings, event_type, **event_structures):
        self.settings = settings
        self.type = event_type
        self.structures = {}
        for structure_name, fields in event_structures.items():
            self.structures[structure_name] = EventStructuresFactory.make(settings, structure_name, fields)
        self.structures_convert_lock = threading.RLock()
        self.id = sha256(f'{self.type}-{json.dumps(event_structures, sort_keys=True)}'.encode()).hexdigest()

    def format_branch_spec(self, fmt):
        return jinja2.Template(fmt).render(event=self)

    def generate_caseRunConfigurations(self, library):
        """ Generates caseRunConfigurations for testcases in library relevant to this event

        :param library: Library
        :type library: tplib.Library
        :return: CaseRunConfigurations
        :rtype: CaseRunConfigurationsList
        """
        caseruns = CaseRunConfigurationsList()

        for testplan in self.filter_testPlans(library):
            # Init testplan configurations as ConfigurationsList
            testplan_configurations = ConfigurationsList(testplan.configurations,
                                                         merge_method=self.settings.get('library', 'defaultCaseConfigMergeMethod'))
            for testcase in testplan.verificationTestCases:
                # Merge testplan configurations with testcase configurations
                caserun_configurations = testplan_configurations.merge(testcase.configurations)
                for configuration in caserun_configurations:
                    # Create CaseRunConfiguration
                    caseruns.append(CaseRunConfiguration(testcase, configuration, [testplan]))

        return caseruns

    def handles_testplan_artifact_type(self, artifact_type):
        """
        Decide if this event is relevant to the provided artifact_type (which
        is found in test plan).
        """
        return dotted_startswith(self.type, artifact_type)

    def filter_testPlans(self, library):
        """ Filters testplan from library based on:
        - event type and testplan.artifact_type
        - testplan execute_on filter

        :param library: pipeline library
        :type library: tplib.Library
        :return: Filtered testplans
        :rtype: list of tplib.TestPlan
        """
        return library.getTestPlansByQuery('event.handles_testplan_artifact_type(tp.artifact_type) and tp.eval_execute_on(event=event)', event=self)

    @property
    def additional_testplans_data(self):
        """ Event can provide additional testplans. Returns python
        dicts, as if they were tplib files read by yaml.safe_load.

        :return: list of testplan data
        :rtype: tuple
        """
        return None

    @property
    def additional_requrements_data(self):
        """ Event can provide additional requrements. Returns python
        dicts, as if they were tplib files read by yaml.safe_load.

        :return: list of requrements data
        :rtype: tuple
        """
        return None

    @property
    def additional_testcases_data(self):
        """ Event can provide additional testcases. Returns python
        dicts, as if they were tplib files read by yaml.safe_load.

        :return: list of testcases data
        :rtype: tuple
        """
        return None

    def __getattr__(self, attrname):
        if attrname not in EventStructuresFactory.known():
            return super().__getattribute__(attrname)
        with self.structures_convert_lock:
            try:
                return self.structures[attrname]
            except KeyError:
                pass
            structure = EventStructuresFactory.convert(attrname, self.structures)
            if structure is NotImplemented:
                # Return None if the requested structure is not compatible to
                # allow jinja templates to not crash on expressions like
                # event.nonexisting_structure.foo but to consider them as None
                return None
            self.structures[attrname] = structure
            return structure

def payload_override(payload_name):
    def decorator(method):
        @functools.wraps(method)
        def decorated(self, *args, **kwargs):
            try:
                return self.payload[payload_name]
            except KeyError:
                return method(self, *args, **kwargs)
        return decorated
    return decorator
