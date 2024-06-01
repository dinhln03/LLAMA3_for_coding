from shallowflow.api.source import AbstractListOutputSource
from shallowflow.api.config import Option


class ForLoop(AbstractListOutputSource):
    """
    Outputs an integer from the specified range.
    """

    def description(self):
        """
        Returns a description for the actor.

        :return: the actor description
        :rtype: str
        """
        return "Outputs an integer from the specified range."

    def _define_options(self):
        """
        For configuring the options.
        """
        super()._define_options()
        self._option_manager.add(Option(name="start", value_type=int, def_value=1,
                                        help="The starting value"))
        self._option_manager.add(Option(name="end", value_type=int, def_value=10,
                                        help="The last value (incl)"))
        self._option_manager.add(Option(name="step", value_type=int, def_value=1,
                                        help="The increment between values"))

    def _get_item_type(self):
        """
        Returns the type of the individual items that get generated, when not outputting a list.

        :return: the type that gets generated
        """
        return int

    def setup(self):
        """
        Prepares the actor for use.

        :return: None if successful, otherwise error message
        :rtype: str
        """
        result = super().setup()
        if result is None:
            if self.get("end") < self.get("start"):
                result = "End value (%s) must be smaller than start (%d)!" % (self.get("end"), self.get("start"))
        return result

    def _do_execute(self):
        """
        Performs the actual execution.

        :return: None if successful, otherwise error message
        :rtype: str
        """
        i = self.get("start")
        step = self.get("step")
        end = self.get("end")
        while i <= end:
            self._output.append(i)
            i += step
        return None
