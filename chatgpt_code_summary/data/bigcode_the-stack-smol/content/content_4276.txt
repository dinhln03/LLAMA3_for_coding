import simpy
import logging

"""
Recursive GraphQL schema for JSON StructureItem - passed as Python dictionary


type StructureItem {
  id: ID!
  type: StructureType
  # optional annotation for a Branch
  annotation: String
  # reference UUID / Name / Num for: Function, Exit / ExitCondition (Exit), Replicate (DomainSet) types
  referenceID: String
  referenceName: String
  referenceNum: String
  structure: [StructureItem]
}
"""


class StructureItem:
    """
    The base class for all call StructureItems (Branch, Parallel, Select, Loop, Function, etc.)
    """

    def __init__(self, env: simpy.Environment, logger: logging.Logger,
                 construct_id: str, systemModel: dict, structureItem: dict):

        from .branch import Branch
        from .function import Function
        #from .exit import Exit, ExitCondition
        from .loop import Loop  # , LoopExit
        from .parallel import Parallel
        #from .replicate import Replicate
        from .select import Select
        import simapp

        self.env = env
        self.logger = logger
        self.construct_id = construct_id
        self.systemModel = systemModel
        self.structureItem = structureItem
        self.structureItems = list()
        self.structureType = ""  # overidden by subclass
        self.name = "" # overidden by subclass

        for num, struct in enumerate(self.structureItem['structure'], start=1):
            next_construct_id = self.construct_id + "." + str(num)

            if struct['type'] == "Branch":
                self.structureItems.append(Branch(self.env, self.logger,
                                                  next_construct_id, self.systemModel, struct))
            elif struct['type'] == "Function":
                try:
                    # Check for override function
                    override_class = getattr(simapp, struct['referenceName'].capitalize())
                    self.structureItems.append(override_class(self.env, self.logger,
                                                    next_construct_id, self.systemModel, struct))
                except AttributeError:
                    # No function override exists
                    self.structureItems.append(Function(self.env, self.logger,
                                                    next_construct_id, self.systemModel, struct))
            elif struct['type'] == "Loop":
                self.structureItems.append(Loop(self.env, self.logger,
                                                next_construct_id, self.systemModel, struct))
            elif struct['type'] == "Parallel":
                self.structureItems.append(Parallel(self.env, self.logger,
                                                    next_construct_id, self.systemModel, struct))
            elif struct['type'] == "Select":
                self.structureItems.append(Select(self.env, self.logger,
                                                  next_construct_id, self.systemModel, struct))

    def __str__(self):
        """
        Recursively print CallStructure
        """
        # indent by construct_id depth (number of dots)
        stmt = ("\n" + "." * self.construct_id.count(".") +
                "Struct: %s: %s" % (self.construct_id, self.structureType))

        for struct in self.structureItems:
            stmt += struct.__str__()

        return (stmt)

    def log_start(self):
        self.logger.info('SIM Time: %08.2f : %-20s:Start:%10s:%-s' %
                         (self.env.now, self.construct_id, self.structureType, self.name))

    def log_end(self):
        self.logger.info('SIM Time: %08.2f : %-20s:  End:%10s:%-s' %
                         (self.env.now, self.construct_id, self.structureType, self.name))
