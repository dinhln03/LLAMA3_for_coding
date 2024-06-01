import jinja2

class SPMObject(object):
    """ Abstract Base Class for all SPM objects.
        
        Even though SPM objects are not Spire tasks (as some of them will modify
        in-place their file_dep, which is not compatible with doit's task 
        semantics), they nonetheless include task-related properties: file_dep
        and targets. Subclasses will have to override the _get_file_dep and
        _get_targets functions to return the correct values.
    """
    
    def __init__(self, name):
        self.name = name
        self.environment = jinja2.Environment()
        self.environment.globals.update(id=__class__._get_id)
    
    def get_script(self, index):
        template = self.environment.from_string(self.template)
        return template.render(index=index, **vars(self))
    
    @property
    def file_dep(self):
        return self._get_file_dep()
    
    @property
    def targets(self):
        return self._get_targets()
    
    @staticmethod
    def _get_id(index, name):
        return "matlabbatch{"+str(index)+"}."+name
    
    def _get_file_dep(self):
        return []
    
    def _get_targets(self):
        return []
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["environment"]
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.environment = jinja2.Environment()
        self.environment.globals.update(id=__class__._get_id)
