#from sql_gen.sql_gen.filters import *

class Prompter(object):
    def __init__(self, template_source):
        self.template_source = template_source

    def get_prompts(self):
        result=[]
        for undeclared_var in self.template_source.find_undeclared_variables():
            result.append(Prompt(undeclared_var,self.template_source.get_filters(undeclared_var)))
        return result


    def build_context(self):
        prompts = self.get_prompts()
        context ={}

        for prompt in prompts:
            prompt.populate_value(context)
        return context

class Prompt:
    def __init__(self, variable_name, filter_list):
        self.variable_name =variable_name
        self.filter_list = filter_list

    def get_diplay_text(self):
        self.display_text = self.variable_name
        for template_filter in self.filter_list:
            self.display_text = template_filter.apply(self.display_text);
        return self.display_text+": "

    def populate_value(self,context):
        var =raw_input(self.get_diplay_text())
        if var:
            context[self.variable_name] = var
