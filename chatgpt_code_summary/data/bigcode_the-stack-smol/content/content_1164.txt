from cloudify import ctx
from cloudify.state import ctx_parameters as inputs
from cloudify.decorators import operation
from cloudify.exceptions import *
from plugin.nodes.utils import *

def build_radl_flavour(config):
    ctx.logger.debug('{0} Infrastructure Manager deployment info:'.format(get_log_indentation()))
    increase_log_indentation()

    type = get_child(dictionary=config, key='type', required=True)
    cores = get_child(dictionary=config, key='cores', required=True)
    memory = get_child(dictionary=config, key='memory', required=True)

    flavour_radl = \
"    instance_type = '" + str(type) + "' and \n" + \
"    cpu.count = " + str(cores) + " and \n" + \
"    memory.size = " + str(memory) + " and \n"

    decrease_log_indentation()
    return flavour_radl

@operation
def configure(config, simulate, **kwargs):
    if (not simulate):
        reset_log_indentation()
        ctx.logger.debug('{0} Configure operation: Begin'.format(get_log_indentation()))
        increase_log_indentation()
        radl = get_child(ctx.instance.runtime_properties, key='settings')
        if not radl:
            radl = create_child(ctx.instance.runtime_properties, key='settings', value={})
        radl_network = create_child(radl, key='flavour', value=build_radl_flavour(config))
        decrease_log_indentation()
        ctx.logger.debug('{0} Configure operation: End'.format(get_log_indentation()))
    
