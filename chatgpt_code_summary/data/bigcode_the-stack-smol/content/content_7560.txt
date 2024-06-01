from ermaket.api.scripts import ReturnContext, UserScript

__all__ = ['script']

script = UserScript(id=2)


@script.register
def no_way(context):
    return ReturnContext(abort=418, abort_msg="I am a teapot")
