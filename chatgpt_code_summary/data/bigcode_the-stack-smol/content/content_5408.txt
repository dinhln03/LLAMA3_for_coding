# =======================================================================================
#                          \    |  | __ __| _ \  |  /  __| \ \  /  __| 
#                         _ \   |  |    |  (   | . <   _|   \  / \__ \ 
# @autor: Luis Monteiro _/  _\ \__/    _| \___/ _|\_\ ___|   _|  ____/ 
# =======================================================================================
from autokeys.engine import Keyboard, HotKeys, SeqKeys, Clipboard


# =======================================================================================
# build credentials config 
# =======================================================================================
def config_credentials(data):
    # actions
    def write_user(user):
        def process(x):
            Keyboard.Type(user, len(x))
        return process
    def write_pass(password):
        def process(x):
            Keyboard.Type(password, len(x))
            Clipboard.Stage(password)
        return process


    # build config
    hotkeys_user = HotKeys(Keyboard.CTRL, Keyboard.ALT, Keyboard.KEY('u'))
    hotkeys_pass = HotKeys(Keyboard.CTRL, Keyboard.ALT, Keyboard.KEY('p'))
    hotkeys_conf = {
        hotkeys_user:{},
        hotkeys_pass:{}}
    for key, entry in data.items():
        # user
        hotkeys_conf[hotkeys_user][SeqKeys(*[Keyboard.KEY(x) for x in key])] = write_user(entry['user']) 
        # pass
        hotkeys_conf[hotkeys_pass][SeqKeys(*[Keyboard.KEY(x) for x in key])] = write_pass(entry['pass'])
    return hotkeys_conf
