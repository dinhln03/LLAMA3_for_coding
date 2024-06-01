import time
import threading
import subprocess
import helpers
from settings import Settings

def listener():
    global data_source
    print("**** SIDE_THREAD ID == ", threading.get_ident())
    while True:
        return_data = subprocess.run([data_source.settings.loaded['localization_bin']], stdout=subprocess.PIPE)
        parsed_return = helpers.parse_yaml_string(return_data.stdout.decode('utf8'))
        data_source.settings.add_runtime('localization', parsed_return)
        time.sleep(data_source.settings.loaded['localization_plugin_wait_time']) # Waits 1 second till the next localization check

def start_plugin(data_source_received):
    global data_source
    data_source = data_source_received
    try:
        thread = threading.Thread(target=listener)
        thread.start()

    except:
        print("Failed to start localization plugin")
