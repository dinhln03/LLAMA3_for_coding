import sys
from jennie.jennie_tools.command_handler import *
from jennie.ubuntu import *

def execute():
    arguments = sys.argv[1:]

    commands = CommandHandler().start(arguments)

    if not commands or commands == None:
        return

    elif commands == True:
        return

    elif commands[0] == "ubuntu":
        if commands[1] == "setup":
            if commands[2] == "elk":
                setup_elasticsearchkibana()
            elif commands[2] == "elasticsearch":
                setup_elasticsearch()
            elif commands[2] == "lemp":
                setup_lemp()
            elif commands[2] == "phpmyadmin":
                install_phpmyadmin()

        elif commands[1] == "deploy":
            info = take_user_input(DEPLOY_INFO_COMMANDS)
            if commands[2] == "web":
                deploy_folder_nginx(info["port"], info["domain"])
            elif commands[2] == "django":
                deploy_django(info["port"], info["domain"])

if __name__ == '__main__':
    execute()