#------------------Bombermans Team---------------------------------# 
# Author  : B3mB4m
# Concat  : b3mb4m@protonmail.com
# Project : https://github.com/b3mb4m/Shellsploit
# LICENSE : https://github.com/b3mb4m/Shellsploit/blob/master/LICENSE
#------------------------------------------------------------------#

import sys
import os
from .core.color import *
from re import findall
from .core.Comp import tab
from lib.base.framework import ShellsploitFramework 


if sys.version_info.major >= 3:
    raw_input = input


class B3mB4m(ShellsploitFramework):   

    def __init__(self):
        ShellsploitFramework.__init__(self)  
        self.argvlist = ["None", "None", "None", "None"]
        self.disassembly = "None"
        self.mycache = "None"

    def control(self, string):
        bash = bcolors.OKBLUE + bcolors.UNDERLINE + "ssf" + bcolors.ENDC
        bash += ":"
        bash += bcolors.RED + string + bcolors.ENDC
        bash += bcolors.OKBLUE + " > " + bcolors.ENDC

        try:
            terminal = raw_input(bash)
        except KeyboardInterrupt:
            B3mB4m.exit("\n[*] (Ctrl + C ) Detected, Trying To Exit ...")

        # Injectors
        if string[:9] == "injectors":
            tab.completion("injectors")
            if terminal[:4] == "help":
                from .core.help import injectorhelp
                injectorhelp()
                self.control(string)

            elif terminal[:4] == "back":
                self.argvlist = ["None", "None", "None", "None"]
                pass

            # elif terminal[:9] == "need help":
                # import XX
                # print youtubelink for this module

            elif terminal[:4] == "exit":
                B3mB4m.exit("\nThanks for using shellsploit !\n")    

            elif terminal[:4] == "pids":
                B3mB4m.pids("wholelist")
                self.control(string)

            elif terminal[:6] == "getpid":
                B3mB4m.pids(None, terminal[7:])
                self.control(string)

            elif terminal[:5] == "clear":
                B3mB4m.clean()
                self.control(string)

            elif terminal[:5] == "unset":
                if string in B3mB4m.bfdlist():
                    if terminal[6:] == "exe" or terminal[6:] == "file":
                        self.argvlist[0] = "None"   
                    elif terminal[6:] == "host":
                        self.argvlist[1] = "None"
                    elif terminal[6:] == "port":
                        self.argvlist[2] = "None"
                    else:
                        print(bcolors.RED + bcolors.BOLD + "[-] Unknown command: {0}".format(terminal) + bcolors.ENDC)                      

                elif string == "injectors/Windows/x86/tLsInjectorDLL":
                    if terminal[6:] == "exe":
                        self.argvlist[0] = "None"   
                    elif terminal[6:] == "dll":
                        self.argvlist[1] = "None"
                    else:
                        print(bcolors.RED + bcolors.BOLD + "[-] Unknown command: {0}".format(terminal) + bcolors.ENDC)   
                elif string == "injectors/Windows/x86/CodecaveInjector":
                    if terminal[6:] == "exe":
                        self.argvlist[0] = "None"
                    elif terminal[6:] == "shellcode":
                        self.argvlist[1] = "None"                 
                else:
                    if terminal[6:] == "pid":
                        self.argvlist[0] = "None"   
                    elif terminal[6:] == "shellcode":
                        self.argvlist[1] = "None"
                    else:
                        print(bcolors.RED + bcolors.BOLD + "[-] Unknown command: {0}".format(terminal) + bcolors.ENDC)
                self.control(string)

            elif terminal[:3] == "set":
                if string in B3mB4m.bfdlist():
                    if terminal[4:7] == "exe" or terminal[4:8] == "file":
                        self.argvlist[0] = terminal[9:]
                    elif terminal[4:8] == "host":
                        self.argvlist[1] = terminal[9:]
                    elif terminal[4:8] == "port":
                        self.argvlist[2] = terminal[9:]
                    else:
                        if not terminal:
                            self.control(string)
                        else:
                            print(bcolors.RED + bcolors.BOLD + "[-] Unknown command: {0}".format(terminal) + bcolors.ENDC)

                elif string == "injectors/Windows/x86/tLsInjectorDLL":
                    if terminal[4:7] == "exe":
                        self.argvlist[0] = terminal[8:]
                    elif terminal[4:7] == "dll":
                        self.argvlist[1] = terminal[8:]
                    else:
                        if not terminal:
                            self.control(string)
                        else:
                            print(bcolors.RED + bcolors.BOLD + "[-] Unknown command: {0}".format(terminal) + bcolors.ENDC)

                elif string == "injectors/Windows/x86/CodecaveInjector":
                    if terminal[4:7] == "exe":
                        self.argvlist[0] = terminal[8:]
                    elif terminal[4:13] == "shellcode":
                        self.argvlist[1] = terminal[14:]
                    else:
                        if not terminal:
                            self.control(string)
                        else:
                            print(bcolors.RED + bcolors.BOLD + "[-] Unknown command: {0}".format(terminal) + bcolors.ENDC)

                else:
                    if terminal[4:7] == "pid":
                        self.argvlist[0] = terminal[8:]
                    elif terminal[4:13] == "shellcode":
                        if ".txt" in terminal[14:]:
                            if os.path.isfile(terminal[14:]):
                                with open(terminal[14:], "r") as shellcode:
                                    cache = shellcode.readlines()   
                                    db = ""
                                    for x in database:
                                        db += x.strip().replace('"', "").replace('+', "").strip()
                                    self.argvlist[1] = db
                            else:
                                print(bcolors.RED + bcolors.BOLD + "\nFile can't find, please try with full path.\n" + bcolors.ENDC)
                                self.control(string)
                        else:
                            self.argvlist[1] = terminal[14:]
                    else:
                        if not terminal:
                            self.control(string)
                        else:
                            print(bcolors.RED + bcolors.BOLD + "[-] Unknown command: {0}".format(terminal) + bcolors.ENDC)
                self.control(string)

            elif terminal[:14] == "show shellcode":
                if string in B3mB4m.bfdlist():
                    print("This option not available for this module.")
                    self.control(string)
                elif string == "injectors/Windowsx86/tLsInjectorDLL":
                    self.control(string)
                else:
                    if self.argvlist[1] != "None":
                        B3mB4m.prettyout(self.argvlist[1])
                    else:
                        print("\nYou must set shellcode before this ..\b")
                    self.control(string)

            elif terminal[:12] == "show options":
                from .core.Injectoroptions import controlset
                if string in B3mB4m.bfdlist():
                    controlset(string, self.argvlist[0], self.argvlist[1], self.argvlist[2])
                    self.control(string)
                else:
                    if string != "injectors/Windows/x86/tLsInjectorDLL":
                        if self.argvlist[1] != "None":
                            self.mycache = "process"
                            controlset(string, self.argvlist[0], self.mycache)
                            self.control(string)
                    controlset(string, self.argvlist[0], self.argvlist[1])
                    self.control(string)

            elif terminal[:5] == "clear":
                B3mB4m.clean()
                self.control(string)   

            elif terminal[:2] == "os":
                B3mB4m.oscommand(terminal[3:])
                self.control(string)

            elif terminal[:6] == "inject":
                if self.argvlist[0] == None or self.argvlist[1] == None:
                    print("\nYou must set pid/shellcode before inject !\n")
                    self.control(string)
                if string == "injectors/Linux86/ptrace":
                    from .inject.menager import linux86ptrace
                    linux86ptrace(self.argvlist[0], self.argvlist[1])
                elif string == "injectors/Linux64/ptrace":
                    from .inject.menager import linux64ptrace
                    linux64ptrace(self.argvlist[0], self.argvlist[1])                  
                elif string == "injectors/Windows/byteman":
                    from .inject.menager import windows
                    windows(self.argvlist[0], self.argvlist[1])
                elif string == "injectors/Windows/x86/tLsInjectorDLL":
                    from .inject.menager import winx86tLsDLL
                    winx86tLsDLL(self.argvlist[0], self.argvlist[1])
                elif string == "injectors/Windows/x86/CodecaveInjector":
                    from .inject.menager import winx86Codecave
                    winx86Codecave(self.argvlist[0], self.argvlist[1])
                elif string == "injectors/Windows/Dllinjector":
                    from .inject.menager import winDLL
                    winDLL(self.argvlist[0], self.argvlist[1])        

                elif string == "injectors/Windows/BFD/Patching":
                    from .inject.menager import winBFD
                    winBFD(self.argvlist[0], self.argvlist[1], int(self.argvlist[2]))

                # elif string == "injectors/MacOSX/BFD/Patching":
                    # from .inject.menager import MacBFD
                    # MacBFD( FILE, HOST, PORT)          

                # elif string == "injectors/Linux/BFD/Patching":
                    # from .inject.menager import LinuxBFD
                    # LinuxBFD( FILE, HOST, PORT)

                # elif string == "injectors/Linux/ARM/x86/BFD/Patching":
                    # from .inject.menager import LinuxARMx86BFD
                    # LinuxARMx86BFD( FILE, HOST, PORT)                    

                # elif string == "FreeBSD/x86/BFD/Patching":
                    # from .inject.menager import FreeBSDx86
                    # FreeBSDx86( FILE, HOST, PORT)                    

                self.control(string)

            # elif terminal[:7] == "extract":
                # Future option
                # Make it executable (Dynamic virus land)
                # from bla bla import executable
                # generator()

            elif terminal[:4] == "back":
                self.argvlist = ["None", "None", "None", "None"]
                pass

            else:
                if not terminal:
                    self.control(string)
                else:
                    print(bcolors.RED + bcolors.BOLD + "[-] Unknown command: {0}".format(terminal) + bcolors.ENDC)
                    self.control(string)

        # Backdoors
        elif string[:9] == "backdoors":
            tab.completion("backdoors")
            if terminal[:4] == "help":
                from .core.help import backdoorshelp
                backdoorshelp()
                self.control(string)

            elif terminal[:4] == "exit":
                B3mB4m.exit("\nThanks for using shellsploit !\n")

            elif terminal[:2] == "os":
                B3mB4m.oscommand(terminal[3:])
                self.control(string)

            elif terminal[:12] == "show options":       
                from .core.SHELLoptions import controlset
                controlset(string, self.argvlist[0], self.argvlist[1])
                self.control(string)

            elif terminal[:5] == "unset":
                if terminal[6:] == "lhost":
                    self.argvlist[0] = "None"   
                elif terminal[6:] == "lport":
                    self.argvlist[1] = "None"
                # elif terminal[6:] == "encoder":
                    # self.argvlist[2] = "None"
                else:
                    print(bcolors.RED + bcolors.BOLD + "[-] Unknown command: {0}".format(terminal) + bcolors.ENDC)
                self.control(string)

            elif terminal[:3] == "set": 
                if terminal[4:9].lower() == "lhost":
                    self.argvlist[0] = terminal[10:]
                elif terminal[4:9].lower() == "lport":
                    self.argvlist[1] = terminal[10:]
                # elif terminal[4:11].lower() == "encoder"
                    # self.argvlist[2] = terminal[11:]
                else:
                    print(bcolors.RED + bcolors.BOLD + "This option is not available." + bcolors.ENDC)
                self.control(string)

            elif terminal[:8] == "generate":
                from .Session.generator import process
                # Custom output path will be add .. 
                if self.argvlist[0] == "None" or self.argvlist[1] == "None":
                    print("\nSet options before generate payload.\n")
                    self.control(string)
                else:
                    process(data=string, HOST=self.argvlist[0], PORT=self.argvlist[1], ENCODER=False, logger=True)
                    self.control(string)

            elif terminal[:5] == "clear":
                B3mB4m.clean()
                self.control(string)

            elif terminal[:4] == "back":
                self.argvlist = ["None", "None", "None", "None"]
                pass

            else:
                if not terminal:
                    self.control(string)
                else:
                    print(bcolors.RED + bcolors.BOLD + "[-] Unknown command: {0}".format(terminal) + bcolors.ENDC)
                    self.control(string)

        # Shellcodes
        else:
            tab.completion("shellcodes")
            if terminal[:4] == "help":
                # if terminal[5:11] == "output":
                    # from Outputs.exehelp import help
                    # print help()
                    # self.control( string)
                from .core.help import shellcodehelp
                shellcodehelp()
                self.control(string)

            elif terminal[:2] == "os":
                B3mB4m.oscommand(terminal[3:])
                self.control(string)

            elif terminal[:4] == "back":
                self.argvlist = ["None", "None", "None", "None"]
                pass

            elif terminal[:4] == "exit":
                B3mB4m.exit("\nThanks for using shellsploit !\n")

            elif terminal[:10] == "whatisthis":
                from .core.whatisthis import whatisthis
                if "egg" in string:
                    message = "Egg-hunt"
                elif "tcp" in string or "reverse" in string or "netcat" in string:
                    message = "Remote"
                elif "download" in string:
                    message = "Download and execute"
                else:
                    message = "Local"
                # Add special part for particul
                whatisthis(message)
                self.control(string)

            elif terminal[:5] == "unset":
                if terminal[6:] == "encoder":
                    self.argvlist[0] = "None"   
                elif terminal[6:] == "iteration":
                    self.argvlist[1] = "None"
                elif terminal[6:] == "file":
                    if string in B3mB4m.readlist():
                        self.argvlist[2] = "None"
                    else:
                        B3mB4m.invalidcommand()
                elif terminal[6:] == "port":
                    if string in B3mB4m.tcpbindlist() or string in B3mB4m.reversetcplist():
                        self.argvlist[2] = "None"
                    else:
                        Base.invalidcommand()
                elif terminal[6:] == "command":
                    if string in B3mB4m.execlist():
                        self.argvlist[2] = "None"
                    else:
                        B3mB4m.invalidcommand()
                elif terminal[6:] == "link":
                    if string in B3mB4m.downloadandexecutelist():
                        self.argvlist[2] = "None"
                    else:
                        B3mB4m.invalidcommand()			
                elif terminal[6:] == "filename":
                    if string in B3mB4m.downloadandexecutelist():
                        self.argvlist[3] = "None"
                    else:
                        B3mB4m.invalidcommand()	
                elif terminal[6:] == "host":
                    if string in B3mB4m.reversetcplist():
                        self.argvlist[3] = "None"
                    else:
                        B3mB4m.invalidcommand()	
                else:
                    B3mB4m.invalidcommand()
                self.control(string)

            elif terminal[:3] == "set":
                if terminal[4:8] == "file":
                    if string in B3mB4m.readlist():
                        self.argvlist[2] = terminal[9:]
                    else:
                        B3mB4m.invalidcommand()   
                elif terminal[4:8] == "port":
                    if string in B3mB4m.tcpbindlist() or string in B3mB4m.reversetcplist():
                        self.argvlist[2] = terminal[9:]
                    else:
                        B3mB4m.invalidcommand()   
                elif terminal[4:11] == "command":
                    if string in B3mB4m.execlist():
                        self.argvlist[2] = terminal[12:]
                    else:
                        B3mB4m.invalidcommand()   
                elif terminal[4:8] == "link":
                    if string in B3mB4m.downloadandexecutelist():
                        self.argvlist[2] = terminal[9:]
                    else:
                        B3mB4m.invalidcommand()   
                elif terminal[4:11] == "message":
                    if string in B3mB4m.messageboxlist():
                        self.argvlist[2] = terminal[12:]
                    else:
                        B3mB4m.invalidcommand()   
                elif terminal[4:8] == "host":
                    if string in B3mB4m.reversetcplist():
                        self.argvlist[3] = terminal[9:]
                    else:
                        B3mB4m.invalidcommand()
                elif terminal[4:12] == "filename":
                    if string in B3mB4m.downloadandexecutelist():
                        self.argvlist[3] = terminal[13:]
                    else:
                        B3mB4m.invalidcommand()					
                elif terminal[4:11] == "encoder":
                    from .core.lists import encoders
                    if terminal[12:] not in encoders():
                        print("This encoder not in list !")
                        self.control(string)
                    self.argvlist[0] = terminal[12:]
                elif terminal[4:13] == "iteration":
                    self.argvlist[1] = terminal[14:]
                else:
                    B3mB4m.invalidcommand()     
                self.control(string)   

            elif terminal[:12] == "show options":
                from .core.SHELLoptions import controlset
                if string[:7] == "linux86":
                    if string == "linux86/read":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "linux86/chmod":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "linux86/tcp_bind":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "linux86/reverse_tcp":
                        controlset(string, self.argvlist[3], self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "linux86/download&exec":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "linux86/exec":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    else:
                        controlset(string, self.argvlist[0], self.argvlist[1])
                    self.control(string)

                elif string[:10] == "solarisx86":
                    if string == "solarisx86/read":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "solarisx86/reverse_tcp":
                        controlset(string, self.argvlist[3], self.argvlist[2], self.argvlist[0], self.argvlist[1]) 
                    elif string == "solarisx86/tcp_bind":
                        controlset(string, self.argvlist[2], self.argvlist[3], self.argvlist[0], self.argvlist[1])     
                    else:
                        controlset(string, self.argvlist[0], self.argvlist[1])
                    self.control(string)

                elif string[:7] == "linux64":
                    if string == "linux64/read":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])                   
                    elif string == "linux64/mkdir":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "linux64/tcp_bind":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])               
                    elif string == "linux64/reverse_tcp":
                        controlset(string, self.argvlist[2], self.argvlist[3], self.argvlist[1], self.argvlist[0])
                    else:
                        controlset(string, self.argvlist[0], self.argvlist[1])
                    self.control(string)

                elif string[:5] == "linux":
                    if string == "linux/read":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "linux/tcp_bind":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "linux/reverse_tcp":
                        controlset(string, self.argvlist[2], self.argvlist[3], self.argvlist[0], self.argvlist[1])                 
                    else:
                        controlset(string, self.argvlist[0], self.argvlist[1])
                    self.control(string)

                elif string[:5] == "osx86":
                    if string == "osx86/tcp_bind":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "osx86/reverse_tcp":
                        controlset(string, self.argvlist[2], self.argvlist[3], self.argvlist[1], self.argvlist[0])
                    else:
                        controlset(string, self.argvlist[0], self.argvlist[1])
                    self.control(string)

                elif string[:5] == "osx64":
                    if string == "osx64/tcp_bind":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "osx64/reverse_tcp":
                        controlset(string, self.argvlist[2], self.argvlist[3], self.argvlist[0], self.argvlist[1])
                    else:
                        controlset(string, self.argvlist[0], self.argvlist[1])

                    self.control(string)

                elif string[:11] == "freebsd_x86":
                    if string == "freebsd_x86/reverse_tcp2":
                        controlset(string, self.argvlist[3], self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "freebsd_x86/reverse_tcp":
                        controlset(string, self.argvlist[3], self.argvlist[2], self.argvlist[0], self.argvlist[1])             
                    elif string == "freebsd_x86/read":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "freebsd_x86/exec":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])                   
                    elif string == "freebsd_x86/tcp_bind":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    else:
                        controlset(string, self.argvlist[0], self.argvlist[1])
                    self.control(string)

                elif string[:11] == "freebsd_x64":
                    if string == "freebsd_x64/tcp_bind":
                        controlset(string, self.argvlist[0], self.argvlist[1], self.argvlist[2], self.argvlist[3])
                    elif string == "freebsd_x64/reverse_tcp":
                        controlset(string, self.argvlist[2], self.argvlist[3], self.argvlist[0], self.argvlist[1]) 
                    elif string == "freebsd_x64/exec":
                        controlset(string, self.argvlist[0], self.argvlist[1], self.argvlist[2])
                    else:
                        controlset(string, self.argvlist[0], self.argvlist[1])
                    self.control(string)

                elif string[:9] == "linux_arm":
                    if string == "linux_arm/chmod":
                        controlset(string, self.argvlist[0], self.argvlist[1], self.argvlist[2])
                    elif string == "linux_arm/exec":
                        controlset(string, self.argvlist[0], self.argvlist[1], self.argvlist[2])
                    elif string == "linux_arm/reverse_tcp":
                        controlset(string, self.argvlist[2], self.argvlist[3], self.argvlist[0], self.argvlist[1])
                    else:
                        controlset(string, self.argvlist[0], self.argvlist[1])
                    self.control(string)

                elif string[:10] == "linux_mips":
                    if string == "linux_mips/chmod":
                        controlset(string, self.argvlist[0], self.argvlist[1], self.argvlist[2])
                    elif string == "linux_mips/reverse_tcp":
                        controlset(string, self.argvlist[0], self.argvlist[1], self.argvlist[2], self.argvlist[3])
                    elif string == "linux_mips/tcp_bind":
                        controlset(string, self.argvlist[0], self.argvlist[1], self.argvlist[2])
                    else:
                        controlset(string, self.argvlist[0], self.argvlist[1])
                    self.control(string)

                elif string[:7] == "windows":
                    if string == "windows/messagebox":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])
                    elif string == "windows/exec":
                        controlset(string, self.argvlist[1], self.argvlist[0], self.argvlist[2])
                    elif string == "windows/download&execute":
                        controlset(string, self.argvlist[0], self.argvlist[1], self.argvlist[2], self.argvlist[3])
                    elif string == "windows/reverse_tcp":
                        controlset(string, self.argvlist[2], self.argvlist[3], self.argvlist[0], self.argvlist[1])
                    elif string == "windows/tcp_bind":
                        controlset(string, self.argvlist[2], self.argvlist[0], self.argvlist[1])                  
                    self.control(string)

            elif terminal[:8] == "generate":
                from .database.generator import generator
                if string[:7] == "linux86":
                    if string == "linux86/binsh_spawn":
                        self.disassembly = generator("linux86", "binsh_spawn")

                    elif string == "linux86/read":
                        if self.argvlist[2] == "None":
                            print("\nFile name must be declared.\n")
                            self.control(string)
                        self.disassembly = generator("linux86", "read", FILE=self.argvlist[2])

                    elif string == "linux86/exec":
                        if self.argvlist[2] == "None":
                            print("\nCommand must be declared.\n")
                            self.control(string)
                        self.disassembly = generator("linux86", "exec", COMMAND=self.argvlist[2])

                    elif string == "linux86/download&exec":
                        if self.argvlist[2] == "None":
                            print("\nLink must be declared.\n")
                            self.control(string)
                        elif "/" not in self.argvlist[2]:
                            print("\nWrong url format example : 127.0.0.1/X\n") 
                            self.control(string)
                        elif len(self.argvlist[2].split("/")[-1]) != 1:
                            print("\nYour filename must be one lenght ..\n")   
                            self.control(string)

                        if "http" in self.argvlist[2] or "https" in self.argvlist[2] or "www." in self.argvlist:
                            try:
                                edit = self.argvlist[2].replace("http://", "").replace("https://", "").replace("www.", "")
                                self.argvlist[2] = edit
                            except:
                                pass
                        self.disassembly = generator("linux86", "download&exec", URL=self.argvlist[2])

                    elif string == "linux86/chmod":
                        if self.argvlist[2] == "None":
                            print("\nFile name must be declared.\n")
                            self.control(string)
                        self.disassembly = generator("linux86", "chmod", FILE=self.argvlist[2])

                    elif string == "linux86/tcp_bind":
                        if self.argvlist[2] == "None":
                            print("\nPORT must be declared.\n")
                            self.control(string)
                        self.disassembly = generator("linux86", "tcp_bind", port=self.argvlist[2])

                    elif string == "linux86/reverse_tcp":   
                        if self.argvlist[2] == "None" or self.argvlist[3] == "None": 
                            print("\nHost&Port must be declared.\n")
                            self.control(string)
                        self.disassembly = generator("linux86", "reverse_tcp", ip=self.argvlist[3], port=self.argvlist[2])

                elif string[:7] == "linux64":
                    if string == "linux64/binsh_spawn":
                        self.disassembly = generator("linux64", "binsh_spawn")
                    elif string == "linux64/tcp_bind":
                        self.disassembly = generator("linux64", "tcp_bind", port=self.argvlist[2])
                    elif string == "linux64/reverse_tcp":
                        self.disassembly = generator("linux64", "reverse_tcp", ip=self.argvlist[3], port=self.argvlist[2])
                    elif string == "linux64/read":
                        self.disassembly = generator("linux64", "read", FILE=self.argvlist[2])    

                if string[:5] == "linux":
                    if string == "linux/read":
                        if self.argvlist[2] == "None":
                            print("\nFile name must be declared.\n")
                            self.control(string)
                        self.disassembly = generator("linux", "read", FILE=self.argvlist[2])
                    elif string == "linux/binsh_spawn":
                        self.disassembly = generator("linux", "binsh_spawn")
                    elif string == "linux/tcp_bind":
                        self.disassembly = generator("linux", "tcp_bind", port=self.argvlist[2])
                    elif string == "linux/reverse_tcp":
                        self.disassembly = generator("linux", "reverse_tcp", ip=self.argvlist[3], port=self.argvlist[2])

                elif string[:5] == "osx86":
                    if string == "osx86/tcp_bind":
                        self.disassembly = generator("osx86", "tcp_bind", port=self.argvlist[2])
                    elif string == "osx86/binsh_spawn":
                        self.disassembly = generator("osx86", "binsh_spawn")
                    elif string == "osx86/reverse_tcp":
                        self.disassembly = generator("osx86", "reverse_tcp", ip=self.argvlist[3], port=self.argvlist[2])

                elif string[:5] == "osx64":
                    if string == "osx64/binsh_spawn":
                        self.disassembly = generator("osx64", "binsh_spawn")
                    elif string == "osx64/tcp_bind":
                        self.disassembly = generator("osx64", "tcp_bind", port=self.argvlist[2])
                    elif string == "osx64/reverse_tcp":
                        self.disassembly = generator("osx64", "reverse_tcp", ip=self.argvlist[3], port=self.argvlist[2])

                elif string[:11] == "freebsd_x86":
                    if string == "freebsd_x86/binsh_spawn":
                        self.disassembly = generator("freebsdx86", "binsh_spawn")
                    elif string == "freebsd_x86/read":
                        self.disassembly = generator("freebsdx86", "read", FILE=self.argvlist[2])
                    elif string == "freebsd_x86/reverse_tcp":
                        self.disassembly = generator("freebsdx86", "reverse_tcp", ip=self.argvlist[3], port=self.argvlist[2])
                    elif string == "freebsd_x86/reverse_tcp2":
                        self.disassembly = generator("freebsdx86", "reverse_tcp2", ip=self.argvlist[3], port=self.argvlist[2])
                    elif string == "freebsd_x86/exec":
                        self.disassembly = generator("freebsdx86", "exec", COMMAND=self.argvlist[2])
                    elif string == "freebsd_x86/tcp_bind":
                        self.disassembly = generator("freebsdx86", "tcp_bind", port=self.argvlist[2])

                elif string[:11] == "freebsd_x64":
                    if string == "freebsd_x64/binsh_spawn":
                        self.disassembly = generator("freebsdx64", "binsh_spawn")
                    elif string == "freebsd_x64/tcp_bind":
                        self.disassembly = generator("freebsdx64", "tcp_bind", port=self.argvlist[2], PASSWORD=self.argvlist[3])
                    elif string == "freebsd_x64/reverse_tcp":
                        self.disassembly = generator("freebsdx64", "reverse_tcp", ip=self.argvlist[3], port=self.argvlist[2])
                    elif string == "freebsd_x64/exec":
                        self.disassembly = generator("freebsdx64", "exec", COMMAND=self.argvlist[2])

                elif string[:9] == "linux_arm":
                    if string == "linux_arm/chmod":
                        self.disassembly = generator("linux_arm", "chmod", FILE=self.argvlist[2])
                    elif string == "linux_arm/binsh_spawn":
                        self.disassembly = generator("linux_arm", "binsh_spawn")
                    elif string == "linux_arm/reverse_tcp":
                        self.disassembly = generator("linux_arm", "reverse_tcp", ip=self.argvlist[3], port=self.argvlist[2])
                    elif string == "linux_arm/exec":
                        self.disassembly = generator("linux_arm", "exec", COMMAND=self.argvlist[2])    

                elif string[:10] == "linux_mips":
                    if string == "linux_mips/reverse_tcp":
                        self.disassembly = generator("linux_mips", "reverse_tcp", ip=self.argvlist[3], port=self.argvlist[2])
                    elif string == "linux_mips/binsh_spawn":
                        self.disassembly = generator("linux_mips", "binsh_spawn")
                    elif string == "linux_mips/chmod":
                        self.disassembly = generator("linux_mips", "chmod", FILE=self.argvlist[2])
                    elif string == "linux_mips/tcp_bind":
                        self.disassembly = generator("linux_mips", "tcp_bind", port=self.argvlist[2])

                elif string[:7] == "windows":
                    if string == "windows/messagebox":
                        self.disassembly = generator("windows", "messagebox", MESSAGE=self.argvlist[2])
                    elif string == "windows/download&execute":
                        self.disassembly = generator("windows", "downloandandexecute", URL=self.argvlist[2], FILENAME=self.argvlist[3])
                    elif string == "windows/exec":
                        self.disassembly = generator("windows", "exec", COMMAND=self.argvlist[2])
                    elif string == "windows/reverse_tcp":
                        self.disassembly = generator("windows", "reverse_tcp", ip=self.argvlist[3], port=self.argvlist[2])                  
                    elif string == "windows/tcp_bind":
                        self.disassembly = generator("windows", "tcp_bind", port=self.argvlist[2])

                elif string[:10] == "solarisx86":                   
                    if string == "solarisx86/binsh_spawn":
                        self.disassembly = generator("solarisx86", "binsh_spawn")
                    elif string == "solarisx86/read":
                        if self.argvlist[2] == "None":
                            print("\nFile name must be declared.\n")
                            self.control(string)
                        self.disassembly = generator("solarisx86", "read", FILE=self.argvlist[2])
                    elif string == "solarisx86/reverse_tcp":
                        self.disassembly = generator("solarisx86", "reverse_tcp", ip=self.argvlist[3], port=self.argvlist[2])
                    elif string == "solarisx86/tcp_bind":
                        self.disassembly = generator("solarisx86", "tcp_bind", port=self.argvlist[2])

                if self.argvlist[0] == "x86/xor_b3m":
                    from .encoders.shellcode.xor_b3m import prestart
                    if self.argvlist[1] == "None":
                        self.argvlist[1] = 1
                    elif self.argvlist[1] == 0:
                        self.argvlist[1] = 1
                    self.disassembly = prestart(self.disassembly.replace("\\x", ""), int(self.argvlist[1]))

                elif self.argvlist[0] == "x86/xor":
                    from .encoders.shellcode.xor import prestart
                    if self.argvlist[1] == "None":
                        self.argvlist[1] = 1
                    elif self.argvlist[1] == 0:
                        self.argvlist[1] = 1
                    self.disassembly = prestart(self.disassembly.replace("\\x", ""), int(self.argvlist[1]))

                else:
                    self.disassembly = self.disassembly 

                # print "\n"+"Shellcode Lenght : %d" % len(str(bytearray(self.disassembly.replace("\\x", "").decode("hex"))))
                B3mB4m.prettyout(self.disassembly)
                self.control(string)

            elif terminal[:6] == "output":
                if self.disassembly == "None":
                    print("Please generate shellcode before save it.")
                    self.control(string)   

                # I'm not sure about this option, should I get this option with params 
                # Or directly inputs ? ..
                if terminal[7:10].lower() == "exe":
                    # Will be add missing parts ..
                    if "linux86" in terminal.lower():
                        OS = "linux86"
                    elif "linux64" in terminal.lower():
                        OS = "linux64"
                    elif "windows" in terminal.lower():
                        OS = "windows"
                    elif "freebsdx86" in terminal.lower():
                        OS = "freebsdx86"
                    elif "freebsdx64" in terminal.lower():
                        OS = "freebsdx64"
                    elif "openbsdx86" in terminal.lower():
                        OS = "openbsdx86"
                    elif "solarisx86" in terminal.lower():
                        OS = "solarisx86"
                    elif "linuxpowerpc" in terminal.lower():
                        OS = "linuxpowerpc"
                    elif "openbsdpowerpc" in terminal.lower():
                        OS = "openbsdpowerpc"           
                    elif "linuxsparc" in terminal.lower():
                        OS = "linuxsparc"
                    elif "freebsdsparc" in terminal.lower():
                        OS = "freebsdsparc"
                    elif "openbsdsparc" in terminal.lower():
                        OS = "openbsdsparc"
                    elif "solarissparc" in terminal.lower():
                        OS = "solarissparc"
                    elif "linuxarm" in terminal.lower():
                        OS = "linuxarm"
                    elif "freebsdarm" in terminal.lower():
                        OS = "freebsdarm"
                    elif "openbsdarm" in terminal.lower():
                        OS = "openbsdarm"
                    else:
                        OS = None

                    from .Outputs.exe import ExeFile
                    ExeFile(self.disassembly, OS)
                    self.control(string)

                elif terminal[7:10].lower() == "c++" or terminal[7:10].lower() == "cpp":
                    from .Outputs.Cplusplus import CplusplusFile
                    if "windows" in string:
                        CplusplusFile(self.disassembly, True)
                    else:
                        CplusplusFile(self.disassembly)

                elif terminal[7:8].lower() == "c":
                    if "windows" in string:
                        from .Outputs.Cplusplus import CplusplusFile
                        CplusplusFile(self.disassembly, True)
                    else:
                        from .Outputs.C import CFile
                        CFile(self.disassembly)                

                elif terminal[7:9].lower() == "py" or terminal[7:13].lower() == "python": 
                    from .Outputs.python import PyFile
                    PyFile(self.disassembly)

                elif terminal[7:10].lower() == "txt":
                    from .Outputs.txt import TxtFile
                    TxtFile(self.disassembly)  

                else:
                    print(bcolors.RED + bcolors.BOLD + "[-] Unknown output type: {0}".format(terminal) + bcolors.ENDC)
                self.control(string)                   

            elif terminal[:5] == "clear":
                B3mB4m.clean()
                self.control(string)

            elif terminal[:2].lower() == "ip":
                B3mB4m.IP()
                self.control(string)

            elif terminal[:13] == "show encoders":
                from .core.lists import encoderlist
                encoderlist()
                self.control(string)

            elif terminal[:5] == "disas":
                B3mB4m().startdisas( self.disassembly, string)
                self.control(string)

            else:
                if not terminal:
                    self.control(string)
                else:
                    print(bcolors.RED + bcolors.BOLD + "[-] Unknown command: {0}".format(terminal) + bcolors.ENDC)
                    self.control(string)
