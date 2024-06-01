from subprocess import check_output

def isrunning(processName):
    tasklist = check_output('tasklist', shell=False)
    tasklist = str(tasklist)

    return(processName in tasklist)