"""
strings and logic related to composing notifications
"""

HELLO_STATUS = "Hello! I'm Vaccination Notifier"
HELLO_MESSAGE = (
    "Hello there!\n"
    "\n"
    "I'm Vaccination Notifier. This is just a message to let you know I'm running and "
    "to test our notification configuration. I'll check for changes to your "
    "vaccination status once every {delay} minutes---unless I crash! Every now and then, "
    "you should probably check on me to make sure nothing has gone wrong.\n"
    "\n"
    "Love,\n"
    "Vaccination Notifier"
)

def hello_message(delay):
    return (HELLO_STATUS, HELLO_MESSAGE.format(delay=delay))

UPDATE_STATUS = "Vaccination update detected"
UPDATE_MESSAGE = (
    "Hello there!\n"
    "\n"
    "I noticed that your vaccination results page was updated recently. Here's "
    "a summary of the update:\n"
    "Health Facility:{facility}\n"
    "Vaccination Location:{location}\n"
    "Date:{date}\n"
    "Time:{time}\n"
    "\n"
    "Love,\n"
    "Vaccination Notifier"
)

def update_message(dict):
    facility = dict['Health Facility:']
    location = dict['Vaccination Location:']
    date = dict['Date:']
    time = dict['Time:']
    return (UPDATE_STATUS, 
        UPDATE_MESSAGE.format(facility=facility, location=location, date=date, time=time))