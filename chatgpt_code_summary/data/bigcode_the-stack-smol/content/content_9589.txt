from flask import Flask, jsonify, request, render_template, redirect, url_for
import datetime
from alarm import AlarmService

app = Flask(__name__, static_url_path='/static')
app.config.from_pyfile('config.py')

# status file for motion detection
motionfile = "motion.txt"

# enable temperature sensor (if library is available)
try:
    import w1thermsensor
    temperature = w1thermsensor.W1ThermSensor()
except ImportError:
    temperature = None
except:
    temperature = None


@app.route('/')
def index():
    return render_template('/index.html',
                           webcam = app.config["WEBCAM"],
                           sensor = app.config["SENSOR"],
                           temperature = app.config["TEMPERATURE"],
                           alarm = app.config["ALARM"],
                           # alarmOn = True,
                           switches = {'A': app.config["SWITCH_A"],
                           'B': app.config["SWITCH_B"],
                           'C': app.config["SWITCH_C"]})

@app.route("/status", methods=["GET"])
def status():
    import os, time
    now = datetime.datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    details = {'time': timeString}   # update time
    # read temperature (if sensor is available)
    if temperature and app.config["TEMPERATURE"]:
        tempcelsius = temperature.get_temperature()
        details.update({'temperature': tempcelsius})
    # read status of PIR sensor
    if os.path.exists(motionfile):
        details['motion'] = time.ctime(os.path.getmtime(motionfile))
    # take snapshot of webcam (max 1 per minute)
    if app.config["WEBCAM"]:
        import sh, os
        timeString = now.strftime("%Y%m%d_%H%M")
        snapshot = "snapshot-%s.jpg" % timeString
        snapshot = os.path.join("static", "webcam", snapshot)
        details['webcam'] = '/'+snapshot
        snapshot = os.path.abspath(snapshot)
        if not os.path.exists(snapshot):
            sh.fswebcam("--title", "Home", "--save", snapshot)
    return jsonify(**details)

@app.route("/alarm/<on>", methods=["POST"])
def alarm(on):
    details  = {}
    if turnAlarm(on=="on"):
        now = datetime.datetime.now()
        timeString = now.strftime("%Y-%m-%d %H:%M")
        details.update({'time': timeString})
    return jsonify(**details)

def turnAlarm(on):
    """write command into file - pass enable/disable command"""
    print ("turn alarm on? %s" % on)
    enable = on
    disable = not on
    alarm = AlarmService()
    alarm.load()
    alarm.save(enable, disable)
    return True

@app.route("/turn/<switch>", methods=["POST"])
def turn(switch):
    on = request.form['cmd']=="on"
    details  = {}
    if turnSwitch(switch, on):
        now = datetime.datetime.now()
        timeString = now.strftime("%Y-%m-%d %H:%M")
        details.update({'time': timeString})
    return jsonify(**details)

def turnSwitch(switch, on=False):
    from elro import  RemoteSwitch
    devices = { 'A': 1, 'B': 2, 'C': 4, 'D': 8, 'E':16 }
    device=devices.get(switch, 0)
    if not device:
        return False
    # print "turn switch '%s' to '%s' - device#%d"  % (switch, "on" if on else "off", device)
    device = RemoteSwitch(device,
                          key=app.config["ELRO_KEY"],
                          pin=app.config["ELRO_PIN"])
    if on:
        device.switchOn()
    else:
        device.switchOff()
    return True


if __name__ == "__main__":
    app.run(host='0.0.0.0',
            port=app.config["HTTP_PORT"],
            debug=app.config["DEBUG"])
