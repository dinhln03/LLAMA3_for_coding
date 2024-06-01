from project import app, socketio


if __name__ == "__main__":
    print('Running BabyMonitorSoS \n')
    socketio.run(app)
