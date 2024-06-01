from common import num_range_scale
from neurons_engine import neurons_request, neurons_blocking_read, neurons_blocking_read

INQUIRE_ID = 0x00

SHAKE_ID = 0x01
ACC_X_ID = 0x02
ACC_Y_ID = 0x03
ACC_Z_ID = 0x04
GYRO_X_ID = 0x05
GYRO_Y_ID = 0x06
GYRO_Z_ID = 0x07
PITCH_ID = 0x08
ROLL_ID = 0x09
ROTATE_Z_ID = 0x0a
ROTATE_X_ID = 0x0b
ROTATE_Y_ID = 0x0c

TILTED_THRESHOLD = 20

SHAKED_THRESHOLD = {"light": 10, "usual": 30, "strong": 50}

ACC_SHIFT = 9.8
FACE_STATUC_THRESHOLD = 9.4 #// 9.4 = 9.8 * cos(15)

def get_acceleration(axis, index = 1):
    if not isinstance(index, (int, float)):
        return 0

    if axis == 'x':
        value = neurons_blocking_read("m_motion_sensor", "get_acc_x", (INQUIRE_ID, ACC_X_ID), index)
    elif axis == 'y':
        value = neurons_blocking_read("m_motion_sensor", "get_acc_y", (INQUIRE_ID, ACC_Y_ID), index)
    elif axis == 'z':
        value = neurons_blocking_read("m_motion_sensor", "get_acc_z", (INQUIRE_ID, ACC_Z_ID), index)
    else:
        return 0

    if value != None:
        return value[0] * ACC_SHIFT
    else:
        return 0

def get_gyroscope(axis, index = 1):
    if not isinstance(index, (int, float)):
        return 0

    if axis == 'x':
        value = neurons_blocking_read("m_motion_sensor", "get_gyr_x", (INQUIRE_ID, GYRO_X_ID), index)
    elif axis == 'y':
        value = neurons_blocking_read("m_motion_sensor", "get_gyr_y", (INQUIRE_ID, GYRO_Y_ID), index)
    elif axis == 'z':
        value = neurons_blocking_read("m_motion_sensor", "get_gyr_z", (INQUIRE_ID, GYRO_Z_ID), index)
    else:
        return 0

    if value != None:
        return value[0]
    else:
        return 0 

def get_rotation(axis, index = 1):
    if not isinstance(index, (int, float)):
        return 0

    if axis == 'x':
        value = neurons_blocking_read("m_motion_sensor", "get_rotate_x", (INQUIRE_ID, ROTATE_X_ID), index)
    elif axis == 'y':
        value = neurons_blocking_read("m_motion_sensor", "get_rotate_y", (INQUIRE_ID, ROTATE_Y_ID), index)
    elif axis == 'z':
        value = neurons_blocking_read("m_motion_sensor", "get_rotate_z", (INQUIRE_ID, ROTATE_Z_ID), index)
    else:
        return 0
        
    if value != None:
        return value[0]
    else:
        return 0 

def reset_rotation(axis = "all", index = 1):
    if not isinstance(index, (int, float)):
        return

    if axis == 'x':
        neurons_request("m_motion_sensor", "reset_x", (), index)
    elif axis == 'y':
        neurons_request("m_motion_sensor", "reset_y", (), index)
    elif axis == 'z':
        neurons_request("m_motion_sensor", "reset_z", (), index)
    elif axis == 'all':
        neurons_request("m_motion_sensor", "reset_x", (), index)
        neurons_request("m_motion_sensor", "reset_y", (), index)
        neurons_request("m_motion_sensor", "reset_z", (), index)

def is_shaked(level = "usual", index = 1):
    if not isinstance(index, (int, float)):
        return False

    value = neurons_blocking_read("m_motion_sensor", "get_shake_strength", (INQUIRE_ID, SHAKE_ID), index)
    if level in SHAKED_THRESHOLD:
        thres = SHAKED_THRESHOLD[level]
    else:
        thres = SHAKED_THRESHOLD["usual"]

    if value != None:
        return bool(value[0] > thres)
    else:
        return False
    
def get_shake_strength(index = 1):
    if not isinstance(index, (int, float)):
        return 0

    value = neurons_blocking_read("m_motion_sensor", "get_shake_strength", (INQUIRE_ID, SHAKE_ID), index)

    if value != None:
        return round(value[0], 1)
    else:
        return 0 

def get_pitch(index = 1):
    if not isinstance(index, (int, float)):
        return 0

    value = neurons_blocking_read("m_motion_sensor", "get_pitch", (INQUIRE_ID, PITCH_ID), index)

    if value != None:
        return value[0]
    else:
        return 0 

def get_roll(index = 1):
    if not isinstance(index, (int, float)):
        return 0

    value = neurons_blocking_read("m_motion_sensor", "get_roll", (INQUIRE_ID, ROLL_ID), index)

    if value != None:
        return value[0]
    else:
        return 0 


def is_tilted_left(index = 1):
    value = get_roll(index)

    return bool(value < -TILTED_THRESHOLD) 

def is_tilted_right(index = 1):
    value = get_roll(index)

    return bool(value > TILTED_THRESHOLD) 

def is_tilted_forward(index = 1):
    value = get_pitch(index)

    return bool(value < -TILTED_THRESHOLD) 

def is_tilted_backward(index = 1):
    value = get_pitch(index)

    return bool(value > TILTED_THRESHOLD)

def is_face_up(index = 1):
    acc_z = get_acceleration('z', index)
    if acc_z < -FACE_STATUC_THRESHOLD:
        return True
    else:
        return False
def is_face_down(index = 1):
    acc_z = get_acceleration('z', index)
    if acc_z > FACE_STATUC_THRESHOLD:
        return True
    else:
        return False

def is_upright(index = 1):
    acc_y = get_acceleration('y', index)
    if acc_y > FACE_STATUC_THRESHOLD or acc_y < -FACE_STATUC_THRESHOLD:
        return True
    else:
        return False
