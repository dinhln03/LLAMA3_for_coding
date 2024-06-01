# simple example demonstrating how to control a Tello using your keyboard.
# For a more fully featured example see manual-control-pygame.py
# 
# Use W, A, S, D for moving, E, Q for rotating and R, F for going up and down.
# When starting the script the Tello will takeoff, pressing ESC makes it land
#  and the script exit.

# 简单的演示如何用键盘控制Tello
# 欲使用全手动控制请查看 manual-control-pygame.py
#
# W, A, S, D 移动， E, Q 转向，R、F上升与下降.
# 开始运行程序时Tello会自动起飞，按ESC键降落
# 并且程序会退出

from djitellopy import Tello
import cv2, math, time

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()
height, width, _ = frame_read.frame.shape
# tello.takeoff()
nSnap   = 0
# w       = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# h       = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
w = width
h= height
folder = "."
name = "snapshot"
fileName    = "%s/%s_%d_%d_" %(folder, name, w, h)
while True:
    # In reality you want to display frames in a seperate thread. Otherwise
    #  they will freeze while the drone moves.
    # 在实际开发里请在另一个线程中显示摄像头画面，否则画面会在无人机移动时静止
    img = frame_read.frame
    cv2.imshow("drone", img)
    # height, width, _ = frame_read.frame.shape
    # video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    key = cv2.waitKey(1) & 0xff
    if key == 27: # ESC
        break
    elif key == ord('w'):
        tello.move_forward(30)
    elif key == ord('s'):
        tello.move_back(30)
    elif key == ord('a'):
        tello.move_left(30)
    elif key == ord('d'):
        tello.move_right(30)
    elif key == ord('e'):
        tello.rotate_clockwise(30)
    elif key == ord('q'):
        tello.rotate_counter_clockwise(30)
    elif key == ord('r'):
        tello.send_command_with_return('downvision 0')
        frame_read = tello.get_frame_read()
    elif key == ord('f'):
        tello.send_command_with_return('downvision 1')
        frame_read = tello.get_frame_read()
    
    elif key == ord(' '):
        print("Saving image ", nSnap)
        cv2.imwrite("%s%d-jpg"%(fileName, nSnap), img)
        nSnap += 1

# tello.land()
