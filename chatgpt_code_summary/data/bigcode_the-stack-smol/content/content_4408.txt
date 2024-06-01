from math import ceil
from time import sleep

from config import *
from servo import *
from camera import *
from pid import *



# Init
servo1 = Servo(2)
servo2 = Servo(3)

servo_count = 0


#servo1.setAngle(90)
#sleep(1)
#servo2.setAngle(90)
#sleep(1)


camera = Camera()
PID = PID()



# Loop
while True:
    data = camera.getBallPos()
	
    #data = None
	
    if not (data[0] and data[1]):
	print("No ball found")
	
    else:
	
	# Calculate PID
	move = PID.calculate(data[0], data[1])

	# Print result
        #print("x", round(move[0]), "y", round(move[1]), round(move[2]*600))
    
        # Move servos
        #move_x = ( (move[0] - conf.cam_x_min) / (conf.cam_x_max - conf.cam_x_min) ) * (conf.servo_max - conf.servo_min) + conf.servo_min
        #move_y = ( (move[1] - conf.cam_y_min) / (conf.cam_y_max - conf.cam_y_min) ) * (conf.servo_max - conf.servo_min) + conf.servo_min
        
        move_x = ((move[0])/2)+90
        move_y = ((-move[1])/2)+90
        
        print(move_x, move_y)
        
        servo_count += 1
        
        if servo_count > 3:
            servo1.setAngle(move_x)
            servo2.setAngle(move_y)
            servo_count = 0
        
        
        
	#servo.moveTo(2, move[1])
        
        
        
        
    # Show frame
    cv2.imshow("Frame", data[2])
    key = cv2.waitKey(1) & 0xFF

    # Stop if 'q' is pressed
    if key == ord("q"):
        servo1.stop()
        servo2.stop()
        camera.stop()
            

      



