import picamera
from time import sleep

camera = picamera.PiCamera()
camera.capture('image.jpg')

#camera.resolution = (640, 480) #max resolution is 2592 x 1944 for stills,
                             #                  1920 x 1080 (<15fps) for video
#camera.framerate = 45
camera.vflip = True

camera.start_recording('video.h264')
camera.image_effect = 'negative'
for j in range (10):
   for i in range(200):
      camera.annotate_text = "everyday we stray further from God's light"
      camera.annotate_text_size = abs(i-100)+6
      camera.brightness = abs(i-100)
      sleep(0.001)
camera.stop_recording()
