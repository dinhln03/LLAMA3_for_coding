import cv2
import pose_detection as pose_d

pose_model = pose_d.load_pose_model('pre_trained\AFLW2000.pkl')

def detect_face(img_PATH, model_PATH):
  # Load the cascade
  face_cascade = cv2.CascadeClassifier(model_PATH)
  # Read the input image
  img = cv2.imread(img_PATH)
  # Convert into grayscale 
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Detect faces
  faces = face_cascade.detectMultiScale(gray, 1.1, 4)

  if len(faces) > 1:
    print('Multiple faces detected')
    return False
  elif len(faces) < 1:
    print('No faces detected')
    return False
  # Draw rectangle around the faces
  for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
  # Display the output
  #cv2_imshow(img)
  cv2.waitKey()
  return True # TO DO may want to return face at some point as well

def detect_face_video(pose_model):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # To capture video from webcam. 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # To use a video file as input 
    # cap = cv2.VideoCapture('filename.mp4')
    
    while True:
        # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Get pose estimate
        yaw, pitch, roll = pose_d.run_pose_detection(pose_model, pose_d.load_img(img))
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #draw pose label
        img = pose_d.draw_labels(yaw, pitch, roll, img)
        # Display
        cv2.imshow('img', img)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_face_video(pose_model)