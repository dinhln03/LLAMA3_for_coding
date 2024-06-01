import cv2

cap = cv2.VideoCapture(1)
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # while True:
    #     ret, frame = cap.read()
    #
    #     # Our operations on the frame come here
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #     try:
    #         number = len(faces)
    #         size = [faces[0][2], faces[0][3]]
    #         position = [faces[0][0], faces[0][1]]
    #         break
    #     except:
    #         a = 1
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # print(number)
    # print(size)
    # print(position)
    #print(len(faces))
    # Display the resulting frame
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = frame[y:y+h, x:x+w]



    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
