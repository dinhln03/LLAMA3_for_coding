"""
input: image
output: little squares with faces
"""

import face_recognition
image = face_recognition.load_image_file("people.png")
face_locations = face_recognition.face_locations(image)

print(face_locations)