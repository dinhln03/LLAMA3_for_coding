import cv2
import numpy as np

import definitions
from playgrounds.core.features import Feature
from playgrounds.keras_models.features.girl_boy.workers.custom_workers import CustomWorker1
from playgrounds.opencv import face_detection
from playgrounds.utilities import opencv_utilities


class GenderClassifier(Feature):

    def __init__(self):
        super().__init__()
        self.workers = {
            "custom1": CustomWorker1()
        }

    def runFeature(self, worker, inputData, inType ="image"):
        self.worker = self.workers.get(worker)
        func = self.inputTypes.get(inType)
        func(worker, inputData)


    def trainFeature(self, worker):
        self.worker = self.workers.get(worker)
        self.worker.train()


    def runOnVideo(self, worker, inputData):
        self.worker = self.workers.get(worker)
        self.worker.buildModel()
        cap = cv2.VideoCapture(inputData)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path = definitions.ROOT_DIR + "/outputs/tiny_yolo/" + opencv_utilities.getFileNameFromPath(inputData)
        out = cv2.VideoWriter(path, fourcc, 20.0, (854, 356))
        while cap.isOpened():
            ret, frame = cap.read()
            self.getGender(frame)
            # out.write(frame)
            cv2.imshow("V", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def getGender(self, img):
        faces = opencv_utilities.getFaceFromImage(img)
        for (x, y, width, height) in faces:
            if x > 40 and y > 40:
                x = x-40
                y = y-40
                width += 40
                height += 40
            crop_img = img[y:y + height, x:x + width]
            crop_img = cv2.resize(crop_img, (64, 64))
            crop_img = crop_img / 255
            crop_img = np.expand_dims(crop_img, axis=0)

            text = self.worker.predict(crop_img)
            if text > 0.6:
                text = "Man"
            elif text < 0.6:
                text = "Woman"
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (x, y), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
