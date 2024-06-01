from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from Logger.app_logger import  App_logger
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix




class Training:

    def __init__(self,train_path,test_path,val_path):
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
        self.log_object = App_logger()

    def train(self):
        self.log_object.log(self.file_object,"Entered in to train method in Training class.Training started")
        try:
            x_train = []

            for folder in os.listdir(self.train_path):

                sub_path = self.train_path + "/" + folder

                for img in os.listdir(sub_path):
                    image_path = sub_path + "/" + img
                    img_arr = cv2.imread(image_path)
                    if img_arr is None:
                        os.remove(image_path)
                        continue
                    elif img_arr.shape[0] < 224:
                        os.remove(image_path)
                        continue
                    else:
                        img_arr = cv2.resize(img_arr, (224, 224))
                        x_train.append(img_arr)

            x_test = []

            for folder in os.listdir(self.test_path):

                sub_path = self.test_path + "/" + folder

                for img in os.listdir(sub_path):
                    image_path = sub_path + "/" + img

                    img_arr = cv2.imread(image_path)
                    if img_arr is None:
                        os.remove(image_path)
                        continue
                    elif img_arr.shape[0] < 224:
                        os.remove(image_path)
                        continue
                    else:

                        img_arr = cv2.resize(img_arr, (224, 224))

                        x_test.append(img_arr)


            x_val = []

            for folder in os.listdir(self.val_path):

                sub_path = self.val_path + "/" + folder

                for img in os.listdir(sub_path):
                    image_path = sub_path + "/" + img
                    img_arr = cv2.imread(image_path)
                    if img_arr is None:
                        os.remove(image_path)
                        continue
                    elif img_arr.shape[0] < 224:
                        os.remove(image_path)
                        continue
                    else:
                        img_arr = cv2.resize(img_arr, (224, 224))
                        x_val.append(img_arr)
            self.log_object.log(self.file_object, "Entered in to train method in Training class.train,test,val split successfull")

            train_x = np.array(x_train) / 255.0
            test_x = np.array(x_test) / 255.0
            val_x = np.array(x_val) / 255.0

            train_datagen = ImageDataGenerator(rescale=1. / 255)
            test_datagen = ImageDataGenerator(rescale=1. / 255)
            val_datagen = ImageDataGenerator(rescale=1. / 255)

            training_set = train_datagen.flow_from_directory(self.train_path,
                                                             target_size=(224, 224),
                                                             batch_size=32,
                                                             class_mode='sparse')
            test_set = test_datagen.flow_from_directory(self.test_path,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode='sparse')
            val_set = val_datagen.flow_from_directory(self.val_path,
                                                      target_size=(224, 224),
                                                      batch_size=32,
                                                      class_mode='sparse')

            train_y = training_set.classes
            test_y = test_set.classes
            val_y = val_set.classes

            IMAGE_SIZE = [224, 224]

            vgg = VGG19(input_shape= IMAGE_SIZE + [3],weights='imagenet',include_top=False)
            self.log_object.log(self.file_object, "Entered in to train method in Training class. Model successfully initialized")

            for layer in vgg.layers:
                layer.trainable = False

            x = Flatten() (vgg.output)

            prediction = Dense(5 ,activation='softmax') (x)
            model = Model(inputs=vgg.input,outputs = prediction)
            model.summary()

            model.compile(loss = 'sparse_categorical_crossentropy',
                          optimizer='adam',metrics=['accuracy'])
            self.log_object.log(self.file_object, "Entered in to train method in Training class.Model compile successfull")
            file_path = 'vgg19_model/checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5'
            self.log_object.log(self.file_object,"check point directory created")
            check_point = ModelCheckpoint(file_path,monitor='val_accuracy', verbose=1,save_best_only=True, mode='max')
            start = datetime.now()
            self.log_object.log(self.file_object, f"Entered in to train method in Training class.Training start time {start}")
            history = model.fit(train_x,train_y,
                      validation_data= (val_x,val_y),
                      epochs=20,
                      callbacks = [check_point],
                      batch_size=64, shuffle=True)

            duration = datetime.now() - start
            self.log_object.log(self.file_object, f"Entered in to train method in Training class.Total time taken is {duration}")

            model.save('mech_tools_model.h5')
            self.log_object.log(self.file_object, f"Entered in to train method in Training class.model saved successfully")



            # accuracies
            plt.plot(history.history['accuracy'], label='train acc')
            plt.plot(history.history['val_accuracy'], label='val acc')
            plt.legend()
            plt.savefig('vgg-acc-rps-1.png')

            # loss
            plt.plot(history.history['loss'], label='train loss')
            plt.plot(history.history['val_loss'], label='val loss')
            plt.legend()
            plt.savefig('vgg-loss-rps-1.png')

            self.log_object.log(self.file_object, "Entered in to train method in Training class.model evaluation started")
            model.evaluate(test_x, test_y, batch_size=32)

            # predict
            y_pred = model.predict(test_x)
            y_pred = np.argmax(y_pred, axis=1)
            self.log_object.log(self.file_object, f"Entered in to train method in Training class.classification report {classification_report(y_pred, test_y)}")
            self.log_object.log(self.file_object, f"Entered in to train method in Training class.confusion matrix is{confusion_matrix(y_pred, test_y)}")
        except Exception as e:
            # logging the unsuccessful Training
            self.log_object.log(self.file_object, 'Unsuccessful End of Training')
            self.log_object.log(self.file_object,f"exception occured.exception is {e}")
            raise Exception
        self.file_object.close()

if __name__ == "__main__":
    train_path = "final_dataset/train"
    test_path = "final_dataset/test"
    val_path = "final_dataset/val"
    train_model = Training(train_path, test_path, val_path)
    train_model.train()