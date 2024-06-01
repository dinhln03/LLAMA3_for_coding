from model import *
from data import *
from keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

# test_dir = "data/membrane/test"
# test_datagen = ImageDataGenerator(rescale=1./255)

# test_generator = test_datagen.flow_from_directory(
#         test_dir,
#         target_size=(256, 256),
#         color_mode="grayscale",
#         batch_size=1)
# test_path = "data/membrane/test"
# image_datagen = ImageDataGenerator(**data_gen_args)
# image_generator = image_datagen.flow_from_directory(
#         test_path,
#         class_mode = None,
#         color_mode = "grayscale",
#         target_size = (256,256),
#         batch_size = 1,
#         save_to_dir = None,
#         seed = 2)

# filenames = test_generator.filenames
# nb_samples = len(filenames)
# print(nb_samples)

# predict = model.predict_generator(test_generator,steps = nb_samples)

# testGene = testGenerator("data/membrane/test")
# filenames = testGene.filenames
# nb_samples = len(filenames)
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/membrane/test",results)


test_path = "data/membrane/test"
target_size = (256,256) 
flag_multi_class = False
img = io.imread(os.path.join(test_path,"%d.png"%30),as_gray = True)
img = img / 255
img = trans.resize(img,target_size)
img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
img = np.reshape(img,(1,)+img.shape)
results = model.predict(img)
print(results)
COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

saveResult("data/membrane/test",results)

#io.imsave(os.path.join(save_path,"%d_predict.png"%31),results)



# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene,31)
# saveResult("data/membrane/test",results)