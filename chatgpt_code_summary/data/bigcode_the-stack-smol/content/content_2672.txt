from config.configure import Configure

conf = Configure()

conf.model_name = 'vgg16.h5'
conf.classes = ['no_breads', 'breads']
conf.no_breads_path = './dataset/data/pool/no_breads/*'
conf.breads_path = './dataset/data/pool/breads/*'
# conf.baked_breads_path = './dataset/data/pool/breads/*'

conf.lr = 1e-4
conf.momentum = 0.9
conf.batch_size = 20
conf.epochs = 20
conf.image_size = 224
