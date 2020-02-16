from MobileNet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras.utils import multi_gpu_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
import random
import keras

random.seed(10)  ###

###
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7" # 使用编号为0号的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # 每个GPU现存上届控制在90%以内
session = tf.Session(config=config)

# 设置session ###
KTF.set_session(session)

save_weight_file= './model_weights/'
weight_path = './model_weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5'
# weight_path = './model_weight/first_try.h5'
img_width, img_height = 224,224

print('2222222')

base_model = MobileNetV2(input_shape=(img_width, img_height,3),
				alpha=0.5,
				include_top=True,
				weights= None
				)
				
base_model.load_weights(weight_path)


base_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


x = base_model.get_layer('global_average_pooling2d_1').output

preds = Dense(2, activation='softmax')(x)

mobilenet_model = Model(inputs = base_model.input, outputs=preds)
# mobilenet_model = multi_gpu_model(mobilenet_model, gpus=2)
# mobilenet_model.load_weights(weight_path)



# 模型编译
mobilenet_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# print(mobilenet_model.summary())

print('1111111111111')

train_datagen = ImageDataGenerator(
		rotation_range=40, # 是一个0~180的度数，用来指定随机选择图片的角度。
		width_shift_range=0.2, # 用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比例。
		height_shift_range=0.2,
		rescale=1./255, # 值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。
		shear_range=0.2, # 是用来进行剪切变换的程度。
		zoom_range=0.2, # 用来进行随机的放大。
		horizontal_flip=True, # 随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候。
		fill_mode='nearest') # 用来指定当需要进行像素填充，如旋转，水平和竖直位移时，如何填充新出现的像素。

train_generator = train_datagen.flow_from_directory(
		'/home/cym/Datasets/photos/train',
		target_size=(224,224),
		batch_size=32,
		class_mode='categorical')
# print(base_model)

early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                              patience=2, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=False)


test_datagen = ImageDataGenerator(rescale=1./255)	
validation_generator = test_datagen.flow_from_directory(
        '/home/cym/Datasets/photos/val',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')	

mobilenet_model.fit_generator(
		train_generator,
		samples_per_epoch = 769,  # 所有照片数量 1071
		nb_epoch = 10,
		validation_data = validation_generator,
		nb_val_samples = 109, # 所有照片数量 466
		callbacks = [early_stopping]
		)
mobilenet_model.save_weights(save_weight_file+'first_try_1.h5')

# mobilenet_model.fit_generator(
# 		train_generator,
# 		samples_per_epoch = 769,
# 		nb_epoch = 20,
# 		validation_data = validation_generator,
# 		nb_val_samples = 109,
# 		callbacks = [early_stopping]
# 		)
# mobilenet_model.save_weights(save_weight_file+'first_try_5.h5')
