from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
import glob

test_data_dir = "data/test"
batch_size = 12
img_width, img_height = 224, 224

def meanSubtraction(x):
	x = x.astype(np.float32)
	means = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
	x = x - means
	return x

datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=meanSubtraction)
validation_generator = datagen.flow_from_directory(test_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode="categorical")

print("Loading model...")
model = load_model("fine-tuned-model.h5")
print("Fine tuned model loaded...")

pillDictionary = validation_generator.class_indices
print(pillDictionary)

total = 0
correct = 0
with open("log-test-dataset.txt", "w") as logFile:
	logFile.write("Pill Image\tReal Class\tPredicted Class\n----------\t----------\t---------------\n")
	for imagePath in glob.glob(os.path.join(test_data_dir, "*/*.jpg")):
		print "\n\nTrying to classificate " + imagePath
		
		image = load_img(imagePath, target_size=(img_width, img_height))
		image = img_to_array(image)
		image[:,:,0] -= 123.68
		image[:,:,1] -= 116.779
		image[:,:,2] -= 103.939
		image = image / 255
		image = np.expand_dims(image, axis=0)
		prediction = model.predict(image)

		ID = prediction.argmax()
		mapping = {v: k for k, v in pillDictionary.items()}
		label = mapping[ID]

		total += 1
		pathComponents = imagePath.split('/')
		if pathComponents[2] == label:
			correct += 1
		else:
			print(prediction)

		print("Predicted Class ID: {}, Label: {}, Real Label: {}, OK? {}".format(ID, label, pathComponents[2], pathComponents[2] == label))
		logFile.write('{}\t{}\t{}\n'.format(pathComponents[3], pathComponents[2], label))

	print("Done!")
	finalMsg = "Total images classified: {}, total correct: {}".format(total, correct)
	print(finalMsg)
	logFile.write(finalMsg)
