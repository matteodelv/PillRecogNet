from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
import glob

import settings as s
import utils as u


# File paths
log_path = os.path.join(s.results_dir, 'log_test_dataset.txt')

# Load fine tuned model
model = load_model(s.fine_tuned_model_path)
print("Fine tuned model loaded...")

# Prepare test images for prediction
datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=u.meanSubtraction)
test_generator = datagen.flow_from_directory(s.test_data_dir, target_size=(s.img_width, s.img_height), batch_size=s.test_batch_size, class_mode="categorical")

pillDictionary = test_generator.class_indices
print(pillDictionary)

total = 0
correct = 0

# Create log file for reference about predictions
with open(log_path, "w") as logFile:
	logFile.write("Pill Image\tReal Class\tPredicted Class\n----------\t----------\t---------------\n")
	for imagePath in glob.glob(os.path.join(s.test_data_dir, "*/*.jpg")):
		print("\n\nTrying to classificate " + imagePath)
		
		# Load image, preprocess it and classify it
		image = load_img(imagePath, target_size=(s.img_width, s.img_height))
		image = img_to_array(image)
		image = u.meanSubtraction(image)
		image = image / 255
		image = np.expand_dims(image, axis=0)
		prediction = model.predict(image)

		# Get ID of the best prediction to obtain the associated class label
		ID = prediction.argmax()
		mapping = {v: k for k, v in pillDictionary.items()}
		label = mapping[ID]

		total += 1
		pathComponents = imagePath.split('/')
		if pathComponents[2] == label:
			correct += 1
		else:
			# print prediction value if classification is wrong
			print(prediction)

		print("Predicted Class ID: {}, Label: {}, Real Label: {}, Correct? {}".format(ID, label, pathComponents[2], pathComponents[2] == label))
		logFile.write('{}\t{}\t{}\n'.format(pathComponents[3], pathComponents[2], label))

	finalMsg = "Total images classified: {}, total correct: {}, final ratio: {:.2f}%".format(total, correct, correct * 100.0 / total)
	print(finalMsg)
	logFile.write(finalMsg)

print("Done!")
