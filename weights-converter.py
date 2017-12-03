from keras.models import load_model
import os

import settings as s
import utils as u


def exportWeights():
	# Load fine tuned model with weights
	model = load_model(s.fine_tuned_model_path)
	print("Model loaded...")
	print("Getting weights...")

	W = model.get_weights()

	# Reshape and transpose are necessary to use weights with Metal Performance Shaders
	for i, w in enumerate(W):
		j = i // 2 + 1
		# print for info
		data = "weights" if i % 2 == 0 else "bias"
		print("Layer {}: exporting {} with shape: {}...".format(j, data, w.shape))

		# Handle fully connected weights
		if (j == 14 or j == 15) and i % 2 == 0:
			fc_shape = (7, 7, 512, 512) if (j == 14) else (1, 1, 512, 12)
			num = j - 13

			channel1 = fc_shape[0] * fc_shape[1] * fc_shape[2]
			channel2 = fc_shape[3]
			handledFCWeights = w.reshape(fc_shape).transpose(3, 0, 1, 2).reshape(channel1, channel2)
			handledFCWeights.tofile(os.path.join(s.params_path, "fc%d_weights.bin" % num))
		# Handle fully connected bias
		elif (j == 14 or j == 15) and i % 2 == 1:
			num = j - 13
			w.tofile(os.path.join(s.params_path, "fc%d_bias.bin" % num))
		# Handle convolutional weights and bias
		else:
			if i % 2 == 0:
				w.transpose(3, 0, 1, 2).tofile(os.path.join(s.params_path, "conv%d_weights.bin" % j))
			else:
				w.tofile(os.path.join(s.params_path, "conv%d_bias.bin" % j))


if __name__ == '__main__':
	u.createDirIfNotExisting(s.params_path)
	exportWeights()
	print("Done!")