from keras.models import load_model
import numpy as np
import os

params_path = "params"
model_path = "fine-tuned-model.h5"

model = load_model(model_path)
print("Model loaded...")
print("Getting weights...")

W = model.get_weights()

for i, w in enumerate(W):
	j = i // 2 + 1
	print("Current layer: %d" % j)
	print w.shape

	if (j == 14 or j == 15) and i % 2 == 0:
		fc_shape = (7, 7, 512, 512) if (j == 14) else (1, 1, 512, 12)
		num = j - 13

		channel1 = fc_shape[0] * fc_shape[1] * fc_shape[2]
		channel2 = fc_shape[3]
		handledFCWeights = w.reshape(fc_shape).transpose(3, 0, 1, 2).reshape(channel1, channel2)
		handledFCWeights.tofile(os.path.join(params_path, "fc%d_weights.bin" % num))
	elif (j == 14 or j == 15) and i % 2 == 1:
		num = j - 13
		w.tofile(os.path.join(params_path, "fc%d_bias.bin" % num))
	else:
		if i % 2 == 0:
			w.transpose(3, 0, 1, 2).tofile(os.path.join(params_path, "conv%d_weights.bin" % j))
		else:
			w.tofile(os.path.join(params_path, "conv%d_bias.bin" % j))

print("Done!")