'''
The dataset used is from the ABC version of the Nottingham Music Database.
link: http://abc.sourceforge.net/NMD/
'''

'''
MIT License

Copyright (c) 2017 Panagiotis Petridis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import glob
from tqdm import tqdm
import sys
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

# Loading the data.
paths = glob.glob('data/midi_abc/*.txt')

# First add them all in one big string
data = ""
for path in paths:
    data += (open(path).read())

data_split = []
tmp = ""

# Split the data at every 'X' so that it is easier to train is small pieces. Also remove newlines.
for i in data:
    if i=='X' and tmp!="":
        tmp = tmp.rstrip()
        data_split.append(tmp)
        tmp=""
    tmp += i

# The first element of the data is empty since I (unnecessarly) start cutting the string 
# by assuming that the first character won't be an 'X' but It's ok. I can just delete it.
del data_split[0]

# This is probably THE WORST way to do this but because I want the data to be in one big string. I combine all 
# the data_split portions to one string (now they don't contain newlines though). I do this because then removing new lines like I did above
# there is an issue where it only deletes the very last consequitive newlines.
data = ""
for i in data_split:
    data += i

# I then proceed to creating a dictionary with which I can convert characters to integers and vice versa (basically vectorization)
chars = sorted(list(set(data)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(data)
n_vocab = len(chars)

# The sequences have a length of 100 characters
seq_length = 100
# So just split the big string to 100-character segments and train the model afterwards 
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = data[i:i + seq_length]
	seq_out = data[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_samples = len(dataX)
X = np.reshape(dataX, (n_samples, seq_length, 1))
X = X / float(n_vocab)

y = np_utils.to_categorical(dataY) # Basically what we do is classification. Classifying what the next character 
								   # should be based on the previus characters

# Building the Keras model
model = Sequential()
# I use 2 LSTM layers with 512 neurons each
# Using dropout aids generalization (although the model still manages to overfit the train data a bit :/ )
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)) # Because I add one more LSTM layer I need to return the sequences
model.add(Dropout(0.2)) 
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# As always. Some eye candy to see how the model graph looks like (Also helps make sure nothing has been messed up).
print(model.summary())

# Usually it isn't a very good idea to start training from scratch but this particular weight file
# is the one that the slightly overfit model used. So although the results are decent, you may want to comment lines 84-86 
if os.path.isfile('best_weights.hdf5'):
	model.load_weights('best_weights.hdf5')
	model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "/output/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5" # Saving the model each time it improves
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fitting the model to the data.
model.fit(X, y, epochs=50, batch_size=1024, callbacks=callbacks_list)

# Making predictions
# I make 5 big predictions in hope of sounds that aren't all similar to each other. 
# This also helps pick out the nice parts from a big file
for k in range(5):
	start = np.random.randint(0, len(dataX)-1)
	count = 0
	pattern = dataX[start] # Pick up a random character from the dataset to start from.
	
	# Initial prediction is empty
	prediction_string = ""

	# Start predicting a sequence of characters.
	for i in tqdm(range(1000)):
		x = np.reshape(pattern, (1, len(pattern), 1)) # Resize the data to fit to the model.
		x = x/float(n_vocab)

		pred = model.predict(x, verbose=0) # Make prediction
		index = np.argmax(pred) # Get the character with the highest probability
		result = int_to_char[index] # Convert the one index of the character to a char.
		#seq_in = [int_to_char[value] for value in pattern] 
		prediction_string += result # add the result to the precition string
		#sys.stdout.write(result)
		pattern.append(index) # Add the predicted character to the pattern that will be fed next time the model.
						 	  # For example if the pattern is 'abb' and the model predicts 'a'. make sure to change the pattern that
						 	  # will be fed next to 'abba' and shorten it to fit so it becomes 'bba'.
		pattern = pattern[1:len(pattern)]
		# Due to the small size of the dataset the model will probably slightly overfit. To counter that and add a bit 
		# of variation I add something like noise to it's predictions. So I add random pathes of the data to force
		# the neural network to change the pattern and choose a different not to make the music a bit more interesting.
		start+=np.random.randint(0, 1)
		count+=1
		if count%250==0:
			start += np.random.randint(-100*i, 100*2*i)
			while start+np.random.randint(0, 200) > len(dataX):
				start = start-np.random.randint(0, 500)
			pattern.extend(dataX[start][:50])
			pattern = pattern[50:len(pattern)]
	print('Done')

	# Write the predictions to a .txt file.
	fl = open('big_pred_'+ str(k) +'.txt', 'w')
	fl.write(prediction_string)
	print(prediction_string)