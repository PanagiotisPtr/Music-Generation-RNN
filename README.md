# Music-Generation-RNN
A Neural Network that uses LSTM layers in Keras that learns how to compose music. The neural network consists of 2 LSTM layers and once dense.
I am treating the music generation task as classification where the neural network tries to predict the 'class'/character that is most likely
to appear next based on the sequence it has seen so far. The output file is in ABC format so you will need `abc2midi` and `timidity` to 
convert abc files to midi and play the midi files form the terminal.

The dataset used is the ABC version of the Nottingham Music Database. And can be found here: http://abc.sourceforge.net/NMD/

### Deep Learning Topics Applied on this project
  - Dropout
  - LSTM
  - Time series prediction
  - Vectorization
  - Cross Entropy Loss Function
  - Gradient Descent Optimization (Done by the tensorflow API)

### What is this project for?
This is one of my personal projects and I wrote this neural network mostly as a mean of practice. 

### Some Samples:
[![SoundCloud](https://raw.githubusercontent.com/PanagiotisPtr/Music-Generation-RNN/master/playlistThumbnail.jpg)](https://soundcloud.com/panos_ptr/sets/deepnote-predictions)  
Song playlist on **[SoundCloud](https://soundcloud.com/panos_ptr/sets/deepnote-predictions)**

##### Disclaimer
This project isn't meant for production and is mostly an experimentation. I still am a novice Python programmer and as such the code design isn't all that great.
Also I do not own the dataset, please visit http://abc.sourceforge.net/NMD/ for legal information and licesing of the dataset.

##### Foot note
I hope that you find this project informative and educational. Also the model slightly overfits the data so if you use the provided weight file
you may want to do a bit of editing on the ABC file and select the most interesting parts. Keep on learning!

Panagiotis Petridis
