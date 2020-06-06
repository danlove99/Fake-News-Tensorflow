import pandas as pd  
import numpy as np  
import tensorflow as tf  
from tensorflow import keras 
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Definitions 

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 35000

# Load data, merge fake and true, shuffle dataset, drop unnecasery columns

fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')
fake, true = fake.drop('title', axis=1), true.drop('title', axis=1)
fake, true = fake.drop('subject', axis=1), true.drop('subject', axis=1)
fake, true = fake.drop('date', axis=1), true.drop('date', axis=1)
true['True'], fake['True'] = 1, 0
df = true.append(fake)
df = df.sample(frac=1).reset_index(drop=True)

# Append data to arrays

sentences = []
labels = []

for index, row in df.iterrows():
	sentences.append(row['text'])
	labels.append(row['True'])

print("Succesfully appended data to array")


# Split data for training and testing

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


# Word to vec with pading

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


# work with TensorFlow 2
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(GlobalAveragePooling1D())
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(training_padded, training_labels, epochs=10, validation_data=(testing_padded, testing_labels), verbose=2, batch_size=10)