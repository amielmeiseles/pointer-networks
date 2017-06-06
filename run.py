from keras.models import Model
from keras.layers import LSTM, Input
from keras.callbacks import LearningRateScheduler
from keras.utils.np_utils import to_categorical
from PointerLSTM import PointerLSTM
import pickle
#import tsp_data as tsp
import simple_seq as seq
import numpy as np

def scheduler(epoch):
    return learning_rate
    if epoch < nb_epochs/4:
        return learning_rate
    elif epoch < nb_epochs/2:
        return learning_rate*0.5
    return learning_rate*0.1

print("preparing dataset...")

seq_len = 5
encoding_len=2

s = seq.Simple_Seq()
X, Y = s.next_batch(1000, seq_len, encoding_len)
#X, Y = t.overfit(10000)
x_test, y_test = s.next_batch(2)

YY = np.eye(seq_len)[Y]

hidden_size = 128
nb_epochs =  10
learning_rate = 0.3

print("building model...")
main_input = Input(shape=(seq_len, encoding_len), name='main_input')

encoder = LSTM(units = hidden_size, return_sequences=True,name="encoder")(main_input)
decoder = PointerLSTM(hidden_size, units=hidden_size, return_sequences=False,name="decoder")(encoder)

model = Model(inputs=main_input, outputs=decoder)
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, YY, epochs=nb_epochs, batch_size=64,callbacks=[LearningRateScheduler(scheduler),])
#print(model.predict(x_test))
predictions = model.predict(X)
pred_index = np.array([np.argmax(predictions[i],0) for i in xrange(len(predictions))])
print(pred_index[:5])
print("------")
print(Y[:5])
#print(to_categorical(y_test))
model.save_weights('model_weight_100.hdf5')
