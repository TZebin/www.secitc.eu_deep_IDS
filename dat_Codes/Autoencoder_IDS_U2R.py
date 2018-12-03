# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:55:26 2018
https://github.com/smellslikeml/deepIDS/

https://www.datacamp.com/community/tutorials/autoencoder-classifier-python
@author: mchijtz4:tahmina.zebin@manchester.ac.uk
"""
from tensorflow import set_random_seed
set_random_seed(12)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

X_total =pd.read_csv('G:\\NSL_KDD-master\\NSL-KDD\\nsltrain.csv',sep=',',encoding= 'utf-8-sig')
#train20_df=pd.read_csv('G:\\NSL_KDD-master\\KDDTrain+_20Percent.txt',sep=',',encoding= 'utf-8-sig',names =col_names)

#kdd_t=pd.read_csv('G:\\NSL_KDD-master\\KDDTest+.txt',sep=',',encoding= 'utf-8-sig',names =kdd_cols)
#
#X=[kdd, kdd_t]
#X_total=pd.concat(X)


# Here we opt for the 5-class problem

#Define functions cone-hot encoding and log scaling for variables
def cat_encode(df, col):
    return pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col].values)], axis=1)

def log_trns(df, col):
    return df[col].apply(np.log1p)
# Convert protocol, service and flag to label encoded/dummy variables
cat_lst = ['protocol_type', 'service', 'flag']
for col in cat_lst:
    X_total = cat_encode(X_total, col)

# Apply log scale on duratiom, source byte and destination byte Variables    
log_lst = ['duration', 'src_bytes', 'dst_bytes']
for col in log_lst:
    X_total[col] = log_trns(X_total, col)
   
X_total_cols2= list(X_total.columns.values)   
X_total = X_total[X_total_cols2]

#Data Ready and Prepared for classification Algorithms
#Select training set and test set variables and features

X_total.head()

difficulty = X_total.pop('difficulty')
X_total.to_csv('total_kdd.csv', index=False, header=True)
labels = X_total.pop('class')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_total, labels, test_size=0.25, random_state=10)

# We rescale features to [0, 1]

min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

X_test = min_max_scaler.transform(X_test)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

y_test = y_test.values
y_train = y_train.values

#for idx, col in enumerate(list(X_train.columns)):
#    print(idx, col)
 #Bluild the model with keras   
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adadelta,SGD,Adam,RMSprop

encoding_dim = 61
input_img = Input(shape=(122,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(122, activation='sigmoid')(encoded)
encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(loss='binary_crossentropy', optimizer = 'adadelta')
autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=50,
                validation_split=0.2,
                shuffle=True)

autoencoder.save_weights('autoencoder.h5')

def fc(enco):
    
    den = Dense(122, activation = 'relu')(enco)
    den1=Dense(64,activation = 'relu')(den)
    in1=BatchNormalization()(den1)
    out = Dense(5, activation='softmax')(in1)
    return out
encode = encoder(input_img)
full_model = Model(input_img,fc(encode))

for l1,l2 in zip(full_model.layers[:10],autoencoder.layers[0:10]):
    l1.set_weights(l2.get_weights())
    
for layer in full_model.layers[0:10]:
    layer.trainable = False
    
full_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
full_model.summary()   
    

history=full_model.fit(x=X_train, y=y_train, epochs=100, validation_split=0.2, batch_size=50,shuffle=True)
full_model.save_weights('autoencoder_classification.h5')


for layer in full_model.layers[0:5]:
    layer.trainable = True
full_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
full_model.summary()   
#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')
history=full_model.fit(x=X_train, y=y_train, epochs=100, validation_split=0.2, batch_size=50,shuffle=True)
full_model.save_weights('classification_complete.h5')

#NN.fit(x=train, y=target, epochs=100, validation_split=0.1, batch_size=128, callbacks=[early_stopping])
#plot accuracy graph for different iterations
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 6), dpi=80)

print(history.history.keys())
accuracy=[ history.history['acc'], history.history['val_acc'],history.history['loss'],history.history['val_loss']]
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'], '--')
plt.title('Model accuracy with Dense ANN',fontsize=16)
plt.ylabel('Average Accuracy [0-1]',fontsize=16)
plt.xlabel('No. of Epoch',fontsize=16)
plt.ylim((.97,1))
plt.legend(['Training Acuracy', 'Validation Accuracy'], loc='lower right')
plt.tight_layout()
plt.show()

#plot confusion matrix

from sklearn.metrics import confusion_matrix,classification_report,roc_curve

# With the confusion matrix, we can aggregate model predictions
# This helps to understand the mistakes and refine the model
preds = full_model.predict(X_test)
y_pred = (preds > 0.5).astype(int)
pred_lbls = np.argmax(preds, axis=1)
true_lbls = np.argmax(y_test, axis=1)

cm= confusion_matrix(true_lbls, pred_lbls)
full_model.evaluate(X_test, y_test)
target_names =['DoS','probe','R2L','U2R','Normal']
print(classification_report(true_lbls, pred_lbls,target_names=target_names))