# -*- coding: utf-8 -*-
"""
This script runs the text-only, audio-only an dmultimodal models
described in the main paper:
    
    Rafael Mestre, Razvan Milicin, Stuart E. Middleton, Matt Ryan, 
    Jiatong Zhu, Timothy J. Norman. 2021. M-Arg: Multimodal Argument 
    Mining Dataset for Political Debates with Audio and Transcripts.
    8th Workshop on Argument Mining, 2021 at 2021 Conference on 
    Empirical Methods in Natural Language Processing (EMNLP).


License:
    BSD 4-clause with attribution License

    Copyright (c) 2021 University of Southampton

@authors: Razvan Milicin (razvan.milicin@gmail.com)
          Rafael Mestre (R.Mestre@soton.ac.uk)*

*corresponding author
          
"""


# pip install -q tf-models-official

# pip install -q -U tensorflow-text


#importing useful libraries
import pandas as pd
from pathlib import Path

import tensorflow as tf
import numpy as np
import librosa #for audio
# # %matplotlib inline
import matplotlib.pyplot as plt
import librosa.display #for audio
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm




#One-hot encoding function for the labels
def obj_to_oh(labels):
    """
    One-hot encoding function to encode the relation labels.

    Parameters
    ----------
    labels : numpy.ndarray
        Array of labels to one-hot encode of shape (N,).

    Returns
    -------
    labels_oh : numpy.ndarray
        Array of one-hot encoded labels of shape of shape (N,M),
        where M is the number of unique labels found.
        In our case, since we have 'support','attack', and 'neither',
        M = 3.

    """
    labels = np.array(labels.tolist())
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    oe = OneHotEncoder(sparse=False)
    labels = labels.reshape(len(labels), 1)
    labels_oh = oe.fit_transform(labels)
    return labels_oh


def audio_feat_ext(df):
    """
    Audio feature extraction function. 

    Parameters
    ----------
    df : Pandas DataFrame
        Dataframe with the data to extract features. Must have two columns called
        "sentence_1_audio" and "sentence_2_audio" with the IDs of the audio files
        corresponding to each utterance.

    Returns
    -------
    audio_features1 : LIST
        List of numpy.ndarrays with the audio features of the first sentence.
        Each element of the list corresponds to the first sentence of a pair
        and it's a numpy array of shape (45,T), where T can be different 
        for each sentence, depending on its duration.
    audio_features2 : LIST
        List of numpy.ndarrays with the audio features of the second sentence.
        Each element of the list corresponds to the second sentence of a pair
        and it's a numpy array of shape (45,T), where T can be different 
        for each sentence, depending on its duration.
    max_shape : TUPLE
        Tuple with the shape of the feature vector with the longest T dimension.
        It will be used to pad all the vectors with 0's to obtain the same length.

    """
    audio_features1 = []
    audio_features2 = []
    max_shape = 0

    #Loop through the whole dataframe that extracts the audio features of
    #the first and second sentences of the pair
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            #first sentence

            x, sr = librosa.load('/content/drive/MyDrive/Internship1/US_2020_presidential_debates/Audio_sentences/' + row['sentence_1_audio'])
            mfccs = librosa.feature.mfcc(x, sr=sr,n_mfcc=25)[2:]
            spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(x, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(x, sr=sr)
            chroma_ft = librosa.feature.chroma_stft(x, sr=sr)
            features = np.concatenate((spectral_centroids, spectral_bandwidth, spectral_rolloff, spectral_contrast, chroma_ft, mfccs),axis=0)
            if features.shape[1] > max_shape:
                max_shape = features.shape[1]
            audio_features1.append(features)
        except:
            #this is for the case when the audio sentences have 0 duration (there are some because of the alignment software interracting with certain complicated situations)
            df = df.drop(index=index)
            print("Pair removed from dataset due to faulty audio feature extraction.")
            continue

            #second sentence
        try:
            x, sr = librosa.load('/content/drive/MyDrive/Internship1/US_2020_presidential_debates/Audio_sentences/' + row['sentence_2_audio'])
            mfccs = librosa.feature.mfcc(x, sr=sr,n_mfcc=25)[2:]
            spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(x, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(x, sr=sr)
            chroma_ft = librosa.feature.chroma_stft(x, sr=sr)
            features = np.concatenate((spectral_centroids, spectral_bandwidth, spectral_rolloff, spectral_contrast, chroma_ft, mfccs),axis=0)
            if features.shape[1] > max_shape:
                max_shape = features.shape[1]
            audio_features2.append(features)
        except:
            #this is for the case when the audio sentences have 0 duration (there are some because of the alignment software interracting with certain complicated situations)
            df = df.drop(index=index)
            print("Pair removed from dataset due to faulty audio feature extraction.")
            audio_features1.pop()

    return audio_features1, audio_features2, max_shape



def audio_padding(audio_features,max_shape):
    """
    Padding function for the audio features, since they can have different lengths.

    Parameters
    ----------
    audio_features : list
        List of numpy.ndarray with the audio features of each sentence
        of shape (45, T), where T is variable depending on the duration of the 
        utterance. If T < max_shape, trailing 0's are added to the vector.
    max_shape : FLOAT
        Shape of the longest sentence to compare and add 0's.

    Returns
    -------
    padded_audio_features : list
        List of numpy.ndarray with the same audio features padded with
        trailing 0's. Each element of the list now has shape (45,max_shape).

    """
    padded_audio_features = []
    for features in audio_features:
        if features.shape[1] <= max_shape:
            features = np.concatenate((features,np.zeros((features.shape[0],(max_shape - features.shape[1])))), axis = 1)
            features = features.T
            padded_audio_features.append(features)
        #print (features.shape)
    return padded_audio_features


def audio_pre_processing(df_train, df_valid):
    """
    Audio pre-processing function.

    Parameters
    ----------
    df_train : Pandas DataFrame
        Dataframe with the training data.
    df_valid : Pandas DataFrame
        Dataframe with the validation data.

    Returns
    -------
    padded_audio_train1 : numpy.ndarray
        Training numpy.ndarray of audio features, where each element
        corresponds to the first sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    padded_audio_train2 : TYPE
        Trainng numpy.ndarray of audio features, where each element
        corresponds to the second sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    padded_audio_valid1 : TYPE
        Validation numpy.ndarray of audio features, where each element
        corresponds to the second sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    padded_audio_valid2 : TYPE
        Validation numpy.ndarray of audio features, where each element
        corresponds to the second sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    text_train1 : Panda Series
        Training set of utterances in text form that make up the first sentence of the pair.
    text_train2 : Panda Series
        Training set of utterances in text form that make up the second sentence of the pair.
    text_valid1 : Panda Series
        Validation set of utterances in text form that make up the first sentence of the pair.
    text_valid2 : Panda Series
        Validation set of utterances in text form that make up the second sentence of the pair.
    y_train : Panda Series
        Training set of relation labels in text form (support, attack, neither).
    y_valid : TYPE
        Validation set of relation labels in text form (support, attack, neither).

    """

    audio_features_valid1, audio_features_valid2, max_shape_valid = audio_feat_ext(df_valid)
    audio_features_train1, audio_features_train2, max_shape_train = audio_feat_ext(df_train)
    y_valid, text_valid1, text_valid2 = df_valid['relation'], df_valid['sentence_1'], df_valid['sentence_2']
    y_train, text_train1, text_train2 = df_train['relation'], df_train['sentence_1'], df_train['sentence_2']
    
    
    #padding the audio files with trailing 0's
    max_shape = max(max_shape_valid,max_shape_train)    

    padded_audio_valid1 = np.array(audio_padding(audio_features_valid1,max_shape))
    padded_audio_valid2 = np.array(audio_padding(audio_features_valid2,max_shape))
    padded_audio_train1 = np.array(audio_padding(audio_features_train1,max_shape))
    padded_audio_train2 = np.array(audio_padding(audio_features_train2,max_shape))
    
    
    return padded_audio_train1, padded_audio_train2, padded_audio_valid1, padded_audio_valid2, text_train1, text_train2, text_valid1, text_valid2, y_train, y_valid

#Multimodal auidio-text model

def multimodal_model(input_shape, dropout_audio = 0.2, dropout_text = 0.1):
    """
    Multimoal audio-text model. For more details see the main paper.

    Parameters
    ----------
    input_shape : Tuple
        Shape of the input layer.
    dropout_audio : FLOAT, optional
        Value of the dropout rate for audio in this model. The default is 0.2.
    dropout_text : FLOAT, optional
        Value of the dropout rate for text in this model. The default is 0.1.

    Returns
    -------
    model : tf.keras.Model
        Text-only model.

    """
    #Inputs
    In_text_1 = tf.keras.layers.Input(shape = (), dtype=tf.string, name='text1')
    In_text_2 = tf.keras.layers.Input(shape = (), dtype=tf.string, name='text2')

    In_audio_1 = tf.keras.layers.Input(shape = input_shape)
    In_audio_2 = tf.keras.layers.Input(shape = input_shape)

    #text part: BERT model + dense layer for reshaping output (dense layer is shared between the two sentences)
    bert_preprocess_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    preproc_text_1 = bert_preprocess_model(In_text_1)
    encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3', trainable=False, name='BERT_encoder')
    enc_outputs_1 = encoder(preproc_text_1)
    X_text_1 = enc_outputs_1['pooled_output']
    X_text_1 = tf.keras.layers.Dropout(rate=dropout_text)(X_text_1)

    preproc_text_2 = bert_preprocess_model(In_text_2)
    enc_outputs_2 = encoder(preproc_text_2)
    X_text_2 = enc_outputs_2['pooled_output']
    X_text_2 = tf.keras.layers.Dropout(rate=dropout_text)(X_text_2)

    #audio part
    #lstm + attention + dense, layers are shared between the two sentences of each pair
    Bi_LSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(l2=0.5)))
    #Att = tf.keras.layers.Attention()
    f0 = tf.keras.layers.Flatten()
    dr0 = tf.keras.layers.Dropout(rate=dropout_audio)

    X_seq1 = Bi_LSTM(In_audio_1[:,:,:,0])
    #X_seq1 = Att([X_seq1,X_seq1])
    X_seq1 = f0(X_seq1)
    X_seq1 = dr0(X_seq1)

    X_seq2 = Bi_LSTM(In_audio_2[:,:,:,0])
    #X_seq2 = Att([X_seq2,X_seq2])
    X_seq2 = f0(X_seq2)
    X_seq2 = dr0(X_seq2)

    #cnn part for audio, layers are shared between the two sentences of each pair. Layers:
    #7x7x8, 1x1 stride convolutional
    #2x2, 2x2 stride maxpool
    #7x7x8, 1x1 stride convolutional
    #2x2, 2x2 stride maxpool
    #dense layer for reshaping
    conv1 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 7, strides = (1,1), padding = 'valid')
    bn1 = tf.keras.layers.BatchNormalization()
    a1 = tf.keras.layers.Activation('relu')
    dr1 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp1 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))
    conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = 7, strides = (1,1), padding = 'valid')
    bn2 = tf.keras.layers.BatchNormalization()
    a2 = tf.keras.layers.Activation('relu')
    dr2 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp2 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))
    f = tf.keras.layers.Flatten()

    X1 = conv1(In_audio_1)
    X1 = bn1(X1)
    X1 = a1(X1)
    X1 = dr1(X1)
    X1 = mp1(X1)
    X1 = conv2(X1)
    X1 = bn2(X1)
    X1 = a2(X1)
    X1 = dr2(X1)
    X1 = mp2(X1)
    X1 = f(X1)

    X2 = conv1(In_audio_2)
    X2 = bn1(X2)
    X2 = a1(X2)
    X2 = dr1(X2)
    X2 = mp1(X2)
    X2 = conv2(X2)
    X2 = bn2(X2)
    X2 = a2(X2)
    X2 = dr2(X2)
    X2 = mp2(X2)
    X2 = f(X2)

    X_fin = tf.keras.layers.Concatenate()([X_text_1,X_text_2,X_seq1,X_seq2,X1,X2])
    X_fin = tf.keras.layers.Dense(units = 100)(X_fin)
    X_fin = tf.keras.layers.BatchNormalization()(X_fin)
    X_fin = tf.keras.layers.Activation('relu')(X_fin)
    X_fin = tf.keras.layers.Dropout(rate=0.1)(X_fin)
    #X_out = tf.keras.layers.Concatenate()([X_fin,In_conf])

    out = tf.keras.layers.Dense(units = 3, activation = tf.nn.softmax)(X_fin)
    model = tf.keras.Model(inputs=[In_text_1,In_text_2,In_audio_1,In_audio_2], outputs=out)
    return model

#model for only text data

def text_model(dropout_text = 0.1):
    """
    Text-only model. For details see the main paper.

    Parameters
    ----------
    dropout_text : FLOAT, optional
        Value of the dropout rate for text in this model. The default is 0.1.

    Returns
    -------
    model : tf.keras.Model
        Text-only model.

    """
    #Inputs
    In_text_1 = tf.keras.layers.Input(shape = (), dtype=tf.string, name='text1')
    In_text_2 = tf.keras.layers.Input(shape = (), dtype=tf.string, name='text2')

    #text part: BERT model + dense layer for reshaping output (dense layer is shared between the two sentences)
    bert_preprocess_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    preproc_text_1 = bert_preprocess_model(In_text_1)
    encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3', trainable=False, name='BERT_encoder')
    enc_outputs_1 = encoder(preproc_text_1)
    X_text_1 = enc_outputs_1['pooled_output']
    X_text_1 = tf.keras.layers.Dropout(rate=dropout_text)(X_text_1)

    preproc_text_2 = bert_preprocess_model(In_text_2)
    enc_outputs_2 = encoder(preproc_text_2)
    X_text_2 = enc_outputs_2['pooled_output']
    X_text_2 = tf.keras.layers.Dropout(rate=dropout_text)(X_text_2)

    X_fin = tf.keras.layers.Concatenate()([X_text_1,X_text_2])
    X_fin = tf.keras.layers.Dense(units = 100)(X_fin)
    X_fin = tf.keras.layers.BatchNormalization()(X_fin)
    X_fin = tf.keras.layers.Activation('relu')(X_fin)
    X_fin = tf.keras.layers.Dropout(rate=0.1)(X_fin)

    out = tf.keras.layers.Dense(units = 3, activation = tf.nn.softmax)(X_fin)
    model = tf.keras.Model(inputs=[In_text_1,In_text_2], outputs=out)
    return model

#model for audio data only

def audio_model(input_shape, dropout_audio = 0.2):
    """
    Audio-only model. For details see the main paper.

    Parameters
    ----------
    input_shape : Tuple
        Shape of the input layer.
    dropout_audio : FLOAT, optional
        Value of the dropout rate for audio in this model. The default is 0.2.

    Returns
    -------
    model : tf.keras.Model
        Text-only model.

    """
    #Inputs
    In_audio_1 = tf.keras.layers.Input(shape = input_shape)
    In_audio_2 = tf.keras.layers.Input(shape = input_shape)

    #audio part
    #lstm + attention + dense, layers are shared between the two sentences of each pair
    Bi_LSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(l2=0.5)))
    #Att = tf.keras.layers.Attention()
    f0 = tf.keras.layers.Flatten()
    dr0 = tf.keras.layers.Dropout(rate=dropout_audio)

    X_seq1 = Bi_LSTM(In_audio_1[:,:,:,0])
    #X_seq1 = Att([X_seq1,X_seq1])
    X_seq1 = f0(X_seq1)
    X_seq1 = dr0(X_seq1)

    X_seq2 = Bi_LSTM(In_audio_2[:,:,:,0])
    #X_seq2 = Att([X_seq2,X_seq2])
    X_seq2 = f0(X_seq2)
    X_seq2 = dr0(X_seq2)

    #cnn part for audio, layers are shared between the two sentences of each pair. Layers:
    #7x7x8, 1x1 stride convolutional
    #2x2, 2x2 stride maxpool
    #7x7x8, 1x1 stride convolutional
    #2x2, 2x2 stride maxpool
    #dense layer for reshaping
    conv1 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 7, strides = (1,1), padding = 'valid')
    bn1 = tf.keras.layers.BatchNormalization()
    a1 = tf.keras.layers.Activation('relu')
    dr1 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp1 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))
    conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = 7, strides = (1,1), padding = 'valid')
    bn2 = tf.keras.layers.BatchNormalization()
    a2 = tf.keras.layers.Activation('relu')
    dr2 = tf.keras.layers.Dropout(rate=dropout_audio)
    mp2 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))
    f = tf.keras.layers.Flatten()

    X1 = conv1(In_audio_1)
    X1 = bn1(X1)
    X1 = a1(X1)
    X1 = dr1(X1)
    X1 = mp1(X1)
    X1 = conv2(X1)
    X1 = bn2(X1)
    X1 = a2(X1)
    X1 = dr2(X1)
    X1 = mp2(X1)
    X1 = f(X1)

    X2 = conv1(In_audio_2)
    X2 = bn1(X2)
    X2 = a1(X2)
    X2 = dr1(X2)
    X2 = mp1(X2)
    X2 = conv2(X2)
    X2 = bn2(X2)
    X2 = a2(X2)
    X2 = dr2(X2)
    X2 = mp2(X2)
    X2 = f(X2)

    X_fin = tf.keras.layers.Concatenate()([X_seq1,X_seq2,X1,X2])
    X_fin = tf.keras.layers.Dense(units = 100)(X_fin)
    X_fin = tf.keras.layers.BatchNormalization()(X_fin)
    X_fin = tf.keras.layers.Activation('relu')(X_fin)
    X_fin = tf.keras.layers.Dropout(rate=0.1)(X_fin)
    #X_out = tf.keras.layers.Concatenate()([X_fin,In_conf])

    out = tf.keras.layers.Dense(units = 3, activation = tf.nn.softmax)(X_fin)
    model = tf.keras.Model(inputs=[In_audio_1,In_audio_2], outputs=out)
    return model

def run_audio_model(padded_audio_train1,padded_audio_train2,padded_audio_valid1, 
                    padded_audio_valid2, y_train_oh, y_valid_oh, 
                    dropout_audio = 0.2):
    """
    Function that creates and runs (fits) the audio-only model.

    Parameters
    ----------
    padded_audio_train1 : numpy.ndarray
        Training numpy.ndarray of audio features, where each element
        corresponds to the first sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    padded_audio_train2 : numpy.ndarray
        Training numpy.ndarray of audio features, where each element
        corresponds to the second sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    padded_audio_valid1 : numpy.ndarray
        Validation numpy.ndarray of audio features, where each element
        corresponds to the first sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    padded_audio_valid2 : numpy.ndarray
        Validation numpy.ndarray of audio features, where each element
        corresponds to the second sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    y_train_oh : numpy.ndarray
        Training labels of the relationship between both sentences in one-hot encoded form:
            (1,0,0) for attack, (0,1,0) for neither, and (0,0,1) for support.
    y_valid_oh : numpy.ndarray
        Validation labels of the relationship between both sentences in one-hot encoded form:
            (1,0,0) for attack, (0,1,0) for neither, and (0,0,1) for support.
    dropout_audio : FLOAT, optional
        Value of the dropout rate for audio in this model. The default is 0.2.

    Returns
    -------
    model : tf.keras.Model
        Fitted model.

    """

    input_shape = padded_audio_train1.shape[1:] + tuple([1])
    #Initialise model
    model = audio_model(input_shape, dropout_audio)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss = 'CategoricalCrossentropy', metrics=['acc',tf.keras.metrics.AUC(multi_label=True,num_labels=3)])
    model.fit(x=[padded_audio_train1,padded_audio_train2], y=y_train_oh, batch_size=batch_size,epochs=epochs,validation_data=([padded_audio_valid1,padded_audio_valid2],y_valid_oh),callbacks=[scheduler])
    
    return model

def run_multimodal_model(text_train1,text_train2,padded_audio_train1,
                         padded_audio_train2, text_valid1,text_valid2,
                         padded_audio_valid1, padded_audio_valid2, y_train_oh, 
                         y_valid_oh, dropout_audio = 0.2, dropout_text = 0.1):
    """
    Function that creates and runs (fits) the multimodal model.

    Parameters
    ----------
    text_train1 : Pandas Series
        Training set of utterances in text form that make up the first sentence of the pair.
    text_train2 : Pandas Series
        Training set of utterances in text form that make up the second sentence of the pair.
    padded_audio_train1 : numpy.ndarray
        Training numpy.ndarray of audio features, where each element
        corresponds to the first sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    padded_audio_train2 : numpy.ndarray
        Training numpy.ndarray of audio features, where each element
        corresponds to the second sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    text_valid1 : Pandas Series
        Validation set of utterances in text form that make up the first sentence of the pair.
    text_valid2 : Pandas Series
        Validation set of utterances in text form that make up the second sentence of the pair.
    padded_audio_valid1 : numpy.ndarray
        Validation numpy.ndarray of audio features, where each element
        corresponds to the first sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    padded_audio_valid2 : numpy.ndarray
        Validation numpy.ndarray of audio features, where each element
        corresponds to the second sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    y_train_oh : numpy.ndarray
        Training labels of the relationship between both sentences in one-hot encoded form:
            (1,0,0) for attack, (0,1,0) for neither, and (0,0,1) for support.
    y_valid_oh : numpy.ndarray
        Validation labels of the relationship between both sentences in one-hot encoded form:
            (1,0,0) for attack, (0,1,0) for neither, and (0,0,1) for support.
    dropout_audio : FLOAT, optional
        Value of the dropout rate for audio in this model. The default is 0.2.
    dropout_text : FLOAT, optional
        Value of the dropout rate for text in this model. The default is 0.1.

    Returns
    -------
    model : tf.keras.Model
        Fitted model.

    """

    input_shape = padded_audio_train1.shape[1:] + tuple([1])
    #Initialise model
    model = multimodal_model(input_shape, dropout_audio, dropout_text)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss = 'CategoricalCrossentropy', metrics=['acc',tf.keras.metrics.AUC(multi_label=True,num_labels=3)])
    model.fit(x=[text_train1,text_train2,padded_audio_train1,padded_audio_train2], y=y_train_oh, batch_size=batch_size,epochs=epochs,validation_data=([text_valid1,text_valid2,padded_audio_valid1,padded_audio_valid2],y_valid_oh),callbacks=[scheduler])
    
    return model

def run_text_model(text_train1, text_train2, text_valid1, text_valid2, 
                   y_train_oh, y_valid_oh, dropout_text = 0.1):
    """
    Function that creates and runs (fits) the text-only model.

    Parameters
    ----------
    text_train1 : Pandas Series.
        Training set of utterances in text form that make up the first sentence of the pair.
    text_train2 : Pandas Series.
        Training set of utterances in text form that make up the second sentence of the pair.
    text_valid1 : Pandas Series.
        Validation set of utterances in text form that make up the first sentence of the pair.
    text_valid2 : Pandas Series.
        Validation set of utterances in text form that make up the second sentence of the pair.
    y_train_oh : numpy.ndarray
        Training labels of the relationship between both sentences in one-hot encoded form:
            (1,0,0) for attack, (0,1,0) for neither, and (0,0,1) for support.
    y_valid_oh : numpy.ndarray
        Validation labels of the relationship between both sentences in one-hot encoded form:
            (1,0,0) for attack, (0,1,0) for neither, and (0,0,1) for support.
    dropout_text : FLOAT, optional
        Value of the dropout rate in this model. The default is 0.1.

    Returns
    -------
    model : tf.keras.Model
        Fitted model.

    """

    model = text_model(dropout_text)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss = 'CategoricalCrossentropy', metrics=['acc',tf.keras.metrics.AUC(multi_label=True,num_labels=3)])
    model.fit(x=[text_train1,text_train2], y=y_train_oh, batch_size=batch_size,epochs=epochs,validation_data=([text_valid1,text_valid2],y_valid_oh),callbacks=[scheduler])
    
    return model

def get_metrics(model, run_nb, y_valid_oh, text_valid1 = None, 
                text_valid2 = None, padded_audio_valid1 = None, 
                padded_audio_valid2 = None, dropouts = []):
    """
    Function that, given a model and a set of x-y values, computes the prediction
    and generates a report of metrics that evaluates the performance
    of the model.

    Parameters
    ----------
    model : tf.keras.Model
        Model to be evaluated.
    run_nb : FLOAT
        Number that indicates the "run number", meaning the number
        of times the model was fitted, in order to save the results
        with a different filename, appending "_run_nb" at the end.
        Otherwise, it'll overwrite the previuos model's results.
    y_valid_oh : numpy.ndarray
        Validation labels of the relationship between both sentences in one-hot encoded form:
            (1,0,0) for attack, (0,1,0) for neither, and (0,0,1) for support.
    text_valid1 : Pandas Series, optional
        Validation set of utterances in text form that make up the first sentence of the pair.
        The default is None.
    text_valid2 : Pandas Series, optional
        Validation set of utterances in text form that make up the second sentence of the pair.
        The default is None.
    padded_audio_valid1 : numpy.ndarray, optional
        Validation numpy.ndarray of audio features, where each element
        corresponds to the first sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    padded_audio_valid2 : numpy.ndarray, optional
        Validation numpy.ndarray of audio features, where each element
        corresponds to the second sentence of a pair. Each element has
        the same shape (45, Tmax), where Tmax is the shape of
        the longest sentence, and all of them have been padded to this
        length with trailing 0's.
    dropouts : list, optional
        List of dropout rates of 2 dimensions. The first element corresponds
        to the text dropout and the second element to the audio dropout. 
        The default is [].

    Returns
    -------
    None. It saves a .png and .svg file with the confusion matrix,
    a .csv file with the classficiation report, and it prints it
    on screen.

    """
    #This part figures out which time of model it is according to the
    #validation features introduced
    if padded_audio_valid1 is None and padded_audio_valid2 is None:
        #No audio
        if text_valid1 is None and text_valid2 is None:
            raise ValueError("No text or audio were in input.")
        else: #There is no audio but there is text
            model_type = 'text'
    elif text_valid1 is None and text_valid2 is None:
        #there is audio but no text
        model_type = 'audio'
    else:
        #there is audio AND text
        model_type = 'multimodal'
    
    #We predict the labels of the validation set
    if model_type == 'text':
        y_pred = model.predict(x=[text_valid1,text_valid2])
        savename = "metrics_text_0"+str(dropouts[0]*10)[0]
    elif model_type == 'audio':
        y_pred = model.predict(x=[padded_audio_valid1,padded_audio_valid2])
        savename = "metrics_audio_0"+str(dropouts[1]*10)[0]
    elif model_type == 'multimodal':
        y_pred = model.predict(x=[text_valid1,text_valid2,padded_audio_valid1,padded_audio_valid2])
        savename = "metrics_multimodal_0"+str(dropouts[0]*10)+"_0"+str(dropouts[1]*10)[0]
    
    #We transform the one-hot encoding of the form (x,x,x)
    #to a single label between of 0,1,2
    y_pred = np.argmax(y_pred,axis=1)
    y_true = np.argmax(y_valid_oh,axis=1)
    #We compute the confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    
    #This part of the code is aesthetics and is shared by every model
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    
    group_ratios = [value*100 for value in
                    cf_matrix[0,:].flatten()/np.sum(cf_matrix[0,:])]
    
    group_ratios = group_ratios + [value*100 for value in
                    cf_matrix[1,:].flatten()/np.sum(cf_matrix[1,:])]
    
    group_ratios = group_ratios + [value*100 for value in
                    cf_matrix[2,:].flatten()/np.sum(cf_matrix[2,:])]
    
    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix[0,:].flatten()/np.sum(cf_matrix[0,:])]
    
    group_percentages = group_percentages + ["{0:.2%}".format(value) for value in
                        cf_matrix[1,:].flatten()/np.sum(cf_matrix[1,:])]
    
    group_percentages = group_percentages + ["{0:.2%}".format(value) for value in
                        cf_matrix[2,:].flatten()/np.sum(cf_matrix[2,:])]
    
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_counts,group_percentages)]
    
    group_ratios = np.asarray(group_ratios).reshape(3,3)
    labels = np.asarray(labels).reshape(3,3)
    fig = plt.figure(figsize=(6,6)) 
    ax = plt.gca()
    sns.heatmap(group_ratios, annot=labels, fmt="", cmap='Blues', ax=ax, annot_kws={"size":15}, vmin=0, vmax=100)
    
    ax.set_xlabel('Predicted labels', fontsize = 14, labelpad= 10)
    ax.set_ylabel('True labels', fontsize = 14, labelpad = 10)
    ax.set_title('Confusion Matrix', fontsize = 14)
    ax.xaxis.set_ticklabels(['attack','neither','support'], fontsize = 14)
    ax.yaxis.set_ticklabels(['attack','neither','support'], fontsize = 14)
    ax.collections[0].colorbar.set_label("Percentage (%)", fontsize=15)
    cbar_ax = ax.figure.axes[-1].yaxis
    cbar_ax.set_ticklabels(cbar_ax.get_ticklabels(), fontsize=15)
    
    #We save the confusion matrix of the results
    fig.savefig(savename+"_cm_"+str(run_nb)+".png", dpi=300)
    fig.savefig(savename+"_cm_"+str(run_nb)+".svg", dpi=300)
    
    #Classification report
    report = classification_report(y_true, y_pred, target_names=['attack','neither','support'],output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    #There is a column named "support", which is the number of occurences in each class
    #Since this is confusing with our "support" label, we change the name
    df_report = df_report.rename(columns = {'support':'occurences'})
    
    #We save the classification report in .csv and we print it on screen
    df_report.to_csv(savename+"_metrics_"+str(run_nb)+".csv")
    print(df_report)
















'''
--------------------------------------------------------------------------
'''

#Loading the annotated dataset
filepath_annotated = Path(r'..\annotated dataset')
filename_agg = 'aggregated_dataset.csv'
df = pd.read_csv(Path(filepath_annotated,filename_agg))

#Loading the full dataset to link sentences to audio clips
filename_original = r'full_feature_extraction_dataset.csv'
filepath_original = Path(r'..\data\preprocessed full dataset')
df_text_audio = pd.read_csv(Path(filepath_original,filename_original))


#this section of code builds a new dataframe from the table containing annotated data
df_final = pd.DataFrame(columns=['id','relation','confidence',
                                 'sentence_1','sentence_2',
                                 'sentence_1_audio','sentence_2_audio'])

for index, row in df.iterrows():
    #ids
    id1 = row["pair_id"]
    #labels
    relation1 = row["relation"]
    #label confidence
    conf1 = row["relation:confidence"]
    #sentences
    s1t = row["sentence_1"]
    s2t = row["sentence_2"]
    #correponding audio sentences based on the text
    s1a = df_text_audio['audio_file'].loc[df_text_audio['text'] == s1t].values[0]
    s2a = df_text_audio['audio_file'].loc[df_text_audio['text'] == s2t].values[0]
    # If we want to filter by annotation confidence we can add here the following if statement
    # if row["relation:confidence"] > 0.85:
    df_final = df_final.append({'id' : id1, 'relation' : relation1, 'confidence' : conf1, 'sentence_1' : s1t, 'sentence_2' : s2t, 'sentence_1_audio' : s1a, 'sentence_2_audio' : s2a}, ignore_index=True)




#Defining model hyperparameters
global learning_rate, decay, batch_size, epochs, scheduler
learning_rate = 0.00005
decay = 0.0000002
batch_size = 16
epochs = 50

#learning rate schedule function
def schedule(epoch,learning_rate):
    """Learning rate schedule function."""
    return learning_rate*1/(1+epoch*decay)
scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

#early stopping for the situation when you might want to use it (not used here)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)






#########
#IMPORTANT:
#Remember to modify the run number to save the models with a different name
#########
run_nb = 1




#Randomly splitting the data between training and validation, and randomly reordering 
#Only once per iteration, so the audio/text/multimodal models share exactly the same
#train and test data, but we re-do the sampling every time we do another trial

df_final = df_final.sample(frac=1).reset_index(drop=True) #Reshuffle
df_train, df_valid = train_test_split(df_final, test_size=0.2)


####TEXT ONLY####
#If only using text, we don't call the audio feature extraction

y_valid, text_valid1, text_valid2 = df_valid['relation'], df_valid['sentence_1'], df_valid['sentence_2']
y_train, text_train1, text_train2 = df_train['relation'], df_train['sentence_1'], df_train['sentence_2']

#Making y labels one-hot encoded
split = y_train.shape[0]
y = np.concatenate((y_train,y_valid))
y_oh = obj_to_oh(y)
y_train_oh = y_oh[:split,:]
y_valid_oh = y_oh[split:,:]

#Run the model
#Note: comment next line if you don't want to run the text-only model
model_text = run_text_model(text_train1, text_train2, text_valid1, text_valid2, y_train_oh, y_valid_oh, dropout_text = 0.1)

#Run the metrics
get_metrics(model_text, run_nb, y_valid_oh, 
            text_valid1=text_valid1, text_valid2=text_valid2, dropouts = [0.1,0.2])



####AUDIO ONLY####
#For the audio-only and the multimodal, we call the audio pre-processing function
#Extraction of features might take a long time to do, if you are not using audio features, do not run this
#A note for future improvement is to save the audio features next to the audio clips
#and load them each time. Not implemented here.
padded_audio_train1, padded_audio_train2, padded_audio_valid1, padded_audio_valid2, text_train1, text_train2, text_valid1, text_valid2, y_train, y_valid = audio_pre_processing(df_train, df_valid)

#Labels are one-hot encoded again because pre-processing audio might have dropped some examples
#if the audio could not be pre-processed for some reason (e.g. too short to extract features)
#In that case, that whole example will be discarded and the lengths of the vector change
split = y_train.shape[0]
y = np.concatenate((y_train,y_valid))
y_oh = obj_to_oh(y)
y_train_oh = y_oh[:split,:]
y_valid_oh = y_oh[split:,:]

#Run the model
#Note: comment next line if you don't want to run the text-only model
model_audio = run_audio_model(padded_audio_train1,padded_audio_train2,padded_audio_valid1, padded_audio_valid2, y_train_oh, y_valid_oh, dropout_audio = 0.2)

#Run the metrics
get_metrics(model_audio, run_nb, y_valid_oh, 
            padded_audio_valid1=padded_audio_valid1,padded_audio_valid2=padded_audio_valid2, dropouts = [0.1,0.2])




####MULTIMODAL
#With audio dropout 0.1
####

###Run the audio pre-processing, padding, etc, from the audio-only model above
###if you haven't done already, in order to run the following.

#Run the model (no need to pre-process audio again) with dropout audio 0.1
model_multimodal1 = run_multimodal_model(text_train1,text_train2,padded_audio_train1,padded_audio_train2, text_valid1,text_valid2,padded_audio_valid1, padded_audio_valid2, y_train_oh, y_valid_oh, dropout_text=0.1, dropout_audio=0.1)

#Run the metrics
get_metrics(model_multimodal1, run_nb, y_valid_oh, 
            text_valid1, text_valid2, padded_audio_valid1,padded_audio_valid2, dropouts = [0.1,0.1])

####MULTIMODAL
#With audio dropout 0.2
####

###Run the audio pre-processing, padding, etc, from the audio-only model above
###if you haven't done already, in order to run the following.

#Run the model (no need to pre-process audio again) with dropout audio 0.2
model_multimodal2 = run_multimodal_model(text_train1,text_train2,padded_audio_train1,padded_audio_train2, text_valid1,text_valid2,padded_audio_valid1, padded_audio_valid2, y_train_oh, y_valid_oh, dropout_text=0.1, dropout_audio=0.2)

#Run the metrics
get_metrics(model_multimodal2, run_nb, y_valid_oh, 
            text_valid1, text_valid2, padded_audio_valid1,padded_audio_valid2, dropouts = [0.1,0.2])
del model_multimodal2

####MULTIMODAL
#With audio dropout 0.3
####

###Run the audio pre-processing, padding, etc, from the audio-only model above
###if you haven't done already, in order to run the following.

#Run the model (no need to pre-process audio again) with dropout audio 0.3
model_multimodal3 = run_multimodal_model(text_train1,text_train2,padded_audio_train1,padded_audio_train2, text_valid1,text_valid2,padded_audio_valid1, padded_audio_valid2, y_train_oh, y_valid_oh, dropout_text=0.1, dropout_audio=0.3)

#Run the metrics
get_metrics(model_multimodal3, run_nb, y_valid_oh, 
            text_valid1, text_valid2, padded_audio_valid1,padded_audio_valid2, dropouts = [0.1,0.3])

