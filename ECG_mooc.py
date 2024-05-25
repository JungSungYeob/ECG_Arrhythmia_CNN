

import numpy as np
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix



# split 개수, 셔플 여부 및 seed 설정
str_kf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 14)


Real_result = []
R_Arr_results = []


# label num
Arr_num = -1
model_count = 0



# EarlyStopping
class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self, *args, **kwargs):
        super(MyModelCheckpoint, self).__init__(*args, **kwargs)

    # redefine the save so it only activates after 100 epochs
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= 1: super(MyModelCheckpoint, self).on_epoch_end(epoch, logs)



# Arrhythmia classification model
###############################################################################
def build_model_arr():
    inputs = Input(shape=(170, 1), name='main_input') #170: 시계열 데이터의 길이, 1: 채널 수
    """ 저차원 특징 추출 """
    #1D convolution layer
    #32개 필터, 커널 크기:3, Relu 활성화 함, 입출력 크기 동일
    x = Conv1D(filters=32, kernel_size=3, activation='relu',padding='same')(inputs)
    #Max pooling layer
    #특정 맵의 크기 줄이고, 중요한 특징 더 강조 역할(pool=size=2 : 풀링 크기 2로 설정하여 인접한 2개의 값 중 최대값 취함)
    x = MaxPooling1D(pool_size=2)(x)
    #배치 정규화 레이어:입력 정규화로 과적합 방지
    x = BatchNormalization()(x)
    #드롭 아웃, 무작위 뉴런 꺼서 과적합 방지
    x = Dropout(0.2)(x)
    """ 저차원 특징 바탕으로 더 복잡한 고차원 특징 학습 """
    #이번 Conv1D에는 x를 input으로 넣는다.
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    #Flatten: 다차원 배열 -> 1차원 배열 , Convolution layer에서 추출된 특징을 Dense 레이어에 전달하도록 함.
    x = Flatten()(x)    
    #Dense 레이어로 100개 뉴런 추가, 특징 맵 기반 분류 위한 특징 학습
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation='softmax', name='arrhythmia')(x)
    #모델 정의(입출력 정의)
    model = Model(inputs = inputs, outputs=outputs)
    
    #모델 반환
    return model




# data call
###############################################################################
#각자 위치로 조정할것
train_data = np.load('./train_data.npy')
test_data = np.load('./test_data.npy')


# model input shape 맞춰주기 = 3 dimension
###############################################################################
"""앞서 모델에서 Conv1D는 3차원 형태(batch size, sequence_length, num_features)"""
"""np.expand_dims를 통해 num_features 추가"""
train_data = np.expand_dims(train_data, axis = -1)
test_data = np.expand_dims(test_data, axis = -1)



# Arrhythmia lable
###############################################################################
train_label = np.zeros([4200, 1])
test_label = np.zeros([1260, 1])



#train_label 절반씩 나누기: 2100개 씩
for i in range(len(train_label)) :
    if i < 2100:
        train_label[i] = 0
    else:
        train_label[i]= 1

#test_label 절반씩 나누기: 630개 씩
for i in range(len(test_label)) :
    if i < 630 :
        test_label[i]=0
    else:
        test_label[i]=1
 
 
 # Arrhythmia train
 ###############################################################################    
for train_index, val_index in str_kf.split(train_data, train_label):
    
    T_data, V_data = train_data[train_index], train_data[val_index]
    T_label, V_label = train_label[train_index], train_label[val_index]
    
    
    # One Hot Encoding Lable
    T_label = to_categorical(T_label)
    V_label = to_categorical(V_label)
    t_label = to_categorical(test_label)

    

    print("R_Arrhythmia")
    model = build_model_arr()
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # EarlyStopping 조기종료 및 모델 학습
    early_stopping = tf.keras.callbacks.EarlyStopping(patience = 10)
    check_point = MyModelCheckpoint('best_model.keras', monitor = 'val_loss', mode = 'min', save_best_only = True, verbose = 1)

    hist = model.fit(T_data, T_label, epochs=50, batch_size=32, verbose=1, validation_data=(V_data, V_label), callbacks=[early_stopping, check_point])
    
                            
    model.summary() 
    model_count += 1


# Arrhythmia Test
###############################################################################    
    Real_loss, Real_acc = model.evaluate({'main_input' : test_data}, {'arrhythmia': t_label})
    Real_pred = model.predict(test_data)
    Real_pred = Real_pred[:,0]
    

    for i in range(len(Real_pred)):
        if(0.5 <= Real_pred[i]):
            Real_pred[i]=1
        else:
            Real_pred[i]=0
            
            
    # confusion matrix 생성  
    t_label = t_label[:,0]
    
    te_conf_matrix = confusion_matrix(t_label, Real_pred)
    print(te_conf_matrix)
    print('\n')
    
    
    # confusion matrix를 이용한 지표
    tp = te_conf_matrix[1,1]
    tn = te_conf_matrix[0,0]
    
    rows = np.sum(te_conf_matrix,axis=1)
    cols = np.sum(te_conf_matrix,axis=0)

    Real_pre = tp / cols[1] if cols[1]>0 else 0
    Real_rec = tp / rows[1] if rows[1]>0 else 0
    Real_spe = tn / rows[0] if rows[0]>0 else 0
      
    
    print('\n')
    Real_result.append(Real_loss)
    Real_result.append(Real_acc)
    Real_result.append(Real_pre)
    Real_result.append(Real_rec)
    Real_result.append(Real_spe)
    
model_count = 0


print("-----------------------------Result----------------------")
print("Arrhythmia detection")
print()

