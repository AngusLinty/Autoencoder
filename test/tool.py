import scipy.io as sio
import os
import numpy as np
def normalize(input=[499,310]):
    temp_r = np.array(input)
    #temp_c = np.array(input)
    rows_sum = input.sum(axis=1)
    #cols_sum = input.sum(axis=0)
    temp_r = temp_r/rows_sum[:, np.newaxis]
    #temp_c = temp_c/cols_sum
    return temp_r
    #return(temp_r, temp_c)

def loadMatData():
    mat_datas = []
    data_dir_eeg = '/Users/Angus/Desktop/autoencoder/test/data/eegsplit/'
    for file in os.listdir(data_dir_eeg):
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".mat":
            mat_datas.append(sio.loadmat(data_dir_eeg+file))
    print(len(mat_datas))
    
    mat_labels = []
    data_dir_eye = '/Users/Angus/Desktop/autoencoder/test/data/eyesplit/'
    for file in os.listdir(data_dir_eye):
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".mat":
            mat_labels.append(sio.loadmat(data_dir_eye+file))
    print(len(mat_labels))

    return(mat_datas, mat_labels)

def getMaterial():
    mat_datas, mat_labels = loadMatData()

    mat_datas_train = []
    mat_datas_test = []
    mat_labels_train = []
    mat_labels_test = []
    
    for mat_i in range(len(mat_datas)):
        mat_datas_train.append(normalize(mat_datas[mat_i]['train_de']))
        mat_datas_test.append(normalize(mat_datas[mat_i]['test_de']))
    
        mat_labels_train.append(mat_labels[mat_i]['train_label_eye'])
        mat_labels_test.append(mat_labels[mat_i]['test_label_eye'])
    
    return(mat_datas_train, mat_datas_test, mat_labels_train, mat_labels_test)

#from liblinearutil import *
#accmax_list = []
#s_best_list = []
#c_best_list = []
#plabel_best_list = []
#for mat_i in range(len(mat_datas)):
#    print("%dth mat:" %(mat_i))
#    data_train_list = mat_datas_train[mat_i].tolist()
#    label_train_list = mat_labels_train[mat_i].T[0].tolist()
#    
#    data_test_list = mat_datas_test[mat_i].tolist()
#    label_test_list = mat_labels_test[mat_i].T[0].tolist()
#
#    prob = problem(label_train_list, data_train_list)
#    accmax = 0
#    s_best = 0
#    c_best = 0
#    plabel_best = 0
#    
#    for s in range(6):
#        for c in np.arange(1.5, 10.5, 0.5):
#            print(s,c)
#            param = parameter('-s %d -c %f -q'%(s, c))
#            m = train(prob, param)
#            p_labels, p_acc, p_vals = predict(label_test_list, data_test_list, m)
#            if p_acc[0] > accmax:
#                accmax = p_acc[0]
#                s_best = s
#                c_best = c
#                p_label_best = p_labels
#    
#    accmax_list.append(accmax)
#    s_best_list.append(s_best)
#    c_best_list.append(c_best)
#    plabel_best_list.append(plabel_best)
#
#print(accmax_list)
#print(sum(accmax_list)/len(accmax_list))
#print(s_best_list)
#print(c_best_list)
