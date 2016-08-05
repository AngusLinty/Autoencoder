import numpy as np
import scipy.io as sio
mat_data = sio.loadmat('djc_20131027_eeg.mat')
train_de_data = mat_data['train_de']
test_de_data = mat_data['test_de']

# %% Normalization definition
def normalize(input=[499, 310]):
    temp_r = np.array(input)
    temp_c = np.array(input)
    rows_sum = input.sum(axis=1)
    cols_sum = input.sum(axis=0)

    temp_r = temp_r/rows_sum[:, np.newaxis]
    temp_c = temp_c/cols_sum

    return(temp_r, temp_c)

    #print(input)
    #print(temp_c)
    #print(input.sum(axis=1))
    #print(temp_c.sum(axis=0))


train_r, train_c = normalize(train_de_data)
test_r, test_c = normalize(test_de_data)
#print(train_r, train_r.sum(axis=1)[10])
#print(train_c, train_c.sum(axis=0)[10])
sio.savemat('train_de_norm_r.mat', {'train_de_norm_r': train_r})
sio.savemat('train_de_norm_c.mat', {'train_de_norm_c': train_c})
sio.savemat('test_de_norm_r.mat', {'test_de_norm_r': test_r})
sio.savemat('test_de_norm_c.mat', {'test_de_norm_c': test_c})
