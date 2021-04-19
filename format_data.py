"""
This takes the data for one ppt in one session (.mat files) and organises it into
two arrays, one for left and one for right motor imagery data. 
All data not related to the MI trials is disregarded. Also option to flatten into 2D arrays or keep as 3D arrays depending on the classifier to be used.
"""
import numpy as np
import scipy.io as spio




path_training = 'C:/Users/hlw69/Documents/gtecspringschool/stroke/stroke/data/P1_pre_training.mat'
path_test = 'C:/Users/hlw69/Documents/gtecspringschool/stroke/stroke/data/P1_pre_test.mat'
train_mat = spio.loadmat(path_training, squeeze_me=True)
test_mat = spio.loadmat(path_test, squeeze_me=True)



def describe_organise_data(mat):
    print("The keys of this data dictionary for this ppt are :", mat.keys())
    triggers = mat['trig']
    fs = mat['fs']
    eeg_data = mat['y']
    print("There are ", len(triggers), " triggers") 
    print("sampling frequency is ", mat['fs'])
    print("Length of data set is ", len(eeg_data))
    return triggers, fs, eeg_data


# get indices that indicate change in blocks, such as 000s then 1111s in order to split these
def format_data(triggers, fs, eeg_data):
    v=np.asarray(triggers)
    indices = np.where(np.diff(v,prepend=np.nan))[0] 

    listvalues = []
    listclasses = []
    ind = 0
    
    for ind in range(0, len(indices)-1):
        listclasses.append(triggers[indices[ind]]) # means we know which class each block relates to
        listvalues.append(eeg_data[indices[ind]:indices[ind+1]]) # gets the block as a subset
         
    left_class = []
    left_value = []
    right_class = []
    right_value = []
    for i,x in enumerate(listclasses):
        if x == 1:
            left_class.append(listclasses[i])
            left_value.append(listvalues[i])
        elif x == -1:
            right_class.append(listclasses[i])
            right_value.append(listvalues[i])
        
 
    left_data = np.array([np.array(x) for x in left_value])
    right_data = np.array([np.array(x) for x in right_value])
    left_labels = np.ones(left_data.shape[0])
    right_labels = np.zeros(right_data.shape[0]) - 1
    
    data = np.concatenate((left_data,right_data))
    labels = np.concatenate((left_labels,right_labels))
    return data,labels




# splits the data into the two separate conditions left and right, removes 0 triggers
# this also means each trial should now be equal length so can be put into an array
test_triggers, test_fs, test_eeg = describe_organise_data(test_mat)
train_triggers, train_fs, train_eeg = describe_organise_data(train_mat)

train_data, train_labels =format_data(train_triggers, train_fs, train_eeg)
test_data, test_labels = format_data(test_triggers, test_fs, test_eeg )




   
    # have to flatten into two dimensions, cannot take 16 channels, 
train_data_2D = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
test_data_2D = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])
    listvalues = []
    listclasses = []
    ind = 0
    
    for ind in range(0, len(indices)-1):
        listclasses.append(triggers[indices[ind]]) # means we know which class each block relates to
        listvalues.append(eeg_data[indices[ind]:indices[ind+1]]) # gets the block as a subset
         
    left_class = []
    left_value = []
    right_class = []
    right_value = []
    for i,x in enumerate(listclasses):
        if x == 1:
            left_class.append(listclasses[i])
            left_value.append(listvalues[i])
        elif x == -1:
            right_class.append(listclasses[i])
            right_value.append(listvalues[i])
        
 
    left_data = np.array([np.array(x) for x in left_value])
    right_data = np.array([np.array(x) for x in right_value])
    left_labels = np.ones(left_data.shape[0])
    right_labels = np.zeros(right_data.shape[0]) - 1
    
    data = np.concatenate((left_data,right_data))
    labels = np.concatenate((left_labels,right_labels))
    return data,labels




# splits the data into the two separate conditions left and right, removes 0 triggers
# this also means each trial should now be equal length so can be put into an array
test_triggers, test_fs, test_eeg = describe_organise_data(test_mat)
train_triggers, train_fs, train_eeg = describe_organise_data(train_mat)

train_data, train_labels =format_data(train_triggers, train_fs, train_eeg)
test_data, test_labels = format_data(test_triggers, test_fs, test_eeg )




   
    # have to flatten into two dimensions, cannot take 16 channels, 
train_data_2D = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
test_data_2D = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

