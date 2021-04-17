# -*- coding: utf-8 -*-
"""
This takes the data for one ppt in one session and organises it into
two arrays, one for left and one for right motor imagery data. 
All data not related to the MI trials is disregarded.
"""
import numpy as np
import scipy.io as spio
#change path
path = 'C:/Users/hlw69/Documents/gtecspringschool/stroke/stroke/P1_pre_training.mat'
mat = spio.loadmat(path, squeeze_me=True)

print("The keys of this data dictionary for this ppt are :", mat.keys())
triggers = mat['trig']
fs = mat['fs']
eeg_data = mat['y']
print("There are ", len(triggers), " triggers")
print(max(triggers)) 
print("sampling frequency is ", mat['fs'])
print("Length of data set is ", len(eeg_data))




# get indices that indicate change in blocks, such as 000s then 1111s in order to split these
v=np.asarray(triggers)
indices = np.where(np.diff(v,prepend=np.nan))[0]
#print(indices)

listvalues = []
listclasses = []
ind = 0
for ind in range(0, len(indices)-1):
    listclasses.append(triggers[indices[ind]]) # means we know which class each block relates to
    listvalues.append(eeg_data[indices[ind]:indices[ind+1]]) # gets the block as a subset


# splits the data into the two separate conditions left and right, removes 0 triggers
# this also means each trial should now be equal length so can be put into an array
left_class = []
left_value = []
right_class = []
right_value = []

for i,x in enumerate(listclasses):
    if x ==1:
        left_class.append(listclasses[i])
        left_value.append(listvalues[i])
    elif x ==-1:
        right_class.append(listclasses[i])
        right_value.append(listvalues[i])
        
# only keep data with relevant time points, 2 till 8 seconds
start_point = 2*fs
end_point = 8*fs
for ind in range(0, len(left_value)-1):
    left_value[ind] = left_value[ind][start_point:end_point]
for ind in range(0, len(right_value)-1):
    right_value[ind] = right_value[ind][start_point:end_p
                                            
        

left_data = np.asarray(left_value)
right_data = np.asarray(right_value)
print(left_data.shape, right_data.shape)
left_labels = np.ones(40)
right_labels = np.zeros(40) - 1
print(right_labels)





