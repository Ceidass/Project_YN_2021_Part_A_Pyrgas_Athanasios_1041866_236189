import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
#from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
import matplotlib.pyplot as plt

vocablen = 8520 # Number of vocabulary words
classnum = 20 # Number of classes in our problem (outputs)

# Function for export datasets from txt to numpy array
# This function saves dataset to .npy file and also return it in case we want it now
def set_export(load_path, save_path=None):

    set = np.zeros((0,vocablen))

    with open(load_path, "r") as txt:
        #Iterating through lines
        for line in txt:
            sentfind = False
            wordfind = False
            insentcount = False
            inwordcount = False
            sentNo = ""
            wordNo = ""
            word = ""
            total_words = 0
            spaces = 0
            vect = np.zeros((1,vocablen)) # Helper vector

            # Iterating through line's characters
            for char in range(0,len(line)-1):
                if not(sentfind): # If we have NOT found the "number of sentences" label 
                    if line[char] == "<":
                        sentfind = True # Indicates that we HAVE found the "number of sentences" label
                        insentcount = True # Indicates that we are inside the "number of sentences" label 
                else:   # If we HAVE found the "number of sentences" label 
                    if insentcount: # If we are inside the "number of sentences" label
                        if line[char] != "<":
                            if line[char] == ">":
                                insentcount = False # Indicates that we EXIT the "number of sentences" label
                                sentNo = int(sentNo)
                            else:
                                sentNo += line[char]
                    else: # If we have exited the "number of sentences" label
                        if not(wordfind): # If we have NOT found a "number of words" label
                            if line[char] == "<":
                                wordfind = True # Indicates that we HAVE found a "number of words" label
                                inwordcount = True # Indicates that we are inside a "number of words" label
                        else: # If we HAVE found a "number of words" label
                            if inwordcount: # If we are inside a "number of words" label
                                if line[char] != "<":
                                    if line[char] == ">":
                                        wordNo = int(wordNo)
                                        total_words += wordNo
                                        #wordNo = "" # try # Re-initializes wordNo to an empty string for the next sentence
                                        inwordcount = False  #Re-initializes inwordcount to False for the next sentence
                                    else:   
                                        wordNo += line[char]
                            else: # If we have exited the "number of words" label
                                if line[char] == " " and line[char-1] != ">": # If we hit a SPACE character
                                    word = int(word)
                                    vect[0,word] += 1
                                    #vect = np.append(vect,) # try
                                    spaces += 1
                                    word = "" # Re-initializes word to an empty string for the next word
                                    if spaces == wordNo:
                                        spaces = 0 # Re-initializes spaces to zero for the next sentence
                                        wordfind = False
                                        wordNo = "" # Re-initializes wordNo to an empty string for the next sentence
                                else:
                                    word += line[char]

            set = np.append(set, vect, axis=0)
            #trainset = np.append(trainset, vect/sentNo, axis=0) # try # It will be good if we had word embedings 
            #print(set.shape) # debug

    #If we have set a save_path
    if save_path != None:
      #Save dataset in array format
      data = np.asarray(set)
      np.save(save_path, data)

    return set

def label_export(load_path, save_path=None):
    set = np.zeros((0,classnum))
    with open(load_path, "r") as txt:
        helper = np.zeros((1,classnum))
        #Iterating through lines
        for line in txt:
            # Iterating through line's characters
            for char in range(0,len(line)-1):
                if line[char] == "1":
                    helper[char/2] = 2
        set = np.append(set,helper)
    
    np.save(save_path,set)

    return set


# Function for loading data from .npy format
def load_npy(load_path):
    data = np.load(load_path)
    return data

# We use this part of code if we want to export data from .txt to .npy format or if we want to store them to a variable 
set_export("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/train-data.dat", "/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/train-data.npy")
set_export("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/test-data.dat", "/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/test-data.npy")

label_export("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/train-label.dat", "/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/train-label.npy")
label_export("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/test-label.dat", "/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/test-label.npy")

# We use this part of code if we have already store our data to .npy format and we want to load them
train_set = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/train-data.npy")
test_set = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/test-data.npy")

train_label = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/train-label.npy")
test_label = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/test-label.npy")

# Apply first the normalization
#train_set /= np.linalg.norm(train_set)
#test_set /= np.linalg.norm(test_set)

# We use this part of code if we want our data sets to be centered
# Centering data
#train_set -= train_set.mean()
#test_set -= test_set.mean()

# We use this part of code if we want our data sets to be normalized
# Normalizing data sets
#train_set /= np.linalg.norm(train_set)
#test_set /= np.linalg.norm(test_set)

