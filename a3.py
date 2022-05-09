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

#######################################################  A2 e STARTS HERE  #############################################################################

# K-Fold sets creation
kf = KFold(n_splits=5, shuffle=False, random_state=None)

# Function for testing number of first hidden layer nodes
def fit_test(hidden_nodes=vocablen, lr=0.001, m=0.0, bs=100, epochs_num=100, plot=[("loss","val_loss")]):  
  
  # Arrays for storing loss sum, accuracy sum and MSE sum of all five training iterations
  loss_sum = np.zeros(epochs_num)
  val_loss_sum = np.zeros(epochs_num)
  acc_sum = np.zeros(epochs_num)
  val_acc_sum = np.zeros(epochs_num)
  mse_sum = np.zeros(epochs_num)
  val_mse_sum = np.zeros(epochs_num)
  
  # Dictionary for calling arrays with Keras known names
  arr_dict ={
      "loss" : loss_sum,
      "val_loss" : val_loss_sum,
      "accuracy" : acc_sum,
      "val_accuracy" : val_acc_sum,
      "mean_squared_error" : mse_sum,
      "val_mean_squared_error" : val_mse_sum
  }

  #Define model object
  model = None

  for i,(trn,tst) in enumerate(kf.split(train_set)):
      # Create model
      model = Sequential()

      model.add(Dense(hidden_nodes, input_dim=vocablen))
      model.add(LeakyReLU(alpha=0.1))# Add an activation function of the previous (hidden) layer
      model.add(Dense(20, activation="sigmoid"))

      # Optimizers to choose from for the neural network
      sgd = optimizers.SGD(learning_rate=lr, momentum=m)
      #adam = optimizers.Adam()

      # Compile model
      model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=["accuracy", "binary_crossentropy", "mean_squared_error"])

      # Fit model
      model.fit(train_set[trn], train_label[trn], epochs=epochs_num, batch_size=bs, verbose=1, validation_data=(train_set[tst],train_label[tst]))

      loss_sum += model.history.history["loss"]
      val_loss_sum += model.history.history["val_loss"]
      acc_sum += model.history.history["accuracy"]
      val_acc_sum += model.history.history["val_accuracy"]
      mse_sum += model.history.history["mean_squared_error"]
      val_mse_sum += model.history.history["val_mean_squared_error"]

  # Create plot for every metric we ask for
  for j in range(len(plot)):
      # Plot loss function on training and validation sets to compare
      plt.style.use('ggplot')
      plt.plot(np.arange(epochs_num), arr_dict[plot[j][0]]/5, np.arange(epochs_num), arr_dict[plot[j][1]]/5)
      plt.xlabel("epochs")
      plt.ylabel(plot[j][0])
      if plot[j][0] == "accuracy" or plot[j][0] == "val_accuracy":
            plt.legend(["Train", "Validation"], loc ="lower right")
      else:
            plt.legend(["Train", "Validation"], loc ="upper right")
      if hidden_nodes == classnum:
          plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/"+plot[j][0]+".png")
      
      if hidden_nodes == (vocablen+classnum)/2:
          plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden4270/"+plot[j][0]+".png")
      
      if hidden_nodes == vocablen:
          plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/"+plot[j][0]+".png")

      plt.show()

  print(model.evaluate(train_set[tst], train_label[tst], verbose=1)) # Evaluation with the test part of each fold
  print(model.evaluate(test_set, test_label, verbose=1)) # Evaluation with the given test set

  return arr_dict

# Call fit_test for 3 different hidden layer nodes number and plot every metric for each of them through epochs
small_hidden_metrics = fit_test(hidden_nodes=classnum, lr=0.1, epochs_num=200, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/loss.npy", np.asarray(small_hidden_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/val_loss.npy", np.asarray(small_hidden_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/accuracy.npy", np.asarray(small_hidden_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/val_accuracy.npy", np.asarray(small_hidden_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/mean_squared_error.npy", np.asarray(small_hidden_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/val_mean_squared_error.npy", np.asarray(small_hidden_metrics["val_mean_squared_error"]))

medium_hidden_metrics = fit_test(hidden_nodes=(classnum+vocablen)/2, lr=0.1, epochs_num=200, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden4270/loss.npy", np.asarray(medium_hidden_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden4270/val_loss.npy", np.asarray(medium_hidden_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden4270/accuracy.npy", np.asarray(medium_hidden_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden4270/val_accuracy.npy", np.asarray(medium_hidden_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden4270/mean_squared_error.npy", np.asarray(medium_hidden_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden4270/val_mean_squared_error.npy", np.asarray(medium_hidden_metrics["val_mean_squared_error"]))

large_hidden_metrics = fit_test(hidden_nodes=vocablen, lr=0.1, epochs_num=200, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/loss.npy", np.asarray(large_hidden_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/val_loss.npy", np.asarray(large_hidden_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/accuracy.npy", np.asarray(large_hidden_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/val_accuracy.npy", np.asarray(large_hidden_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/mean_squared_error.npy", np.asarray(large_hidden_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/val_mean_squared_error.npy", np.asarray(large_hidden_metrics["val_mean_squared_error"]))

# Load data to create plots for comparing
small_val_loss = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/val_loss.npy")
medium_val_loss = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden4270/val_loss.npy")
large_val_loss = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/val_loss.npy")

small_val_accuracy = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/val_accuracy.npy")
medium_val_accuracy = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden4270/val_accuracy.npy")
large_val_accuracy = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/val_accuracy.npy")

small_val_mse = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/val_mean_squared_error.npy")
medium_val_mse = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden4270/val_mean_squared_error.npy")
large_val_mse = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/val_mean_squared_error.npy")

# Plot of the Validation Loss of different second hidden layer nodes
plt.style.use('ggplot')
plt.plot(np.arange(200), small_val_loss/5, np.arange(200), medium_val_loss/5, np.arange(200), large_val_loss/5)
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.legend(["20 nodes","4270 nodes", "8520 nodes"], loc ="upper right")
plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/compare_plots/val_loss.png")
plt.show()

# Plot of the Validation MSE of different second hidden layer nodes
plt.style.use('ggplot')
plt.plot(np.arange(200), small_val_mse/5, np.arange(200), medium_val_mse/5, np.arange(200), large_val_mse/5)
plt.xlabel("Epochs")
plt.ylabel("Validation MSE")
plt.legend(["20 nodes", "4270 nodes", "8520 nodes"], loc ="upper right")
plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/compare_plots/val_mean_squared_error.png")
plt.show()

# Plot of the Validation Accuracy of different second hidden layer nodes
plt.style.use('ggplot')
plt.plot(np.arange(200), small_val_accuracy/5, np.arange(200), medium_val_accuracy/5, np.arange(200), large_val_accuracy/5)
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend(["20 nodes", "4270 nodes", "8520 nodes"], loc ="lower right")
plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/compare_plots/val_accuracy.png")
plt.show()


#######################################################  A2 st STARTS HERE  #############################################################################

# Function for testing number of the second hidden layer nodes
def fit_test_sec(hidden_nodes=vocablen, second_hidden_nodes=vocablen, lr=0.001, m=0.0, bs=100, epochs_num=100, plot=[("loss","val_loss")]):
    # Arrays for storing loss sum, accuracy sum and MSE sum of all five training iterations
    loss_sum = np.zeros(epochs_num)
    val_loss_sum = np.zeros(epochs_num)
    acc_sum = np.zeros(epochs_num)
    val_acc_sum = np.zeros(epochs_num)
    mse_sum = np.zeros(epochs_num)
    val_mse_sum = np.zeros(epochs_num)
    
    # Dictionary for calling arrays with Keras known names
    arr_dict ={
        "loss" : loss_sum,
        "val_loss" : val_loss_sum,
        "accuracy" : acc_sum,
        "val_accuracy" : val_acc_sum,
        "mean_squared_error" : mse_sum,
        "val_mean_squared_error" : val_mse_sum
    }

    #Define model object
    model = None

    for i,(trn,tst) in enumerate(kf.split(train_set)):
        # Create model
        model = Sequential()

        # First hidden layer
        model.add(Dense(hidden_nodes, input_dim=vocablen))
        model.add(LeakyReLU(alpha=0.1))# Add an activation function of the previous (hidden) layer

        # Second hidden layer
        model.add(Dense(second_hidden_nodes))
        model.add(LeakyReLU(alpha=0.1))# Add an activation function of the previous (second hidden) layer

        # Output layer
        model.add(Dense(20, activation="sigmoid"))

        # Optimizers to choose from for the neural network
        sgd = optimizers.SGD(learning_rate=lr, momentum=m)
        #adam = optimizers.Adam()

        # Compile model
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=["accuracy", "binary_crossentropy", "mean_squared_error"])

        # Fit model
        model.fit(train_set[trn], train_label[trn], epochs=epochs_num, batch_size=bs, verbose=1, validation_data=(train_set[tst],train_label[tst]))

        # Add new values of metrics to sums
        loss_sum += model.history.history["loss"]
        val_loss_sum += model.history.history["val_loss"]
        acc_sum += model.history.history["accuracy"]
        val_acc_sum += model.history.history["val_accuracy"]
        mse_sum += model.history.history["mean_squared_error"]
        val_mse_sum += model.history.history["val_mean_squared_error"]

    # Create plot for every metric we ask for
    for j in range(len(plot)):
        # Plot loss function on training and validation sets to compare
        plt.style.use('ggplot')
        plt.plot(np.arange(epochs_num), arr_dict[plot[j][0]]/5, "r", np.arange(epochs_num), arr_dict[plot[j][1]]/5, "b")
        plt.xlabel("epochs")
        plt.ylabel(plot[j][0])
        if plot[j][0] == "accuracy" or plot[j][0] == "val_accuracy":
            plt.legend(["Train", "Validation"], loc ="lower right")
        else:
            plt.legend(["Train", "Validation"], loc ="upper right")

        if second_hidden_nodes == 10:
            if lr == 0.1:
                if m == 0.0:
                    plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/"+plot[j][0]+".png")
                elif m == 0.6:
                    plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/m0.6/"+plot[j][0]+".png")
            elif lr == 0.05:
                if m == 0.6:
                    plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.05/m0.6/"+plot[j][0]+".png")
            elif lr == 0.001:
                if m == 0.0:
                    plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0/"+plot[j][0]+".png")
                elif m == 0.2:
                    plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.2/"+plot[j][0]+".png")
                elif m == 0.6:
                    plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.6/"+plot[j][0]+".png")
        if second_hidden_nodes == 20:
            plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden20/"+plot[j][0]+".png")

        if second_hidden_nodes == 40:
            plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden40/"+plot[j][0]+".png")

        if second_hidden_nodes == classnum:
            plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden20/"+plot[j][0]+".png")
        
        if second_hidden_nodes == (vocablen+classnum)/2:
            plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden4270/"+plot[j][0]+".png")
        
        if second_hidden_nodes == vocablen:
            plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden8520/"+plot[j][0]+".png")

        if second_hidden_nodes == 2 * vocablen:
            plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden17040/"+plot[j][0]+".png")

        plt.show()

    print(model.evaluate(train_set[tst], train_label[tst], verbose=1)) # Evaluation with the test part of each fold
    print(model.evaluate(test_set, test_label, verbose=1)) # Evaluation with the given test set

    # Returns the metrics arrays if we want to compare some models afterwards
    return arr_dict


# Fit model for different number of second hidden layer nodes, with 8520 nodes on first hidden layer
small_sec_hidden_metrics = fit_test_sec(second_hidden_nodes=classnum, lr=0.1, epochs_num=200, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden20/loss.npy", np.asarray(small_sec_hidden_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden20/val_loss.npy", np.asarray(small_sec_hidden_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden20/accuracy.npy", np.asarray(small_sec_hidden_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden20/val_accuracy.npy", np.asarray(small_sec_hidden_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden20/mean_squared_error.npy", np.asarray(small_sec_hidden_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden20/val_mean_squared_error.npy", np.asarray(small_sec_hidden_metrics["val_mean_squared_error"]))

medium_sec_hidden_metrics = fit_test_sec(second_hidden_nodes=(classnum+vocablen)/2, lr=0.1, epochs_num=200, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden4270/loss.npy", np.asarray(medium_sec_hidden_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden4270/val_loss.npy", np.asarray(medium_sec_hidden_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden4270/accuracy.npy", np.asarray(medium_sec_hidden_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden4270/val_accuracy.npy", np.asarray(medium_sec_hidden_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden4270/mean_squared_error.npy", np.asarray(medium_sec_hidden_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden4270/val_mean_squared_error.npy", np.asarray(medium_sec_hidden_metrics["val_mean_squared_error"]))

large_sec_hidden_metrics = fit_test_sec(second_hidden_nodes=vocablen, lr=0.1, epochs_num=200, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden8520/loss.npy", np.asarray(large_sec_hidden_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden8520/val_loss.npy", np.asarray(large_sec_hidden_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden8520/accuracy.npy", np.asarray(large_sec_hidden_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden8520/val_accuracy.npy", np.asarray(large_sec_hidden_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden8520/mean_squared_error.npy", np.asarray(large_sec_hidden_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden8520/val_mean_squared_error.npy", np.asarray(large_sec_hidden_metrics["val_mean_squared_error"]))

xlarge_sec_hidden_metrics = fit_test_sec(second_hidden_nodes=2*vocablen, lr=0.1, epochs_num=200, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden17040/loss.npy", np.asarray(xlarge_sec_hidden_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden17040/val_loss.npy", np.asarray(xlarge_sec_hidden_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden17040/accuracy.npy", np.asarray(xlarge_sec_hidden_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden17040/val_accuracy.npy", np.asarray(xlarge_sec_hidden_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden17040/mean_squared_error.npy", np.asarray(xlarge_sec_hidden_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/sec_hidden17040/val_mean_squared_error.npy", np.asarray(xlarge_sec_hidden_metrics["val_mean_squared_error"]))

# Plot of the Validation Loss of different second hidden layer nodes
plt.style.use('ggplot')
plt.plot(np.arange(200), small_sec_val_loss/5, np.arange(200), medium_sec_val_loss/5, np.arange(200), large_sec_val_loss/5, np.arange(200), xlarge_sec_val_loss/5)
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.legend(["20 nodes", "4270 nodes", "8520 nodes", "17040 nodes"], loc ="upper left")
plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/compare_plots_sec/val_loss.png")
plt.show()

# Plot of the Validation MSE of different second hidden layer nodes
plt.style.use('ggplot')
plt.plot(np.arange(200), small_sec_val_mse/5, np.arange(200), medium_sec_val_mse/5, np.arange(200), large_sec_val_mse/5, np.arange(200), xlarge_sec_val_mse/5)
plt.xlabel("Epochs")
plt.ylabel("Validation MSE")
plt.legend(["20 nodes", "4270 nodes", "8520 nodes", "17040 nodes"], loc ="upper right")
plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/compare_plots_sec/val_mean_squared_error.png")
plt.show()

# Plot of the Validation Accuracy of different second hidden layer nodes
plt.style.use('ggplot')
plt.plot(np.arange(200), small_sec_val_accuracy/5, np.arange(200), medium_sec_val_accuracy/5, np.arange(200), large_sec_val_accuracy/5, np.arange(200), xlarge_sec_val_accuracy/5)
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend(["20 nodes", "4270 nodes", "8520 nodes", "17040 nodes"], loc ="upper right")
plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden8520/compare_plots_sec/val_accuracy.png")
plt.show()

# Fit model for different number of second hidden layer nodes, with 20 nodes on first hidden layer
small_sec_hidden_metrics = fit_test_sec(hidden_nodes=20, second_hidden_nodes=10, lr=0.1, epochs_num=200, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/loss.npy", np.asarray(small_sec_hidden_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/val_loss.npy", np.asarray(small_sec_hidden_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/accuracy.npy", np.asarray(small_sec_hidden_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/val_accuracy.npy", np.asarray(small_sec_hidden_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/mean_squared_error.npy", np.asarray(small_sec_hidden_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/val_mean_squared_error.npy", np.asarray(small_sec_hidden_metrics["val_mean_squared_error"]))

medium_sec_hidden_metrics = fit_test_sec(hidden_nodes=20, second_hidden_nodes=20, lr=0.1, epochs_num=200, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden20/loss.npy", np.asarray(medium_sec_hidden_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden20/val_loss.npy", np.asarray(medium_sec_hidden_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden20/accuracy.npy", np.asarray(medium_sec_hidden_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden20/val_accuracy.npy", np.asarray(medium_sec_hidden_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden20/mean_squared_error.npy", np.asarray(medium_sec_hidden_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden20/val_mean_squared_error.npy", np.asarray(medium_sec_hidden_metrics["val_mean_squared_error"]))

large_sec_hidden_metrics = fit_test_sec(hidden_nodes=20, second_hidden_nodes=40, lr=0.1, epochs_num=200, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden40/loss.npy", np.asarray(large_sec_hidden_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden40/val_loss.npy", np.asarray(large_sec_hidden_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden40/accuracy.npy", np.asarray(large_sec_hidden_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden40/val_accuracy.npy", np.asarray(large_sec_hidden_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden40/mean_squared_error.npy", np.asarray(large_sec_hidden_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden40/val_mean_squared_error.npy", np.asarray(large_sec_hidden_metrics["val_mean_squared_error"]))


# Load data to create plots for comparing different second hidden layers with 20 nodes on first hidden layer
small_sec_val_loss = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/val_loss.npy")
medium_sec_val_loss = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden20/val_loss.npy")
large_sec_val_loss = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden40/val_loss.npy")

small_sec_val_accuracy = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/val_accuracy.npy")
medium_sec_val_accuracy = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden20/val_accuracy.npy")
large_sec_val_accuracy = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden40/val_accuracy.npy")

small_sec_val_mse = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/val_mean_squared_error.npy")
medium_sec_val_mse = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden20/val_mean_squared_error.npy")
large_sec_val_mse = load_npy("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden40/val_mean_squared_error.npy")


# Plot of the Validation Loss of different second hidden layer nodes, with first hidden layer with 20 nodes
plt.style.use('ggplot')
plt.plot(np.arange(200), small_sec_val_loss/5, np.arange(200), medium_sec_val_loss/5, np.arange(200), large_sec_val_loss/5)
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.legend(["10 nodes", "20 nodes", "40 nodes"], loc ="lower right")
plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/compare_plots_sec/val_loss.png")
plt.show()

# Plot of the Validation Loss of different second hidden layer nodes, with first hidden layer with 20 nodes
plt.style.use('ggplot')
plt.plot(np.arange(200), small_sec_val_mse/5, np.arange(200), medium_sec_val_mse/5, np.arange(200), large_sec_val_mse/5)
plt.xlabel("Epochs")
plt.ylabel("Validation MSE")
plt.legend(["10 nodes", "20 nodes", "40 nodes"], loc ="upper right")
plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/compare_plots_sec/val_mean_squared_error.png")
plt.show()

# Plot of the Validation Loss of different second hidden layer nodes, with first hidden layer with 20 nodes
plt.style.use('ggplot')
plt.plot(np.arange(200), small_sec_val_accuracy/5, np.arange(200), medium_sec_val_accuracy/5, np.arange(200), large_sec_val_accuracy/5)
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.legend(["10 nodes", "20 nodes", "40 nodes"], loc ="lower right")
plt.savefig("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/compare_plots_sec/val_accuracy.png")
plt.show()


#######################################################  A3 STARTS HERE  #############################################################################



# Fit best found network for different learning rates and momentum values
advanced_metrics = fit_test_sec(hidden_nodes=20, second_hidden_nodes=10, lr=0.001, m=0.0, epochs_num=1000, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0/loss.npy", np.asarray(advanced_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0/val_loss.npy", np.asarray(advanced_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0/accuracy.npy", np.asarray(advanced_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0/val_accuracy.npy", np.asarray(advanced_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0/mean_squared_error.npy", np.asarray(advanced_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0/val_mean_squared_error.npy", np.asarray(advanced_metrics["val_mean_squared_error"]))

advanced_metrics = fit_test_sec(hidden_nodes=20, second_hidden_nodes=10, lr=0.001, m=0.2, epochs_num=1000, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.2/loss.npy", np.asarray(advanced_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.2/val_loss.npy", np.asarray(advanced_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.2/accuracy.npy", np.asarray(advanced_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.2/val_accuracy.npy", np.asarray(advanced_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.2/mean_squared_error.npy", np.asarray(advanced_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.2/val_mean_squared_error.npy", np.asarray(advanced_metrics["val_mean_squared_error"]))

advanced_metrics = fit_test_sec(hidden_nodes=20, second_hidden_nodes=10, lr=0.001, m=0.6, epochs_num=1000, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.6/loss.npy", np.asarray(advanced_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.6/val_loss.npy", np.asarray(advanced_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.6/accuracy.npy", np.asarray(advanced_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.6/val_accuracy.npy", np.asarray(advanced_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.6/mean_squared_error.npy", np.asarray(advanced_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.001/m0.6/val_mean_squared_error.npy", np.asarray(advanced_metrics["val_mean_squared_error"]))

advanced_metrics = fit_test_sec(hidden_nodes=20, second_hidden_nodes=10, lr=0.05, m=0.6, epochs_num=200, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.05/m0.6/loss.npy", np.asarray(advanced_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.05/m0.6/val_loss.npy", np.asarray(advanced_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.05/m0.6/accuracy.npy", np.asarray(advanced_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.05/m0.6/val_accuracy.npy", np.asarray(advanced_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.05/m0.6/mean_squared_error.npy", np.asarray(advanced_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/lr0.05/m0.6/val_mean_squared_error.npy", np.asarray(advanced_metrics["val_mean_squared_error"]))

advanced_metrics = fit_test_sec(hidden_nodes=20, second_hidden_nodes=10, lr=0.1, m=0.6, epochs_num=200, plot=[("loss","val_loss"),("accuracy", "val_accuracy"), ("mean_squared_error", "val_mean_squared_error")])
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/m0.6/loss.npy", np.asarray(advanced_metrics["loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/m0.6/val_loss.npy", np.asarray(advanced_metrics["val_loss"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/m0.6/accuracy.npy", np.asarray(advanced_metrics["accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/m0.6/val_accuracy.npy", np.asarray(advanced_metrics["val_accuracy"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/m0.6/mean_squared_error.npy", np.asarray(advanced_metrics["mean_squared_error"]))
np.save("/content/drive/MyDrive/Υπολογιστική Νοημοσύνη/1η εργασία/hidden20/sec_hidden10/m0.6/val_mean_squared_error.npy", np.asarray(advanced_metrics["val_mean_squared_error"]))



