import os 
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2 

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import visualkeras
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report

import time 
from datetime import date



################# prepare dataset ###################


# path of dataset 
dataset_path = './Car-Bike-Dataset'



def get_info(dataset_path,vehicle_type,class_number,reshape_width,reshape_height):      
   
    
   
    """ 
    
        stores the path of the vehicle class and the image (numpy array) in 
        a dataframe
        
        
    """
    
    
    
    
    # get the paths of the images of the vehicle type
    vehicle_paths = os.listdir(os.path.join(dataset_path,vehicle_type))
    
    vehicle_imgs_path = []
    
    for i in range(0,len(vehicle_paths)):
        
        vehicle_imgs_path.append(os.path.join(dataset_path,
                                              vehicle_type,
                                              vehicle_paths[i]))
    
    
    
    # replace back slash with forward slash
    vehicle_imgs_path = list(map(lambda x : x.replace('\\','/'),vehicle_imgs_path))
    
    
    #cv2.imread('Car-Bike-Dataset/Car/Car (1).jpeg')
    
    # 
    
    vehicle_imgs = []
    
    
    # read in image and store it in the dataframe
    for i in range(len(vehicle_imgs_path)):
        
        
        img_bgr = cv2.imread(vehicle_imgs_path[i])
        
        img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
        
        
       
        img_rs = cv2.resize(img_rgb, (reshape_width, reshape_height))
        
        
        img_rs = img_rs/255
        
        vehicle_imgs.append(img_rs)
    
    
    # class 0 is for car 
    # class 1 is for bike
    class_list_vehicle = [class_number for i in range(len(vehicle_imgs))]
    
    
    
    
    
    
    
    
    df =    pd.DataFrame(data = {'path': vehicle_imgs_path,
                                  'image': vehicle_imgs,
                                  'class': class_list_vehicle})




    return df







def build_model(reshape_height,reshape_width,X_train,X_test):
    
    """
        builds model
    """
    
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(reshape_height,reshape_width,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))



    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10,activation = 'relu'))

    # N class
    model.add(layers.Dense(2, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss = 'CategoricalCrossentropy',
                  metrics=['accuracy'])


    visualkeras.layered_view(model, legend=True,show_dimension =True)

    

    model.fit(x = X_train, 
              y = y_train,
              epochs=10)



    return model




#  # we need to define an input shape for the CNN 
 # here it is 100 x 100 
reshape_width = 100
reshape_height = 100


# call the get_info function
df_car = get_info(dataset_path,'Car',0,reshape_width,reshape_height)

df_bike = get_info(dataset_path,'Bike',1,reshape_width,reshape_height)



# concat both of the dataset 
df = pd.concat((df_car,df_bike),axis = 0)





########## Convert so we can use the train_test_split ############


# we use the OneHotEncoder for the classes 
# so we get a vector that has the right class in a vector where the index that has '1' 
# is the correct class
encoder = OneHotEncoder(sparse=False)


y = encoder.fit_transform(np.array(df['class']).reshape(-1,1))



# because the dataset are color images, we need to store each image as 
# reshape_heightxrehape_widthx3 numpy array  
X = np.zeros((df.shape[0],reshape_height,reshape_width,3))

# convert to numpy array 
for i in range(df.shape[0]):
    
    X[i,:,:,:] = df['image'].iloc[i]
    



# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size=0.33,
                                                    random_state=42)





########################## building cnn model #####################################

# call the function to build the cnn
model = build_model(reshape_height,reshape_width,X_train,X_test)

# train the model with the data




###################### evaluation of the model ################################

# in this list the predictions are stored 
y_pred = []


for i in range(len(X_test)):
    
    # get the test image 
    test_image = X_test[i,:,:,:]
    
    # we need to reshape the test image: (N,N,3) -> (1,N,N,3)
    test_image_for_prediction = test_image[np.newaxis,:,:,:]

    # now get the prediction
    # the prediciton is a vector containing the predictions for each class
    prediction = model.predict(test_image_for_prediction)

    # the index with the highest propabilty in the vector is the predicted class
    predicted_class = np.argmax(prediction)
    
    # store the predictions 
    y_pred.append(predicted_class)






# get the correct classes from the test data
y_true = np.argmax(y_test,axis =1)


y_pred = np.array((y_pred))


# get the confusion matrix 
cm = confusion_matrix(y_true,y_pred)

# show the confusion matrix 
disp = ConfusionMatrixDisplay(cm,display_labels=['Car','Bike'])

disp.plot()


# get precission,recall and f1 scores for each class 
report = classification_report(y_true,y_pred,output_dict=True)

report_df = pd.DataFrame(data  = report)


# get f1_scores

f1_scores_list = []

# amount of classes in the dataset
n = 2

for i in range(n):
    
    f1_scores_list.append(report_df[str(i)].loc['f1-score'])

    




fig,ax = plt.subplots(1,1)

ax.bar(x = [0,1] ,height = f1_scores_list,width = 0.3,tick_label = ['Car','Bike'])
ax.set_title("F1-Scores")

# Adding value labels on top of the bars
for index, value in enumerate(f1_scores_list):
    
    print(value)
    
    value = round(value,3)
    ax.text(index, value + 0.001, f'{value}', ha='center')


plt.savefig('f1_score.png', format='png')



#################### save model ###########################################


today = str(date.today())

model_name = 'cnn_model_' + today + ".keras"

model.save(model_name)


