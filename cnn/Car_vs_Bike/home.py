import streamlit as st
#from streamlit_image_select import image_select

import altair as alt

import os 

import random
from keras.models import load_model
import numpy as np
import cv2 
from PIL import Image
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from streamlit_extras.image_selector import image_selector 
from streamlit_extras.image_selector import show_selection

import plotly.express as px

from streamlit_cropper import st_cropper



def load_images(images_path,vehicle_type):
    
    
    path = 'Car-Bike-Dataset/'+str(vehicle_type)
    
    vehicle_images = os.listdir(path)
    
    
    for i in range(len(vehicle_images)):
    
        images_path.append(os.path.join(path,vehicle_images[i]))    
        
    
    
    images_path = list(map(lambda x : x.replace('\\','/'),images_path))
        

    
    
    
    return images_path
    





st.set_page_config(page_title = "CNN_Car_vs_Bike",
                   page_icon="📈")

st.title("Convolutional Neronal Network for recognizing 'Car' and " +
         "Bike' Images ")





# images_path = []



# images_path = load_images(images_path,'Car')

# images_path = load_images(images_path,'Bike')


st.header("\n\n")

st.header("Upload an image of a car or a bike")
img_file_buffer = st.file_uploader("upload image", type= ['png', 'jpg','jpeg'], accept_multiple_files=False)




if img_file_buffer is not None: 

    # open image 
    image = Image.open(img_file_buffer)
         
    st.header("\n\n")
    st.header("Select the car or the bike in the image")
    
    # crop image 
    img_cropped = st_cropper(image,
                             realtime_update=True, 
                             box_color="#FFFFFF",
                             aspect_ratio=None,
                             )
    
    
    st.write("\n")
    st.header("Selected image")
   
    # show cropped image
    st.image(img_cropped)
 

    st.write("\n")
    st.header("Click on the 'Start Prediction' button for a prediction on the selected image")

  
  
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        pass
    
    with col2:
        button1 = st.button("Start Prediction",icon = "🤖")
        
    with col3:
        pass
    




    if button1: 
        
        
        # resize image so it has the same dimensions as the 
        # trained cnn model
        resize_width = 100
        resize_height = 100
        
        
        img_cropped_array = np.array(img_cropped)
    
        
        resized_img = cv2.resize(img_cropped_array,(resize_width,resize_height))
        
        
        # cnn was trained with values between 0 and 1
        # we need to divdide by 255
        resized_img = resized_img/255
    
        # some images have 4 instead of 3 channels 
        # if image has 4 channels take the first 3
        if np.shape(resized_img)[2] == 4:
            
            resized_img = resized_img[:,:,0:3]
            
            
            
        # load model     
        model = load_model("cnn_model_2025-01-19.keras")
    
        
        # add axis so we can predict
        resized_img = resized_img[np.newaxis,:,:,:]
    
        # predict 
        prediction = model.predict(resized_img)
    
        # het the class number of the predicted class 
        predicted_class = np.argmax(prediction)
       
    
    
    
        # make a list where probabilities of all the classes are stored  
        prediction_list = []
    
        for i in range(0,np.shape(prediction)[1]):
            
            prediction_list.append(prediction[0,i]*100)
                    
        
        
        # list of all classes 
        class_list =  ["Car","Bike"]
        
        
        st.write("The trained model thinks that the image is a: " + class_list[predicted_class])
    
    
    
    
    
        ############## plot ################
    
        # make DataFrame with the class list and the probabilities of each class 
        df = pd.DataFrame(data = {"classes" :class_list,
                                  "probability %" : prediction_list})    
        
    
       
        # mak ebafr plot 
        # fig,ax = plt.subplots(figsize=(10, 4))
        
        
        
        
        # ax = sns.barplot(data = df, 
        #                        x = "classes" ,
        #                        y = "probability %",
        #                        hue = "classes",
        #                        palette = ["#222fbf","#bf2d22"],
        #                        saturation=1,
        #                        legend  = True,
        #                        width=0.5,
        #                        )
    
        # ax.set_ylim([0,100])
    
         
        # # write probabilities above each bar 
        # for i in range(0,len(class_list)):
        #     ax.bar_label(ax.containers[i], fontsize=10, fmt = '%.2f')
        
       
        fig = px.bar(df, x='classes', y='probability %', title='Probabilities',color = "classes",
                     color_discrete_sequence = ["#222fbf","#bf2d22"],range_y=[0,100],text_auto=True)
       
        
        # Update the bar width
        fig.update_traces(width=0.5)
    
        # Display the plot in Streamlit
        st.plotly_chart(fig)




