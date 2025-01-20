import streamlit as st


st.header("About the dataset")

st.markdown(
    """
    
    - This model was made with the [Car vs Bike dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset)   
      
    - It has around 4000 images in total 
    
    - Dataset was split: 70 % Training and 30 % Test
    
    - A Convolutional Neural Network (CNN) was trained 
    
    - Keras Tensorflow library was used
    
    """)
    
    
    

for i in range(5):
    st.write("\n")



st.header("About the CNN")

st.write("Here is the structure of the CNN")    


st.code(
    
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
    
        # N class, N = 2 
        model.add(layers.Dense(2, activation='softmax'))
    
        model.summary()
    
        model.compile(optimizer='adam',
                      loss = 'CategoricalCrossentropy',
                      metrics=['accuracy'])
    
    
    

"""
,language = "python")


for i in range(5):
    st.write("\n")
    
st.write("Here is a graphical representation of the cnn ")
st.image("model_structure.png")



for i in range(5):
    st.write("\n")
    



st.header("Here are the results for the test dataset")


st.markdown("### Confusion Matrix")

st.markdown(
    """
        - The higher the values are on the main diagonal the more accurately the model can classify 
            
    """)
    
st.image("confusion_matrix.png")



for i in range(5):
    st.write("\n")


st.markdown("### f1 Scores")

st.markdown(
    """
        - The f1 score is a metric to evaluate how good the prediciton for each class is
        - It ranges from 0 to 1 score. The closer the score is to 1 the better the model can predict the class
            
    """)

st.image("f1_score.png")


