A trained YOLO model can predict specific traffic lights and traffic signs on images.

The following steps have been made to make a model: 
1.) Over 500 images have been taken for the training. 
2.) Annotating the images 
3.) Additionally image albumentations have been used in order to make the model more robust. 

This model can differentiate between 8 classes, two traffic light states and 6 traffic signs:

- Red
- Green
- PITIN
- PITOUT
- CrossParkOnly
- ParallelParkOnly
- OvertakingPermitted
- OvertakingProhibited


Here are some examples: 
![Screenshot 2024-12-25 152633](https://github.com/user-attachments/assets/d38443ee-682b-4681-9bdf-c4db228277a2)


![Screenshot 2024-12-25 152735](https://github.com/user-attachments/assets/e3020ab0-852b-4655-bda1-7e6150c84066)


![Screenshot 2024-12-25 152909](https://github.com/user-attachments/assets/37097592-795f-4599-9c88-c147ff156ea5)


![Screenshot 2024-12-25 153120](https://github.com/user-attachments/assets/0d7adc50-e9f8-448f-b43a-8c958df34ae0)


