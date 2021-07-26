# Human_Body_Segmentation

### A Deep Learning project focuses on Semantic Segmentation of Human Body 

| Original | Unet_pred | Overlay |
| :-------------------------: | :-------------------------: | :-------------------------: |
![Unet_1](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/images/Unet/SJ.jpg)  | ![Unet_2](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/images/Unet/pred_SJ.jpg) | ![Unet_3](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/images/Unet/overlay_SJ.jpg) |

| Original | DeepLabV3p_pred | Overlay |
| :-------------------------: | :-------------------------: | :-------------------------: |
![DeepLabV3p_1](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/images/DeeplabV3p/SJ.jpg)  | ![DeepLabV3p_2](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/images/DeeplabV3p/pred_SJ.jpg) | ![DeepLabV3p_3](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/images/DeeplabV3p/overlay_SJ.jpg) |

![Webcam_Unet](https://github.com/tshr-d-dragon/Virtual_Paint/blob/main/Virtual_Paint.gif)

![Webcam_DeepLabV3p](https://github.com/tshr-d-dragon/Virtual_Paint/blob/main/Virtual_Paint.gif)

This projects helps predicting segmentation masks of Human Body and hence changing background. I tried 2 transfer learning models for training: Unet with MobileNetV2 as a backbone and DeepLabV3p with MobileNetV2 as a backbone. Performance for both of the models on validation dataset trained for 45 epochs is given below:

| Model  | precision | recall | f1-score | iou |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| Unet  | **0.9195**  | 0.8912  | 0.9044  | 0.8267  |
| DeepLabV3p  | 0.9069 | **0.9131**  | **0.9095**  | **0.8348** |

All various modes used for changing the background are mentioned below:
- 0: Picture in BG
- 1: Video in BG
- 2: Blurred Picture in BG
- 3: Blurred Video in BG
- 4: B/W Picture in BG
- 5: B/W Vide in BG

## Project Structure
1. config.ini is the configuration file used to specify the parameters such as model_selection, prediction_type, input_file_path, BG_mode, and save_path.
2. predict.py file contains code for prediction. 
3. utils.py file contains all helper functions for changing the background.
4. smooth_borders.py contains code for the smoothening of boundaries of the predicted segmentation mask.
5. Underwater.mp4 and bg.jpg are used for background.
6. train folder training jupyter notebook for Unet and DeepLabV3p.
7. Models folder contains .py files for Unet and DeepLabV3p along with their weights (.h5 file).
8. predictions folder contains prediction of both models on random online images as well as videos taken live from Webcam.
9. requirement.txt file contains all the required dependencies.

## To run the prject, follow below steps
1. Ensure that you are in the project home directory
2. Create anaconda environment
3. Activate environment
4. >pip install -r requirement.txt
5. set the parameters in the config.ini file
6. >python init.py

## Please feel free to connect for any suggestions or doubts!!!
## Credits
1. The credits for dataset used for training goes to https://www.kaggle.com/noulam/tomato
2. I have modified https://github.com/Pawandeep-prog/resnet-flask-webapp/tree/main/templates html templates for flask
3. The credit for image used in html file for background goes to:
 
https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Tomato_Garden_%28Unsplash%29.jpg/800px-Tomato_Garden_%28Unsplash%29.jpg
  
https://www.thespruce.com/thmb/47xukLrGeP6r8jbmyeFFujXn4ug=/960x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/top-tomato-growing-tips-1402587-11-c6d6161716fd448fbca41715bbffb1d9.jpg

##### For better predictions, we need better image quality dataset for training and train it for more epochs with different backbones.
