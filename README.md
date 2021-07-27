# Human_Body_Segmentation

### A Deep Learning project focuses on Semantic Segmentation of Human Body 

| Original | Unet_pred | Overlay |
| :-------------------------: | :-------------------------: | :-------------------------: |
| ![Unet_1](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/images/Unet/75805310.jpg)  | ![Unet_2](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/images/Unet/pred_75805310.jpg) | ![Unet_3](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/images/Unet/overlay_75805310.jpg) |

| Original | DeepLabV3p_pred | Overlay |
| :-------------------------: | :-------------------------: | :-------------------------: |
| ![DeepLabV3p_1](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/images/DeeplabV3p/75805310.jpg)  | ![DeepLabV3p_2](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/images/DeeplabV3p/pred_75805310.jpg) | ![DeepLabV3p_3](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/images/DeeplabV3p/overlay_75805310.jpg) |

### Predictions on live webcam using Unet:
![Webcam_Unet](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/webcam/Final_Unet.gif)

### Predictions on live webcam using DeepLabV3p:
![Webcam_DeepLabV3p](https://github.com/tshr-d-dragon/Human_Body_Segmentation/blob/main/predictions/webcam/Final_Deeplab.gif)

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
- 5: B/W Video in BG

## Project Structure
1. config.ini is the configuration file used to specify the parameters such as model_selection, prediction_type, input_file_path, BG_mode, and save_path.
2. predict.py file contains code for prediction. 
3. utils.py file contains all helper functions for changing the background.
4. Underwater.mp4 and bg.jpg are used for background.
5. train folder training jupyter notebook for Unet and DeepLabV3p.
6. Models folder contains .py files for Unet and DeepLabV3p along with their weights (.h5 file).
7. predictions folder contains prediction of both models on random online images as well as videos taken live from Webcam.
8. requirement.txt file contains all the required dependencies.

## To run the prject, follow below steps
1. Ensure that you are in the project home directory
2. Create anaconda environment
3. Activate environment
4. >pip install -r requirement.txt
5. set the parameters in the config.ini file
6. >python init.py

## Please feel free to connect for any suggestions or doubts!!!
## Credits
1. The credits for dataset used for training goes to https://www.kaggle.com/tapakah68/supervisely-filtered-segmentation-person-dataset
2. I have referred https://github.com/bonlime/keras-deeplab-v3-plus/ repository for DeepLabV3p model.
3. The credit for images and videos used for prediction and background goes to:
   -   https://thedigitalweekly.com/2021/07/06/black-widow-sequel-possible-says-director-cate-shortland/
   -   https://twitter.com/MarvelFansIT/status/1378680224922136576/photo/1
   -   https://twitter.com/carles_madness/status/1359549136480178182/photo/1
   -   Image by <a href="https://pixabay.com/users/pexels-2286921/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2179183">Pexels</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=2179183">Pixabay</a>
   -   Video by <a href="https://pixabay.com/users/waiguobox-2405726/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=5271">Timofey Iasinskii</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=5271">Pixabay</a>

##### For better predictions, we need better image quality dataset for training and train it for more epochs with different backbones.
