# Human_Body_Segmentation
A Deep Learning project focuses on Semantic Segmentation of Human Body 

This projects helps predicting segmentation masks of Human Body and hence changing background. I tried 2 transfer learning models for training: Unet with MobileNetV2 as a backbone and DeepLabV3p with MobileNetV2 as a backbone. Performance for both of the models on validation dataset trained for 45 epochs is given below:

| Model  | precision | recall | f1-score | iou |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Unet  | **0.9195**  | 0.8912  | 0.9044  | 0.8267  |
| DeepLabV3p  | 0.9069 | **0.9131**  | **0.9095**  | **0.8348** |
