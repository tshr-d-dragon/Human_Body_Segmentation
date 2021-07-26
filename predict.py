
"""
Modes:
0: Picture in BG
1: Video in BG
2. Blurred Picture in BG
3: Blurred Video in BG
4: B/W Picture in BG
5: B/W Vide in BG
"""



""" Imports """

import numpy as np
import cv2
from DeepLabV3p import Deeplabv3
from UNET_MobileNetV2 import Unet_MobileNetV2
from configparser import ConfigParser
from utils import Substract_BG
from smooth_borders import fix_segmentation_maps


""" Read config File """
cp = ConfigParser()
cp.read(r'Human_Seg/config.ini')



""" Model Selection and Prediction Functions """

def Model(model_selection = 'DeeplabV3P'):
    
    """ Model """
    
    # model_selection = 'DeeplabV3P'      ## ['DeeplabV3P', 'Unet_MobileNetV2']
    
    if model_selection == 'DeeplabV3P':
        model = Deeplabv3(weights='cityscapes', input_tensor=None, 
                          input_shape=(512,512,3), classes=1, 
                          backbone= 'mobilenetv2', #'xception'
                          OS=16, alpha=1., activation='sigmoid')
        model.load_weights('Human_Seg/Deeplabv3p_MobilenetV2_Human_Seg_030+15.h5')
    
    elif model_selection == 'Unet_MobileNetV2':
        model = Unet_MobileNetV2((512, 512, 3), classes = 1)    
        model.load_weights('Human_Seg/Unet_MobilenetV2_Human_Seg_045.h5')
    
    else:
        print('Invalid Model')
    
    return model
    

def Prediction(model, prediction_type = 'video', mode = 0, path = 0, save_path = r'Human_Seg\Result.mp4'):
    
    """ Prediction """
    
    if prediction_type == 'image':
        """ Prediction on an image """
        
        # alpha = 0.6
        img_path = path
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512))
        frame = img.copy()
        img = img/255.0
        img = img.astype(np.float32)
        img = img.reshape(1, 512, 512, 3)
        
        pred = model.predict(img)
        pred = pred.reshape(512, 512)
        pred[pred > 0.1] = 1
        pred[pred < 0.1] = 0
        pred = pred * 255
        pred = pred.astype(np.uint8)
        # cv2.imwrite(save_path,pred)
        
        if cv2.waitKey(1) & 0xFF == ord('1'): 
            mode = 1
        elif cv2.waitKey(1) & 0xFF == ord('2'): 
            mode = 2
        elif cv2.waitKey(1) & 0xFF == ord('3'): 
            mode = 3
        elif cv2.waitKey(1) & 0xFF == ord('4'): 
            mode = 4
        elif cv2.waitKey(1) & 0xFF == ord('5'): 
            mode = 5
        elif cv2.waitKey(1) & 0xFF == ord('0'): 
            mode = 0
        pred = fix_segmentation_maps(pred)
        mask = Substract_BG(frame, pred, mode)
        cv2.imwrite(save_path, mask)
    
        # pred = np.dstack([pred, pred, pred])
        # overlay = cv2.addWeighted(frame, alpha, pred, 1-alpha, 0)
        # cv2.imwrite('Human_Seg/75805310_overlay.jpg',overlay)
        
        cv2.imshow('overlay', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif prediction_type == 'video':
        """ Prediction on a Video """
        
        # t = 0
        video_path = path
        cap = cv2.VideoCapture(video_path)
        cap.set(3, 1280)  # width
        cap.set(4, 720)   # height
        # vid_fps = cap.get(5)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (512, 512))
        
        # alpha = 0.6
        ret = True
        
        while ret :
            # frameId = int(round(cap.get(1)))    
            ret, img = cap.read() 
            
            img = cv2.resize(img, (512, 512))
            frame = img.copy() 
            img = img/255.0
            img = img.astype(np.float32)
            img = img.reshape(1, 512, 512, 3)
            
            pred = model.predict(img)
            pred = pred.reshape(512, 512)
            pred[pred > 0.1] = 1
            pred[pred < 0.1] = 0
            pred = pred * 255
            pred = pred.astype(np.uint8)
            
            if cv2.waitKey(1) & 0xFF == ord('1'): 
                mode = 1
            elif cv2.waitKey(1) & 0xFF == ord('2'): 
                mode = 2
            elif cv2.waitKey(1) & 0xFF == ord('3'): 
                mode = 3
            elif cv2.waitKey(1) & 0xFF == ord('4'): 
                mode = 4
            elif cv2.waitKey(1) & 0xFF == ord('5'): 
                mode = 5
            elif cv2.waitKey(1) & 0xFF == ord('0'): 
                mode = 0
            pred = fix_segmentation_maps(pred)
            mask =  Substract_BG(frame, pred, mode)
            
            # pred = np.dstack([pred, pred, pred])
            # print(frame.shape, pred.shape, frame.dtype, pred.dtype)
            # overlay = cv2.addWeighted(frame, alpha, pred, 1-alpha, 0)
            
            # s = time.time()
            # fps = int(1/(s-t))
            # t = s
            # cv2.putText(overlay, 'CPU_FPS: '+str(fps), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
            out.write(mask)
            
            cv2.imshow('Frame', mask)
            # print(frameId)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
    else:
        print('Invalid prediction_type')

    return None



""" Main Function """
def main():
    
    model_selection = eval(cp.get('model_selection', 'model_selection'))
    prediction_type = eval(cp.get('prediction_type', 'prediction_type'))
    BG_mode = eval(cp.get('BG_mode', 'BG_mode'))
    path = eval(cp.get('input_file_path', 'path'))
    save_path = eval(cp.get('save_path', 'save_path'))
    
    Prediction(Model(model_selection = model_selection), prediction_type = prediction_type, 
                     mode = BG_mode, path = path, save_path = save_path)

    return None