import numpy as np
import cv2
import mediapipe as mp



mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)


# Mediapipe Virtual glass model

sunglasses = cv2.imread('D:/MY DOC/career/Chashmyar/Virtual glass/Glasses/mercurial_Reflective_cut.jpg')
cap = cv2.VideoCapture(0)
while 1:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    image_width, image_height = image.shape[1], image.shape[0]
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = face_detection.process(image)
    if results.detections:
        detections = results.detections
        for detection in detections:   
            bbox = detection.location_data.relative_bounding_box
            x,y = int(bbox.xmin * image_width),int(bbox.ymin * image_height)
            w,h = int(bbox.width * image_width),int(bbox.height * image_height)
            faces = x,y,w,h

            #Right Eye 
            righteye = detection.location_data.relative_keypoints[0] 
            RE_x,RE_y = int(righteye.x * image_width),int(righteye.y * image_height) 
            
            #Left Eye
            lefteye = detection.location_data.relative_keypoints[1] 
            LE_x,LE_y = int(lefteye.x * image_width),int(lefteye.y * image_height) 
            
            
            #nose
            nose = detection.location_data.relative_keypoints[2] 
            nose_x, nose_y = int(nose.x * image_width),int(nose.y * image_height) 
            
            width = int(2.8*(LE_x-(RE_x+8))) 
            
            height = int(h/1.7)
            Eyes_coord = LE_x,LE_y,width 
                                    
            img_roi = image[ y:y+height,x:x+width] 
            
            
            sunglasses_small = cv2.resize(sunglasses, (width,height),  interpolation=cv2.INTER_AREA) 
            
            gray_sunglasses = cv2.cvtColor(sunglasses_small, cv2.COLOR_BGR2GRAY) 
        
            ret, mask = cv2.threshold(gray_sunglasses, 230, 255,  cv2.THRESH_BINARY_INV) 
            
            mask_inv = cv2.bitwise_not(mask)
            
            masked_face = cv2.bitwise_and(sunglasses_small, sunglasses_small, mask=mask)
            
            masked_frame = cv2.bitwise_and(img_roi,  img_roi, mask=mask_inv)
            image[y:y+height,x:x+width] = cv2.add(masked_face,  masked_frame)
            
        cv2.imshow('image', image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
cap.release()            
