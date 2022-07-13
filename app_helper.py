import cv2
import numpy as np

import time
import sys
import os
import pandas as pd
from PIL import Image
from PIL import *
import PIL

def result_dataframe(shirt_neck_att,sleeve_att):
    x=shirt_neck_att
    attribute_dict={}
    if(x=='Shirt_Collared'):
        attribute_dict = {"shirt type" : "Shirt","neck type" : "Collared Neck"}
    elif(x=='Tshirt_Round'):
        attribute_dict = {"shirt type" : "T-Shirt","neck type" : "Round Neck"}
    elif(x=='Tshirt_Vneck'):
        attribute_dict = {"shirt type" : "T-Shirt","neck type" : "V Neck"}
    elif(x=='Tshirt_Collared'):
        attribute_dict = {"shirt type" : "T-Shirt","neck type" : "Collared Neck"}
    elif(x=='unknown'):
        attribute_dict = {"shirt type" : "--","neck type" : "--"}
    
    if (sleeve_att == "Half_Sleeve"):
        attribute_dict['Sleeve Type'] = 'Half Sleeve'
    elif(sleeve_att == "Full_Sleeve"):
        attribute_dict['Sleeve Type'] = 'Full Sleeve'
    elif(sleeve_att == "unknown"):
        attribute_dict['Sleeve Type'] = '--'
    
    data = [attribute_dict]
    attribute_df = pd.DataFrame.from_dict(data)
    attribute_df = attribute_df.T
        
    
    return attribute_dict

def predict_attributes(input_filename):
    input_path = input_filename

    base_name = os.path.basename(input_path)
    # print(base_name)
    CONFIDENCE = 0.01
    SCORE_THRESHOLD = 0.01
    IOU_THRESHOLD = 0.01

    # the neural network configuration
    config_path_shirt = "config/yolov3_custom_shirt.cfg"
    config_path_sleeve = "config/yolov3_custom_sleeve.cfg"
   

    # the YOLO net weights file
    weights_path_shirt = "config/yolov3_custom_8000.weights"
    weights_path_sleeve = "config/yolov3_custom_4000.weights"
 

    # loading all the class labels (objects)
    labels_sleeve = ['Half_Sleeve','Full_Sleeve']
    labels_shirt = ['Shirt_Collared','Tshirt_Round','Tshirt_Vneck','Tshirt_Collared']
#     print("labels_shirt = "+str(labels_shirt))
#     print("labels_sleeve = "+str(labels_sleeve))
    

    # load the YOLO network
    net_shirt = cv2.dnn.readNetFromDarknet(config_path_shirt, weights_path_shirt)
    net_sleeve = cv2.dnn.readNetFromDarknet(config_path_sleeve, weights_path_sleeve)


    # Temporary image path where the resized image of the input image is stored
    path_name ='static/uploads/'+base_name
    # input image path
    path_name1 = input_filename
    # input image
    im =Image.open(path_name1)
    # resizing the image maintaining the aspect ratio 
    resized_img =PIL.ImageOps.contain(im, (1000,1080),method =Image.Resampling.BICUBIC)
    # saving the resized image to the temporary image path
    resized_img.save(path_name)
    
    # reading the image from temporary image path   

    image = cv2.imread(path_name)

    file_name = os.path.basename(path_name)
    
    filename, ext = file_name.split(".")

    h, w = image.shape[:2]

    # create 4D blob*
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    # sets the blob as the input of the network
    net_shirt.setInput(blob)
    net_sleeve.setInput(blob)

    # get all the layer names
    ln_shirt = net_shirt.getLayerNames()
    ln_sleeve = net_sleeve.getLayerNames()
    try:
        ln_shirt = [ln_shirt[i[0] - 1] for i in net_shirt.getUnconnectedOutLayers()]
        ln_sleeve = [ln_sleeve[i[0] - 1] for i in net_sleeve.getUnconnectedOutLayers()]
    except IndexError:
        # in case getUnconnectedOutLayers() returns 1D array when CUDA isn't available
        ln_shirt = [ln_shirt[i - 1] for i in net_shirt.getUnconnectedOutLayers()]
        ln_sleeve = [ln_sleeve[i - 1] for i in net_sleeve.getUnconnectedOutLayers()]

    # measure how much it took in seconds
    start = time.perf_counter()

    # feed forward (inference) and get the network output
    layer_outputs_shirt = net_shirt.forward(ln_shirt)
    layer_outputs_sleeve = net_sleeve.forward(ln_sleeve)
    
    time_took = time.perf_counter() - start
    
     # measure how much it took in seconds

    print(f"Time took: {time_took:.2f}s")

    boxes_shirt, confidences_shirt, class_ids_shirt = [], [], []
    boxes_sleeve, confidences_sleeve, class_ids_sleeve = [], [], []

    # loop over each of the layer outputs
    for output in layer_outputs_shirt:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id_shirt = np.argmax(scores)
            confidence = scores[class_id_shirt]
            # discard weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes_shirt.append([x, y, int(width), int(height)])
                confidences_shirt.append(float(confidence))
                class_ids_shirt.append(class_id_shirt)

    # loop over each of the layer outputs
    for output in layer_outputs_sleeve:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id_sleeve = np.argmax(scores)
            confidence = scores[class_id_sleeve]
            # discard weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes_sleeve.append([x, y, int(width), int(height)])
                confidences_sleeve.append(float(confidence))
                class_ids_sleeve.append(class_id_sleeve)


    # perform the non maximum suppression given the scores defined before
    idxs_shirt = cv2.dnn.NMSBoxes(boxes_shirt, confidences_shirt, SCORE_THRESHOLD, IOU_THRESHOLD)
    idxs_sleeve = cv2.dnn.NMSBoxes(boxes_sleeve, confidences_sleeve, SCORE_THRESHOLD, IOU_THRESHOLD)

    font_scale = 1
    thickness = 2
    # print("idxs_shirt"+str(idxs_shirt))
    # print("idxs_sleeve"+str(idxs_sleeve))
    # ensure at least one detection exists
    try:

        if len(idxs_shirt) > 0:
            # loop over the indexes we are keeping
            for i in idxs_shirt.flatten():
                # extract the bounding box coordinates
                x, y = boxes_shirt[i][0], boxes_shirt[i][1]
                w, h = boxes_shirt[i][2], boxes_shirt[i][3]

                # draw a bounding box rectangle and label on the image

                # bounding box color and text color for shirt and neck attribute  (0,0,255) = RED

                color_shirt =(0,0,255)

                cv2.rectangle(image, (x, y), (x + w, y + h), color=color_shirt, thickness=thickness)
                #   text with label and confidence
        #   text = f"{labels_shirt[class_ids_shirt[i]]}: {confidences_shirt[i]:.2f}"
        #   text with label 
                text = f"{labels_shirt[class_ids_shirt[i]]}"

                # calculate text width & height to draw the transparent boxes as background of the text
                (text_width, text_height) = cv2.getTextSize(text,  cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color_shirt, thickness=cv2.FILLED)



                # now put the text (label: confidence %)

                cv2.putText(image, text, (x+100, y+h-35), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=color_shirt, thickness=thickness)

    except:
        print("shirt type and neck type not detected")
# ensure at least one detection exists
    try:

        if len(idxs_sleeve) > 0:
        # loop over the indexes we are keeping
            for i in idxs_sleeve.flatten():
                # extract the bounding box coordinates
                x, y = boxes_sleeve[i][0], boxes_sleeve[i][1]
                w, h = boxes_sleeve[i][2], boxes_sleeve[i][3]

                # draw a bounding box rectangle and label on the image
        #   bounding box color and text color for sleeve attribute  (255,0,0) = BLUE
                color_sleeve = (255,0,0)

                cv2.rectangle(image, (x, y), (x + w, y + h), color=color_sleeve, thickness=thickness)
        #   text with label and confidence
        #   text = f"{labels_sleeve[class_ids_sleeve[i]]}: {confidences_sleeve[i]:.2f}"
        #   text with label 
                text = f"{labels_sleeve[class_ids_sleeve[i]]}"

                # calculate text width & height to draw the transparent boxes as background of the text
                (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color_sleeve, thickness=cv2.FILLED)


        # prints the sleeve type in the top left corner of the bounding box (x+10, y +35)

                cv2.putText(image, text, (x+10, y +35), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=color_sleeve, thickness=thickness)
    except:
            print("sleeve type not detected")

    try:

        for i in idxs_shirt.flatten():
    #         print("\n predicted label : ",labels_shirt[class_ids_shirt[i]],confidences_shirt[i])
            shirt_neck_type , shirt_neck_confidence =labels_shirt[class_ids_shirt[i]],confidences_shirt[i]
    except:
        shirt_neck_type="unknown"
        print("shirt type and neck type not detected")
    try:
        for i in idxs_sleeve.flatten():
#         print("\n predicted label : ",labels_sleeve[class_ids_sleeve[i]],confidences_sleeve[i])
            sleeve_type , sleeve_type_confidence =labels_sleeve[class_ids_sleeve[i]],confidences_sleeve[i]
    except:
        sleeve_type , sleeve_type_confidence ="unknown","unknown"
        print("sleeve not detected")
    # dummy value
        sleeve_type , sleeve_type_confidence ="unknown","unknown"
    # cv2.imshow("image", image)

    # file name to be saved --------------------------------------------------------------------------------------------
    cv2.imwrite("static/predictions/"+base_name, image)
    # -------------------------------------------------------------------------------------------------------------------
    
    df = result_dataframe(shirt_neck_type,sleeve_type)

    return (df)

