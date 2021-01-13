# Per compiere inferenza sui vari TF-Hub module di base
import tensorflow as tf
import tensorflow_hub as hub

# Per il download delle immagini dalle varie fonti possibili (cvs, json, http)
import csv
import json
import requests
import os

# Per ritagliare e mostrare le immagini
from PIL import Image
import cv2
import matplotlib.pyplot as plt



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




# Stampiamo la versione di TF (deve essere maggiore di 2.0.0)
print("Tensorflow "+ tf.__version__)




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




#Variabili d'ambiente
csv_path = '/Users/cesco/Downloads/original_images_subset.csv'

json_path = '/Users/cesco/Downloads/images_subset.json'

module_handle_detector = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" 


class_list = ["Clothing", "Cowboy hat", "Sombrero", "Sun hat", "Scarf",
              "Skirt", "Miniskirt", "Jacket", "Fashion accessory", "Glove", 
              "Baseball glove", "Belt", "Sunglasses", "Tiara", "Necklace",
              "Sock", "Earrings", "Tie", "Goggles", "Hat", "Fedora", "Handbag",
              "Watch", "Umbrella", "Glasses", "Crown", "Swim cap", "Trousers",
              "Jeans", "Dress", "Swimwear", "Brassiere", "Shirt", "Coat", "Suit"
              "Footwear", "Roller skates", "Boot", "High heels", "Sandal",
              "Sports uniform", "Luggage & bags", "Backpack", "Suitcase",
              "Briefcase", "Helmet", "Bicycle helmet", "Football helmet"]




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




#Inizializzazione delle strutture dati
original_images_dict = {}

cropped_images_dict = {}




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------



#Funzioni di utilità

def csv_to_json(csv_path, json_path):
    json_array = []
      
    with open(csv_path, encoding='utf-8') as csvf: 
        csv_reader = csv.DictReader(csvf) 

        for row in csv_reader: 
            json_array.append(row)
  
    with open(json_path, 'w', encoding='utf-8') as jsonf: 
        json_string = json.dumps(json_array, indent=4)
        jsonf.write(json_string)
          



#-----------------------------------------------------------------------------




def save_initial_images(json_path):
    with open(json_path) as json_file: 
        original_images_dict = json.load(json_file)
        
    for i in range(len(original_images_dict)):
        id = original_images_dict[i]['id']
        url = original_images_dict[i]['path']
        page = requests.get(url)
        file_name = './original_images/{}'.format(id)
        
        original_images_dict[i]['path'] = file_name

        with open(file_name, 'wb') as f:
            f.write(page.content)
            
            
    print("--------------------------") 
    print("+ Original images loaded +")
    print("--------------------------") 
            
    return original_images_dict
 



#-----------------------------------------------------------------------------
    



def load_model(module_handle):
    detector = hub.load(module_handle).signatures['default']
    print("---------------------") 
    print("+ FasterRCNN loaded +")
    print("---------------------") 
    return detector




#-----------------------------------------------------------------------------




def run_detector(detector, original_images_dict):
    for i in range(len(original_images_dict)):
        path = original_images_dict[i]['path']
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)

        converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        result = detector(converted_img)

        result = {key:value.numpy() for key,value in result.items()}
        crop_objects(img.numpy(), result, path)
        
    print("--------------------------------") 
    print("+ Object Detection - Completed +")
    print("--------------------------------") 
  
    
    
    
#-----------------------------------------------------------------------------    
  
  
  

def crop_objects(img, result, path, max_boxes=3, min_score=0.6):
    image = Image.fromarray(img)
    width, height = image.size
    for i in range(min(result['detection_boxes'].shape[0], max_boxes)):
        if (result['detection_scores'][i] >= min_score):
            detected_class = "{}".format(result["detection_class_entities"][i].decode("ascii"))
            if(detected_class in class_list):
                ymin = int(result['detection_boxes'][i,0]*height)
                xmin = int(result['detection_boxes'][i,1]*width)
                ymax = int(result['detection_boxes'][i,2]*height)
                xmax = int(result['detection_boxes'][i,3]*width)
                crop_img = img[ymin:ymax, xmin:xmax].copy()
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                detected_class = detected_class.replace(" ", "-")
                outfile_name = os.path.basename(path).split('.')[0] + "-" + str(i) + "_" + detected_class
                cv2.imwrite("/Users/cesco/Desktop/cropped/{}.jpg".format(outfile_name), crop_img)
                cropped_images_dict[outfile_name] = detected_class



#-----------------------------------------------------------------------------




def print_original_images(original_images_dict):  
    i = 0
    j = 1
    rows = 1
    columns = 5
    figure=plt.figure(figsize=(30, 20), tight_layout=True)
    
    for i in range(len(original_images_dict)):
        path = original_images_dict[i]['path']
        id = os.path.basename(original_images_dict[i]['id']).split('.')[0]
        image = cv2.imread(path)
        image = cv2.resize(image, (1800, 2400))
        blue, green, red = cv2.split(image)
        frame_rgb = cv2.merge((red, green, blue))
        figure.add_subplot(rows, columns, j)
        plt.title("\n\nID: " + id + "\n\n",  fontsize=22)
        plt.imshow(frame_rgb)

        i = i + 1
        j = j + 1
        if(j > (rows*columns)):
            break
    plt.show()  
         
    
    
#----------------------------------------------------------------------------- 




def print_cropped_images(cropped_images_dict):  
    i = 0
    j = 1
    rows = 1
    columns = 5
    figure=plt.figure(figsize=(30, 20), tight_layout=True)
    
    
    for id in cropped_images_dict:
        path = "/Users/cesco/Desktop/cropped/{}.jpg".format(id)
        image = cv2.imread(path)
        image = cv2.resize(image, (1800, 2400))
        blue, green, red = cv2.split(image)
        frame_rgb = cv2.merge((red, green, blue))
        figure.add_subplot(rows, columns, j)
        plt.title("\n\nID: " + id + "\n\n",  fontsize=22)
        plt.imshow(frame_rgb)

        i = i + 1
        j = j + 1
        if(j > (rows*columns)):
            break
    plt.show()  
        



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




#1) Si parte da un CSV e lo si converte in JSON
csv_to_json(csv_path, json_path)


#2) A partire dal JSON facciamo il download delle immagini
original_images_dict = save_initial_images(json_path)


#3) Stampiamo qualche immagine di prova
print_original_images(original_images_dict)


#4) Carichiamo il modello
detector = load_model(module_handle_detector)


#5) Facciamo crop+label con il modello caricato, sulle immagini salvate
run_detector(detector, original_images_dict)


#6) Stampiamo qualche immagine cropped and labeled per vedere il risultato
print_cropped_images(cropped_images_dict)




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------