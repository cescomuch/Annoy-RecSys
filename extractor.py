# Per compiere inferenza sui vari TF-Hub module di base
import tensorflow as tf
import tensorflow_hub as hub

# Per operazioni di trasformazione di immagini/pixels
import numpy as np

# Per effettuare letture/scritture da file
import os.path
import glob



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




#Variabili d'ambiente
module_handle_extractor = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"




#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------




def load_img(path):
  img = tf.io.read_file(path)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize_with_pad(img, 224, 224)
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

  return img




#-----------------------------------------------------------------------------
  



def get_feature_vectors(module_handle):

  i = 0

  module = hub.load(module_handle)
  print("-----------------------") 
  print("+ MobileNet_v2 loaded +")
  print("-----------------------") 


  for filename in glob.glob('./cropped_and_labeled_images/*.jpg'):
    img = load_img(filename)
    features = module(img)   
    feature_set = np.squeeze(features)  
    outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
    out_path = os.path.join('./features_vectors', outfile_name)
    np.savetxt(out_path, feature_set, delimiter=',')    
    i = i + 1
    

  print("------------------------------------------") 
  print("+ Generating Feature Vectors - Completed +")
  print("------------------------------------------")   
    



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------





#1) Otteniamo i features vectors a partire dalla immagini croppate ed etichettate
get_feature_vectors(module_handle_extractor)





#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------