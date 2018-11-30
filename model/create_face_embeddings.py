from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import numpy as np
import pickle

with tf.Graph().as_default():
      
        with tf.Session() as sess:

            # Get the paths for the corresponding images

            '''
            vinayak =  ['datasets/kar_Vin_aligned/vinayak/' + f for f in os.listdir('datasets/kar_Vin_aligned/vinayak')]
            karthik =  ['datasets/kar_Vin_aligned/karthik/' + f for f in os.listdir('datasets/kar_Vin_aligned/karthik')]
            ashish = ['datasets/kar_Vin_aligned/Ashish/' + f for f in os.listdir('datasets/kar_Vin_aligned/Ashish')]
            saurabh = ['datasets/kar_Vin_aligned/Saurabh/' + f for f in os.listdir('datasets/kar_Vin_aligned/Saurabh')]
            hari = ['datasets/kar_Vin_aligned/Hari/' + f for f in os.listdir('datasets/kar_Vin_aligned/Hari')]
            paths = vinayak+karthik+ashish+saurabh+hari
            '''
            
            arpitjain = [r"C:\Users\AJain7\Desktop\arpit's_model\data_out\arpitjain" +'\\'+ f for f in os.listdir(r"C:\Users\AJain7\Desktop\arpit's_model\data_out\arpitjain")]
            prateekmohan = [r"C:\Users\AJain7\Desktop\arpit's_model\data_out\prateekmohan" +'\\'+ f for f in os.listdir(r"C:\Users\AJain7\Desktop\arpit's_model\data_out\prateekmohan")]
            prachimane = [r"C:\Users\AJain7\Desktop\arpit's_model\data_out\prachimane" +'\\'+ f for f in os.listdir(r"C:\Users\AJain7\Desktop\arpit's_model\data_out\prachimane")]
            prashantsethia = [r"C:\Users\AJain7\Desktop\arpit's_model\data_out\prashantsethia" +'\\'+ f for f in os.listdir(r"C:\Users\AJain7\Desktop\arpit's_model\data_out\prashantsethia")]
            rajivsoni = [r"C:\Users\AJain7\Desktop\arpit's_model\data_out\rajivsoni" +'\\'+ f for f in os.listdir(r"C:\Users\AJain7\Desktop\arpit's_model\data_out\rajivsoni")]
            ravindrasingh = [r"C:\Users\AJain7\Desktop\arpit's_model\data_out\ravindrasingh" +'\\'+ f for f in os.listdir(r"C:\Users\AJain7\Desktop\arpit's_model\data_out\ravindrasingh")]
            smitanegi = [r"C:\Users\AJain7\Desktop\arpit's_model\data_out\smitanegi" +'\\'+ f for f in os.listdir(r"C:\Users\AJain7\Desktop\arpit's_model\data_out\smitanegi")]
            varunrajshanmugavelayudham = [r"C:\Users\AJain7\Desktop\arpit's_model\data_out\varunrajshanmugavelayudham" +'\\'+ f for f in os.listdir(r"C:\Users\AJain7\Desktop\arpit's_model\data_out\varunrajshanmugavelayudham")]

            paths = arpitjain + prateekmohan + prashantsethia + prachimane + rajivsoni + ravindrasingh + smitanegi + varunrajshanmugavelayudham
            
            #np.save("images.npy",paths)
            # Load the model
            facenet.load_model(r"C:\Users\AJain7\Desktop\arpit's_model\model\20180402-114759")
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            image_size = 160
            embedding_size = embeddings.get_shape()[1]
            extracted_dict = {}
            
            # Run forward pass to calculate embeddings
            for i, filename in enumerate(paths):

                images = facenet.load_image(filename, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                extracted_dict[filename] =  feature_vector
                if(i%100 == 0):
                    print("completed",i," images")
            
            with open('extracted_dict_2.pickle','wb') as f:
                pickle.dump(extracted_dict,f)