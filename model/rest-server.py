#!flask/bin/python
################################################################################################################################
#------------------------------------------------------------------------------------------------------------------------------                                                                                                                             
# This file implements the REST layer. It uses flask micro framework for server implementation. Calls from front end reaches 
# here as json and being branched out to each projects. Basic level of validation is also being done in this file. #                                                                                                                                  	       
#-------------------------------------------------------------------------------------------------------------------------------                                                                                                                              
################################################################################################################################
from flask import Flask, jsonify, abort, request, make_response, url_for,redirect, render_template
#from flask.ext.httpauth import HTTPBasicAuth
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
import os
import sys
import random
from tensorflow.python.platform import gfile
from six import iteritems
sys.path.append('..')
import numpy as np
#from lib.src import retrieve
import retrieve
#from lib.src.align import detect_face
import detect_face
import tensorflow as tf
import pickle
from tensorflow.python.platform import gfile
app = Flask(__name__, static_url_path = "")

auth = HTTPBasicAuth()


#==============================================================================================================================
#                                                                                                                              
#    Loading the stored face embedding vectors for image retrieval                                                                 
#                                                                          						        
#                                                                                                                              
#==============================================================================================================================
#with open('../lib/src/face_embeddings.pickle','rb') as f:
#					    	feature_array = pickle.load(f) 

with open("C:\\Users\\AJain7\\Desktop\\arpit's_model\\model\\extracted_dict_2.pickle",'rb') as f:
					    	feature_array = pickle.load(f) 

model_exp = "C:\\Users\\AJain7\\Desktop\\arpit's_model\\model\\20180402-114759"
graph_fr = tf.Graph()
sess_fr = tf.Session(graph=graph_fr)
with graph_fr.as_default():
	saverf = tf.train.import_meta_graph(os.path.join(model_exp, 'model-20180402-114759.meta'))
	saverf.restore(sess_fr, os.path.join(model_exp, 'model-20180402-114759.ckpt-275'))
	pnet, rnet, onet = detect_face.create_mtcnn(sess_fr, None)
#==============================================================================================================================
#                                                                                                                              
#  This function is used to do the face recognition from video camera                                                          
#                                                                                                 
#                                                                                                                              
#==============================================================================================================================
@app.route('/facerecognitionLive', methods=['GET', 'POST'])
def face_det():
    
    retrieve.recognize_face(sess_fr,pnet, rnet, onet,feature_array)

#==============================================================================================================================
#                                                                                                                              
#                                           Main function                                                        	            #						     									       
#  				                                                                                                
#==============================================================================================================================
@app.route("/")
def main():
    
    return render_template("main.html")
    #return '''<form action="/check" method="get">
  #URL: <input type="text" name="fname"><br>
  #<input type="submit" value="Submit">
#</form>'''

if __name__ == '__main__':
    app.run(debug = True, host= '127.0.0.1')