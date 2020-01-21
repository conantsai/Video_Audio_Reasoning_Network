# TensorFlow version of NIPS2016 soundnet

from soundnet.SBIR_soundnetutil import load_from_txt
from soundnet.SBIR_soundnetmodel import Model
import tensorflow as tf
import numpy as np
import argparse
import sys
import os

# Make xrange compatible in both Python 2, 3
try:
    xrange
except NameError:
    xrange = range

local_config = {  
            'batch_size': 1, 
            'eps': 1e-5,
            'sample_rate': 22050,
            'load_size': 22050*20,
            'name_scope': 'SoundNet',
            'phase': 'extract',
            }


# audio_txt = "B-1/Jinag_thesis/backup_thesis/test/sound_testing.csv"
# outpath = "B-1/Jinag_thesis/backup_thesis/test/v1000"
phase = "extract"
layer_min = 24
layer_max = 25
is_save = True
   
def extract_feat(model, sound_input, outpath):
    # Extract feature
    features = {}
    feed_dict = {model.sound_input_placeholder: sound_input}

    for idx in xrange(layer_min, layer_max):
        feature = model.sess.run(model.layers[idx], feed_dict=feed_dict)
        features[idx] = feature
        if is_save:
            np.save(os.path.join(outpath, 'tf_fea{}.npy'.format( \
                str(idx).zfill(2))), np.squeeze(feature))
            print("Save layer {} with shape {} as {}/tf_fea{}.npy".format( \
                    idx, np.squeeze(feature).shape, outpath, str(idx).zfill(2)))
    
    return features


def soundnet_main(audio_txt, outpath):

    # Load pre-trained model
    G_name = 'B-1/Jinag_thesis/backup_thesis/soundnet/models/sound8.npy'
    param_G = np.load(G_name, encoding = 'latin1').item()
        
    # Extract Feature
    sound_samples = load_from_txt(audio_txt, config=local_config)
    
    # Make path
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Init. Session
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement=True
    sess_config.gpu_options.allow_growth = True
    
    with tf.Session(config=sess_config) as session:
        # Build model
        model = Model(session, config=local_config, param_G=param_G)
        init = tf.global_variables_initializer()
        session.run(init)
        
        model.load()
    
        for sound_sample in sound_samples:
            output = extract_feat(model, sound_sample, outpath)
