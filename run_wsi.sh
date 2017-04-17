#!/bin/bash

DEVICE=cuda0
MODE=FAST_RUN
FLAGS=mode=$MODE,device=$DEVICE,floatX=float32,on_unused_input='ignore' #,optimizer=None,exception_verbosity=high
MODEL_PATH=./models/foo.pkl 
SUPP_PATH=./data/supp.pkl
DAT_PATH=./data/dat.pkl
WSI_PATH=$HOME/data/wsi/contexts/xml-format  # path to directory that contains the data files in xml format
WSI_PROG=wsi.py

export CUDA_LAUNCH_BLOCKING=1
THEANO_FLAGS=$FLAGS python $WSI_PROG $MODEL_PATH $WSI_PATH "$@"
