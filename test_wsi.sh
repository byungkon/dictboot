#!/bin/bash

DEVICE=cuda0
MODE=FAST_RUN
FLAGS=mode=$MODE,device=$DEVICE,floatX=float32,on_unused_input='ignore' #,optimizer=None,exception_verbosity=high
MODEL_PATH=./models/foo.pkl 
SUPP_PATH=./data/supp.pkl
DAT_PATH=./data/dat.pkl
WSI_PATH=$HOME/data/wsi/contexts/xml-format  # path to directory that contains the data files
WSI_PROG=wsi.py
SIM_PROG=wordsim.py 
SICK_PROG=eval_sick.py

export CUDA_LAUNCH_BLOCKING=1
THEANO_FLAGS=$FLAGS python $WSI_PROG $MODEL_PATH $WSI_PATH "$@"
#THEANO_FLAGS=$FLAGS python $SIM_PROG $MODEL_PATH $DAT_PATH $SUPP_PATH
#THEANO_FLAGS=$FLAGS python $SICK_PROG 
