#!/bin/bash

DEVICE=cuda0
MODE=FAST_RUN
PROF=False
FLAGS=mode=$MODE,device=$DEVICE,floatX=float32,profile=$PROF,on_unused_input='ignore' #,optimizer=None,exception_verbosity=high
OUT_PATH=./models # path to directory where the models will be saved
IN_PATH=./data  # path to directory that contains the data files
PROG=train-lm.py #train-brnn.py #train-alt.py
PRE_TRAINED=

if [ $# -ge 1 ]; then
	IN_PATH=$1
	shift
	if [ $# -ge 1 ]; then
		OUT_PATH=$1
		shift
		if [ $# -ge 1 ]; then
			PRE_TRAINED=$1
			shift
		fi
	fi
fi

export CUDA_LAUNCH_BLOCKING=1
THEANO_FLAGS=$FLAGS python $PROG $IN_PATH $OUT_PATH $PRE_TRAINED
