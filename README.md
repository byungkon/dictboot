# dictboot
Dictionary bootstrapping code and data.

This project is currently under review at a conference as of Apr. 2017.
As such, please understand that I am forbidden to disclose my identity.

## Setup
For your convenience, we provide the folllowing data to reproduce the results of the paper.
Notice that it may be possible to discover the identity (or at least a clue to it) of the author by actively digging through the URL.

So here they are:
* Trained model URL (Pickled): https://drive.google.com/open?id=0B7V13DJYRnxtaWxwZlAzQW1qeWs
* Data URL (Pickled): https://drive.google.com/open?id=0B7V13DJYRnxtVHBMZlk5eV93Unc
  * Contains definition data (see noparens.out for the raw definition)
* Supplementary data URL (Pickled): https://drive.google.com/open?id=0B7V13DJYRnxtdG42U01oWWZ6OVk
  * Extra information that I forgot to save in the main data file above.. bad design choice.

Try the following approach to bypass the junk data and get straight to the model embeddings.
```
import cPickle as pk
with open('foo.pkl', 'rb') as f: # foo.pkl is the path to the model file
  m = pk.load(f)
embeddings = m['params']['dwe'].get_value() # dwe: Disambiguated Word Embedding
```
## Run
* To begin training, you should modify some of the variables in the run_train.sh file.
OUT_PATH and IN_PATH are the path variables that contain the directory to output the model file, and the directory that contains the data and supplementary files, respectively.
* To reproduce the results of WSI, you must first install the dataset [provided by SemEval](https://www.cs.york.ac.uk/semeval-2013/task13/index.php%3Fid=data.html). Then, modify the WSI_PATH variable in the run_wsi.sh script before running it.
