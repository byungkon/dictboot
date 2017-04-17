# dictboot
Dictionary bootstrapping code and data.

For your convenience, we provide the folllowing data to reproduce the results of the paper.
Notice that it may be possible to discover the identity (or at least a clue to it) of the author by actively digging through the URL.
(If you are a reviewer, you might be discouraged to seek who I am..)

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

To begin training, you should modify some of the variables in the run_train.sh file.
OUT_PATH and IN_PATH are the path variables that contain the directory to output the model file, and the directory that contains the data and supplementary files, respectively.
