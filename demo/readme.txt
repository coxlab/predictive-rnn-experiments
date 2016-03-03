This file contains instructions on how to run PGN demo.
Here are the steps:

1. Install dependencies of Keras, which can be found at http://keras.io/#installation
   Note you don't have to install Keras itself because I will have you use my branched version.

2. Install hickle if you don't already have it:  pip install hickle

3. Download/clone https://github.com/bill-lotter/keras/tree/PGN.
   That is, download the 'PGN' branch of my forked keras code.

4. Specify paths at top of PGN_demo.py
   data_file_train should be the bouncing_balls file I gave you
   save_dir is where plots will be stored

5. Run demo.  In terminal:
   THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python PGN_demo.py
