### Run experiment
Download and unzip the rotated MNIST dataset: [Link](https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits) 
```
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
unzip mnist_rotation_new.zip
```
To train the E(2) invariant autoencoder:
```
python main.py --file_path PATH_TO_MNIST_DATA --progress_bar --save_dir saves --hidden_dim 128 --emb_dim 32 -i 0
```
To train classical autoencoder:
```
python main.py --file_path PATH_TO_MNIST_DATA --progress_bar --save_dir mnist_saves --hidden_dim 128 --emb_dim 32 -i 1 --use_classical
```

### Logging
Track the training with Tensorboard, using e.g. port 9999:
```
tensorboard --logdir . --port 9999
```

### Evaluation
We provide an evaluation notebook in `se2_eval.ipynb`.
