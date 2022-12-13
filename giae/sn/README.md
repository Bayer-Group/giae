### Run experiment 
To train the S(N) invariant autoencoder:
```
python main.py --progress_bar --save_dir saves --hidden_dim 128 --emb_dim 8 --num_digits 20 --num_classes 10
```
To train the classical autoencoder:
```
python main.py --progress_bar --save_dir digit_set --hidden_dim 128 --emb_dim 8 --num_digits 20 --num_classes 10 --use_classical
```

### Logging
Track the training with Tensorboard, using e.g. port 9999:
```
tensorboard --logdir . --port 9999
```

### Evaluation
Evaulation metrics are provided in the tensorboard log.