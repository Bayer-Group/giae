### Run experiment 
```
python main.py --progress_bar --save_dir tetris_se3 --num_layers 5 --hidden_dim 32 --emb_dim 2 --num_epochs 100
```

### Logging
Track the training with Tensorboard, using e.g. port 9999:
```
tensorboard --logdir . --port 9999
```

### Evaluation
We provide an evaluation notebook in `analyze_results.ipynb`.
