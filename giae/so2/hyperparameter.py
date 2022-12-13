import os
DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "saves")

if not os.path.exists(DEFAULT_SAVE_DIR):
    os.makedirs(DEFAULT_SAVE_DIR)



def add_arguments(parser):
    """Helper function to fill the parser object.

    Args:
        parser: Parser object
    Returns:
        parser: Updated parser object
    """

    # GENERAL
    parser.add_argument('-i', '--id', type=int, default=0)
    parser.add_argument('-g', '--gpus', default="1", type=str)
    parser.add_argument('-e', '--num_epochs', default=100, type=int)
    parser.add_argument("--eval_freq", default=250, type=int)
    parser.add_argument("--num_eval_samples", default=200, type=int)
    parser.add_argument("-s", "--save_dir", default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument('--progress_bar', dest='progress_bar', action='store_true')
    parser.set_defaults(progress_bar=False)

    parser.add_argument("--file_path", default="mnist_all_rotation_normalized_float_train_valid.amat", type=str)
    parser.add_argument("--max_epochs", default=10000, type=int)


    parser.add_argument("-b", "--batch_size", default=128, type=int)  # Tetris

    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--resume_ckpt", default="", type=str)

    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--emb_dim", default=32, type=int)

    parser.add_argument('--use_classical', dest='use_classical', action='store_true')
    parser.set_defaults(use_classical=False)


    return parser



