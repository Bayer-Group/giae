import os
DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "saves0")

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
    parser.add_argument('-e', '--num_epochs', default=50, type=int)
    parser.add_argument("--eval_freq", default=250, type=int)
    parser.add_argument("--num_eval_samples", default=200, type=int)
    parser.add_argument("-s", "--save_dir", default=DEFAULT_SAVE_DIR, type=str)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument('--progress_bar', dest='progress_bar', action='store_true')
    parser.set_defaults(progress_bar=False)
    parser.add_argument("--noise", default=0.01, type=float)
    parser.add_argument("--translation", default=5.0, type=float)

    parser.add_argument("--num_points", default=200, type=int)
    parser.add_argument("--encoder_nearest", default=16, type=int)
    parser.add_argument("--decoder_nearest", default=16, type=int)
    parser.add_argument("--omit_layer_norm", default=False, action="store_true")

    parser.add_argument("-b", "--batch_size", default=512, type=int)

    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--resume_ckpt", default="", type=str)

    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--num_layers", default=5, type=int)
    parser.add_argument("--emb_dim", default=2, type=int)


    return parser

