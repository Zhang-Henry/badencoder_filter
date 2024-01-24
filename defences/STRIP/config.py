import argparse


def get_argument():
    parser = argparse.ArgumentParser()

    # Directory option
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--checkpoints", type=str, default="../../checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--temps", type=str, default="./temps")

    # ---------------------------- For Neural Cleanse --------------------------
    # Model hyperparameters
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--detection_boundary", type=float, default=0.2)  # According to the original paper
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--test_rounds", type=int, default=10)

    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=4)  # low-res grid size
    parser.add_argument(
        "--grid-rescale", type=float, default=1
    )  # scale grid values to avoid going out of [-1, 1]. For example, grid-rescale = 0.98



    parser.add_argument('--encoder_usage_info', default='', type=str, help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--encoder', default='', type=str, help='path to the image encoder')
    parser.add_argument('--classifier', default='', type=str, help='path to the image encoder')
    parser.add_argument('--gpu', default='0', type=str, help='the index of gpu used to train the model')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lamda', default=0.01, type=float)
    parser.add_argument('--nn_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--noise', type=str)
    parser.add_argument('--reference_file', default='/home/hrzhang/projects/badencoder_filter/reference/cifar10/one.npz', type=str, help='path to the reference file (default: none)')
    parser.add_argument('--trigger_file', default='', type=str)
    parser.add_argument('--reference_label', default=0, type=str)

    return parser
