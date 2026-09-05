import argparse
import wandb, os

# Disable tokenizer parallelism warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.data_manager import DataManager, setup_seed
from utils.toolkit import count_parameters
from utils.Logger import Logger
from methods.finetune import Finetune
from methods.icarl import iCaRL
from methods.lwf import LwF
from methods.ewc import EWC
from methods.target import TARGET
from methods.lander import LANDER
from methods.ours import Ours
from methods.fedrnc import FedRNC
from methods.wsdm import WSDM
from forgetting_calculator.forgetting_cal import calculate_forgetting
import warnings

warnings.filterwarnings("ignore")


def get_learner(model_name, logger, args):
    name = model_name.lower()
    if name == "icarl":
        return iCaRL(args, logger)
    elif name == "ewc":
        return EWC(args, logger)
    elif name == "lwf":
        return LwF(args, logger)
    elif name == "finetune":
        return Finetune(args, logger)
    elif name == "target":
        return TARGET(args, logger)
    elif name == "lander":
        return LANDER(args, logger)
    elif name == "ours":
        return Ours(args, logger)
    elif name == "fedrnc":
        return FedRNC(args, logger)
    elif name == "wsdm":
        return WSDM(args, logger)
    else:
        assert 0


def train(args):
    args.logdir = os.path.join(args.logdir, args.group + "_" + args.exp_name)
    logger = Logger(args, loss_logger=args.log_loss)
    # logger.log(args)

    args = vars(args)
    setup_seed(args["seed"])
    # setup the dataset and labels
    data_manager = DataManager(
        args["data_paths"], args["dataset_names"], args["classes"]
    )
    learner = get_learner(args["method"], logger, args)

    cnn_curve = []

    # train for each task
    for task in range(data_manager.nb_tasks):
        logger.log(
            "----------------------------------------------------------------------------------------------------------------------"
        )
        para = "All params: {}, Trainable params: {}".format(
            count_parameters(learner._network), count_parameters(learner._network, True)
        )
        print(para)
        logger.log(para)

        learner.incremental_train(data_manager)  # train for one task
        result_auc = learner.eval_task(data_manager)
        learner.after_task()

        cnn_curve.append(result_auc)
        print("CNN curve: {}".format(cnn_curve))
        logger.log("CNN curve: {}".format(cnn_curve))

    F_k = calculate_forgetting(args["logdir"])
    info = f"Last AUC: {cnn_curve[-1]:.4f}, Average AUC: {sum(cnn_curve) / len(cnn_curve):.4f}, Forgetting Rate(%): {F_k * 100:.4f}"
    print(info)
    logger.log(info)


def args_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark for federated continual learning"
    )
    # Exp settings
    parser.add_argument(
        "--illustration",
        type=str,
        default="",
        help="Short experiment note stored in the log",
    )
    parser.add_argument(
        "--log_loss", type=int, default=0, help="Log the training loss curve (0/1)"
    )
    parser.add_argument(
        "--exp_name", type=str, default="lander_b0", help="name of this experiment"
    )
    parser.add_argument("--wandb", type=int, default=0, help="1 for using wandb")
    parser.add_argument("--save_dir", type=str, default="", help="save the syn data")
    parser.add_argument(
        "--syn_data_dir",
        type=str,
        default=None,
        help="external synthetic data root dir",
    )
    parser.add_argument("--logdir", type=str, default="./result", help="log dir")
    parser.add_argument("--project", type=str, default="HB-FCL", help="wandb project")
    parser.add_argument("--group", type=str, default="cxr", help="wandb group")
    parser.add_argument("--seed", type=int, default=2026, help="random seed")
    parser.add_argument("--spec", type=str, default="t1", help="choose a model")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument(
        "--resume", type=int, default=0, help="Resume from a checkpoint (0/1)"
    )

    # federated continual learning settings
    parser.add_argument(
        "--method",
        type=str,
        default="lander",
        help="choose a learner (finetune, ewc, lwf, icarl, target, lander, ours, wsdm, fedrnc)",
    )
    parser.add_argument("--net", type=str, default="desnet121", help="choose a model")
    parser.add_argument(
        "--com_rounds", type=int, nargs="+", help="communication rounds each task"
    )
    parser.add_argument("--local_ep", type=int, default=2, help="local training epochs")
    parser.add_argument("--num_users", type=int, default=5, help="num of clients")
    parser.add_argument("--local_bs", type=int, default=128, help="local batch size")
    parser.add_argument(
        "--beta", type=float, default=0.0, help="control the degree of label skew"
    )
    parser.add_argument(
        "--frac", type=float, default=1.0, help="the fraction of selected clients"
    )
    parser.add_argument("--class_shuffle", type=int, default=1, help="class shuffle")
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=["ChestXR"],
        help="dataset names in sequential order",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[3],
        help="class num of each dataset in sequential order",
    )
    parser.add_argument(
        "--data_paths",
        type=str,
        nargs="+",
        default=["/path/to/CXR_Covid-19_Challenge-001"],
        help="path for each dataset",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="evaluation threshold"
    )

    # Client Training
    parser.add_argument(
        "--num_worker", type=int, default=4, help="number of worker for dataloader"
    )
    parser.add_argument(
        "--mulc", type=str, default="fork", help="type of multi process for dataloader"
    )
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="weight decay for optimizer"
    )
    parser.add_argument(
        "--syn_bs",
        default=1,
        type=int,
        help="number of old synthetic data in training, 1 for similar to local_bs",
    )
    parser.add_argument(
        "--local_lr", default=4e-2, type=float, help="learning rate for optimizer"
    )

    # Baseline WSDM settings
    parser.add_argument(
        "--wsdm_clusters",
        type=int,
        default=2,
        help="cluster number in WSDM diversity clustering",
    )
    parser.add_argument(
        "--wsdm_m", type=float, default=1.1, help="fuzzy index in WSDM c-means"
    )
    parser.add_argument(
        "--wsdm_T", type=int, default=1000, help="diffusion time steps in WSDM"
    )
    parser.add_argument(
        "--wsdm_gen_batch_size",
        type=int,
        default=7,
        help="generation batch size per condition in WSDM",
    )
    parser.add_argument(
        "--wsdm_train_batch_size",
        type=int,
        default=1,
        help="diffusion training batch size in WSDM",
    )
    parser.add_argument(
        "--wsdm_amp",
        type=int,
        default=1,
        help="enable AMP in WSDM diffusion training (1 on, 0 off)",
    )
    parser.add_argument(
        "--wsdm_mid_round_ratio",
        type=float,
        default=0.5,
        help="round ratio to trigger WSDM generation",
    )
    parser.add_argument(
        "--wsdm_image_size", type=int, default=64, help="diffusion image size in WSDM"
    )
    parser.add_argument(
        "--wsdm_save_path",
        type=str,
        default="",
        help="directory for WSDM generated data",
    )

    # Ours
    parser.add_argument(
        "--use_semantic",
        type=int,
        default=1,
        help="Use semantic embeddings (0/1; ablation)",
    )
    parser.add_argument("--r", default=0.015, type=float, help="LTE center radius")
    parser.add_argument(
        "--ltc", default=5, type=float, help="lamda_ltc parameter for LTE center"
    )
    parser.add_argument(
        "--pre",
        type=float,
        default=0.2,
        help="alpha_pre for distilling from previous task",
    )
    parser.add_argument(
        "--cur", type=float, default=0.8, help="alpha_cur for current task training"
    )

    # TSE (Task-specific Selection Expert) Settings
    parser.add_argument(
        "--use_tse_filter", type=int, default=1, help="Use TSE filtering (0/1)"
    )
    parser.add_argument(
        "--tse_lr",
        type=float,
        default=0.1,
        help="TSE local learning rate; defaults to local_lr",
    )
    parser.add_argument(
        "--tse_frac",
        type=float,
        default=1.0,
        help="TSE client sampling fraction; defaults to frac",
    )
    parser.add_argument(
        "--tse_com_rounds",
        type=int,
        nargs="+",
        default=None,
        help="TSE communication rounds",
    )
    parser.add_argument(
        "--tse_local_ep", type=int, default=0, help="TSE local epochs; 0 uses local_ep"
    )
    parser.add_argument(
        "--tse_batch_size",
        type=int,
        default=0,
        help="TSE local batch size; 0 uses local_bs",
    )
    parser.add_argument(
        "--tse_weight_decay",
        type=float,
        default=2e-4,
        help="TSE optimizer weight decay; defaults to weight_decay",
    )
    parser.add_argument(
        "--tse_gamma_neg",
        type=float,
        default=4.0,
        help="Asymmetric Loss focusing factor for negatives",
    )
    parser.add_argument(
        "--tse_gamma_pos",
        type=float,
        default=1.0,
        help="Asymmetric Loss focusing factor for positives",
    )
    parser.add_argument(
        "--tse_clip",
        type=float,
        default=0.05,
        help="Asymmetric Loss negative-probability clip",
    )
    parser.add_argument(
        "--tse_eps",
        type=float,
        default=1e-8,
        help="Asymmetric Loss numerical-stability epsilon",
    )
    parser.add_argument(
        "--tse_filter_bs",
        type=int,
        default=256,
        help="TSE filtering inference batch size",
    )
    parser.add_argument(
        "--tse_filter_radius",
        type=float,
        default=0.015,
        help="Stage-one MSE radius for TSE and LTE embeddings",
    )
    parser.add_argument(
        "--tse_ltc", type=float, default=5.0, help="TSE LTC loss weight"
    )
    parser.add_argument(
        "--pretrained_tse_model_path",
        type=str,
        default="",
        help="Pretrained TSE path; skips task-level TSE training when set",
    )

    # MINIM Diffusion Model Settings
    parser.add_argument(
        "--nums", type=int, default=1024, help="the final num of synthetic data"
    )
    parser.add_argument(
        "--pre_nums",
        type=int,
        default=2048,
        help="the num of synthetic data after filtering",
    )
    parser.add_argument(
        "--use_minim",
        type=int,
        default=1,
        help="Use MINIM for data generation (0/1)",
    )
    parser.add_argument(
        "--minim_pretrained_model_path",
        type=str,
        default="/path/to/RoentGen-v2",
        help="Path to the pretrained Stable Diffusion model",
    )

    # MINIM Training Parameters
    parser.add_argument(
        "--minim_train_batch_size",
        type=int,
        default=128,
        help="MINIM training batch size",
    )
    parser.add_argument(
        "--minim_gradient_accumulation_steps",
        type=int,
        default=1,
        help="MINIM gradient accumulation steps",
    )
    parser.add_argument(
        "--minim_learning_rate", type=float, default=1e-5, help="MINIM learning rate"
    )
    parser.add_argument(
        "--minim_com_rounds",
        type=int,
        default=5,
        help="MINIM federated communication rounds",
    )
    parser.add_argument(
        "--minim_local_epochs",
        type=int,
        default=1,
        help="MINIM local epochs per client",
    )
    parser.add_argument(
        "--minim_frac", type=float, default=1.0, help="MINIM client sampling fraction"
    )
    parser.add_argument(
        "--minim_temp_dir", type=str, default="/tmp", help="MINIM temporary directory"
    )
    parser.add_argument(
        "--minim_lr_scheduler",
        type=str,
        default="constant",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="MINIM learning-rate scheduler",
    )
    parser.add_argument(
        "--minim_lr_warmup_steps", type=int, default=100, help="MINIM warmup steps"
    )
    parser.add_argument(
        "--minim_resolution", type=int, default=256, help="MINIM image resolution"
    )
    parser.add_argument(
        "--minim_center_crop", action="store_true", help="Use center cropping for MINIM"
    )
    parser.add_argument(
        "--minim_random_flip",
        action="store_true",
        help="Use random horizontal flips for MINIM",
    )
    parser.add_argument(
        "--minim_mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="MINIM mixed-precision mode",
    )
    parser.add_argument(
        "--minim_enable_xformers",
        type=int,
        default=0,
        help="Enable xFormers for MINIM (0/1)",
    )
    parser.add_argument(
        "--minim_use_ema", action="store_true", help="Use EMA for MINIM"
    )
    parser.add_argument(
        "--minim_noise_offset", type=float, default=0, help="MINIM noise offset"
    )
    parser.add_argument(
        "--minim_max_grad_norm",
        type=float,
        default=1.0,
        help="MINIM maximum gradient norm",
    )
    parser.add_argument(
        "--minim_dataloader_num_workers",
        type=int,
        default=4,
        help="MINIM data-loader workers",
    )
    parser.add_argument(
        "--minim_adam_beta1", type=float, default=0.9, help="MINIM Adam beta1"
    )
    parser.add_argument(
        "--minim_adam_beta2", type=float, default=0.999, help="MINIM Adam beta2"
    )
    parser.add_argument(
        "--minim_adam_weight_decay",
        type=float,
        default=0.01,
        help="MINIM Adam weight decay",
    )
    parser.add_argument(
        "--minim_adam_epsilon", type=float, default=1e-8, help="MINIM Adam epsilon"
    )

    # MINIM Generation Parameters
    parser.add_argument(
        "--minim_generation_batch_size",
        type=int,
        default=8,
        help="MINIM generation batch size",
    )
    parser.add_argument(
        "--minim_num_inference_steps",
        type=int,
        default=50,
        help="MINIM inference steps",
    )
    parser.add_argument(
        "--minim_guidance_scale", type=float, default=7.5, help="MINIM guidance scale"
    )
    parser.add_argument(
        "--minim_filter_max_rounds",
        type=int,
        default=5,
        help="Maximum MINIM regeneration attempts per label combination",
    )

    parser.add_argument(
        "--select_img_seed", type=int, default=2026, help="Seed for image selection"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = args_parser()
    assert (
        len(args.dataset_names) == len(args.classes)
        and len(args.data_paths) == len(args.classes)
        and len(args.com_rounds) == len(args.classes)
    ), "The input datasets, paths, classes and com_rounds do not match"
    if args.tse_com_rounds is not None:
        assert len(args.tse_com_rounds) == len(
            args.classes
        ), "The input tse_com_rounds do not match"
    args.tasks = len(args.dataset_names)
    print(args)

    args.exp_name = f"{args.group}_{args.exp_name}"

    dir = "store"
    if not os.path.exists(dir):
        os.makedirs(dir)

    dir = "run"
    if not os.path.exists(dir):
        os.makedirs(dir)
    args.save_dir = os.path.join(dir, args.group + "_" + args.exp_name + "" + args.spec)

    if args.wandb == 1:
        wandb.init(
            config=args, project=args.project, group=args.group, name=args.exp_name
        )
        wandb.run.log_code(".")  # Log the project source to Weights & Biases.

    train(args)
