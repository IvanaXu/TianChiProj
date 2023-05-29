import argparse
from transformers import SchedulerType

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on Frame Semantic Role Labeling")
    parser.add_argument(
        "--train_file", type=str, default='./ccl-cfn/cfn-train.json', help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=4, help="A csv or a json file containing the training data."
    )
    parser.add_argument("--do_predict", default=True,action="store_true", help="To do prediction on the question answering model")
    parser.add_argument(
        "--validation_file", type=str, default='./ccl-cfn/cfn-dev.json', help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default='./ccl-cfn/cfn-test.json', help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--frame_data", type=str, default='./ccl-cfn/frame_info.json', help="A csv or a json file containing the frame data."
    )
    parser.add_argument(
        "--task1_res", type=str, default='./ccl-cfn/result/task1_test.json', help="A csv or a json file containing the result of task1."
    )
    parser.add_argument(
        "--task2_res", type=str, default='./ccl-cfn/result/task2_test.json', help="A csv or a json file containing the result of task2."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./bert-base-chinese",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        # required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='../ckpt/', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_argument_length",
        type=int,
        default=30,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this "
        "value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this "
        "value if set.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default='epoch',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--FE_pooling",
        type=str,
        default='max',
        help="max or avg, how we do pooling over tokens of an FE.",
    )
    parser.add_argument(
        "--log_every_step",
        type=int,
        default=None,
        help="How many steps do we log loss."
    )
    parser.add_argument(
        "--post_process",
        type=str,
        default='greedy'
    )
    parser.add_argument("--save_best", action="store_true", help="Whether to save model with best performance on dev dataset.")
    parser.add_argument("--loss_on_context", action="store_true", help="Whether to compute loss only on context.")
    parser.add_argument('--target', action="store_true", help="Whether to use target as label.")
    parser.add_argument(
        "--train_file2",
        type=str,
        default='../data/train_instance_dic_prompt.npy',
    )
    parser.add_argument(
        "--num_train_epochs1",
        type=int,
        default=-1
    )
    parser.add_argument(
        '--model_type', 
        type=str,
        default='srl'
    )
    args = parser.parse_args()

    # Sanity checks
    if (
        args.train_file is None
        and args.validation_file is None
        and args.test_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation/test file.")
    # else:
    #     if args.train_file is not None:
    #         extension = args.train_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #     if args.validation_file is not None:
    #         extension = args.validation_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    #     if args.test_file is not None:
    #         extension = args.test_file.split(".")[-1]
    #         assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


    return args