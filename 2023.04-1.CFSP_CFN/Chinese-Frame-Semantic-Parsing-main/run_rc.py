from accelerate import Accelerator
from accelerate.utils import set_seed
from arguments import parse_args
from dataset import FrameRCDataset
import logging
import math
from model import BertForFrameSRL
from transformers import DataCollatorWithPadding
import os
# from predict import post_process_function_greedy, calculate_F1_metric, post_process_function_with_max_len, save_predictions
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
# from tqdm import tqdm
import transformers
from transformers import BertConfig, BertTokenizerFast, get_scheduler
import json
import evaluate


logger = logging.getLogger(__name__)

def Predict(args, accelerator, model, eval_dataset, eval_dataloader, fe2id):
    id2fe = {v: k for k, v in fe2id.items()}
    # Evaluation
    logger.info("***** Running Prediction *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    model.eval()

    all_predictions = []
    all_labels = []
    all_spans = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            length = batch.pop('length')
            # word_ids = batch.pop('word_ids')
            task_id = batch.pop('task_id')
            labels = batch.pop('labels')
            outputs = model(**batch)
            logits = outputs.logits
            task_id = task_id.cpu().numpy().tolist()
            # word_ids = word_ids.cpu().numpy().tolist()
            predictions = torch.argmax(logits, dim=-2).cpu().numpy().tolist() # (B, 16)
            span_token_idx = batch['span_token_idx'].cpu().numpy().tolist() # (B, 16, 2)
            # print(task_id)
            # print(span_token_idx)
            # print(predictions)
            # labels = batch['labels'].cpu().numpy().tolist()
            # true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
            true_predictions = [
                [id2fe[p] for (p, span) in zip(prediction, span_token) if span[0] != 0]
                for prediction, span_token in zip(predictions, span_token_idx)
            ]
            # all_spans = []
            for tid, pred in zip(task_id, true_predictions):
                tid = tid[0]
                spans = []
                for p in pred:
                    spans.append([tid, p])
                all_spans += spans

    # precision = .0
    # recall = .0
    # F1 = (2 * precision * recall) / (precision + recall + 1e-12)
    # all_metrics = metric.compute(predictions=all_predictions, references=all_labels)
    # precision = all_metrics['overall_precision']
    # recall = all_metrics['overall_recall']
    # F1 = all_metrics['overall_f1']

    with open('./ccl-cfn/result/task2_test.json', 'r') as f:
        all_spans_no_label = json.load(f)
    for s, s_ in zip(all_spans, all_spans_no_label):
        try:
            assert s[0] == s_[0]
        except:
            print(s, s_)
            exit(0)
        s_.append(s[1])
    with open('./ccl-cfn/result/task3_test.json', 'w') as f:
        json.dump(all_spans_no_label, f, ensure_ascii=False)

    return

def Evaluate(args, accelerator, model, eval_dataset, eval_dataloader, metric):
    # label_names = ['O', 'B', 'I']
    # Evaluation
    logger.info("***** Running Evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    model.eval()

    all_predictions = []
    all_labels = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            length = batch.pop('length')
            # word_ids = batch.pop('word_ids')
            task_id = batch.pop('task_id')
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-2).cpu().numpy().tolist()
            labels = batch['labels'].cpu().numpy().tolist()
            true_labels = [[l for l in label if l != -100] for label in labels]
            true_predictions = [
                [p for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            for tp in true_predictions:
                all_predictions += tp
            # all_predictions += true_predictions
            for tl in true_labels:
                all_labels += tl
            # all_labels += true_labels

    # precision = .0
    # recall = .0
    # F1 = (2 * precision * recall) / (precision + recall + 1e-12)
    all_metrics = metric.compute(predictions=all_predictions, references=all_labels)
    acc = all_metrics['accuracy']
    # recall = all_metrics['overall_recall']
    # F1 = all_metrics['overall_f1']

    return acc

            


def train(args, accelerator, model, train_dataset, train_dataloader, optimizer, lr_scheduler, eval_dataset, eval_dataloader, tokenizer):
    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    if args.save_best:
        best_acc = -1

    metric = evaluate.load('accuracy')
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            length = batch.pop('length')
            # if not args.loss_on_context:
            #     context_length = batch.pop('context_length')
            # word_ids = batch.pop('word_ids')
            # FE_num = batch.pop('FE_num')
            task_id = batch.pop('task_id')
            # gt_FE_word_idx = batch.pop('gt_FE_word_idx')
            # gt_start_positions = batch.pop('gt_start_positions')
            # gt_end_positions = batch.pop('gt_end_positions')
            # FE_core_pts = batch.pop('FE_core_pts')
            try:
                outputs = model(**batch)
            except:
                for k, v in batch.items():
                    print(k, v.shape, v)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            if args.log_every_step is not None and step % args.log_every_step == 0:
                logger.info(f"  | batch loss: {loss.detach().float():.6f} step = {step}")
            # if args.with_tracking:
            #     total_loss += loss.detach().float()
            #     if args.log_every_step is not None and step % args.log_every_step == 0:
            #         logger.info(f"  | batch loss: {loss.detach().float():.6f} step = {step}")
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1


            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        # if args.with_tracking:
        #     logger.info(f"  Epoch Loss {total_loss:.6f}")
        logger.info(f"  Epoch Loss {total_loss:.6f}")

        acc = Evaluate(args, accelerator, model, eval_dataset, eval_dataloader, metric)
        logger.info(f"  Accuracy {acc:.6f}")
        # logger.info(f"  TP: {total_TP} FP: {total_FP} FN: {total_FN}")


        if args.with_tracking:
            log = {
                "train_loss": total_loss,
                "step": completed_steps,
                "acc": acc,
            }
            accelerator.log(log)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        if args.output_dir is not None:
            # print(f'best {best_F1} current {F1}')
            if args.save_best:
                if best_acc <= acc:
                    best_Facc = acc
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir)
            else:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)



def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None and not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    accelerator.wait_for_everyone()

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file

    if args.config_name:
        config = BertConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = BertConfig.from_pretrained(args.model_name_or_path)
    config.update({'FE_pooling':args.FE_pooling})

    if args.tokenizer_name:
        tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name, use_fast=True)
    elif args.model_name_or_path:
        tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.add_tokens(['<t>', '</t>', '<f>', '</f>', '<a>', '</a>'])
    with open(args.frame_data, 'r') as f:
        data = json.load(f)
    fe2id = {}
    cnt = 0
    for frame in data:
        for fe in frame['fes']:
            name = fe['fe_name']
            if name not in fe2id:
                fe2id[name] = cnt
                cnt += 1
    config.num_labels = len(fe2id)

    if args.model_name_or_path:
        model = BertForFrameSRL.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )     
    else:
        logger.info("Training new model from scratch")
        model = BertForFrameSRL.from_config(config)
    model.resize_token_embeddings(len(tokenizer))

    # frame_data = {}
    # with open('./ccl-cfn/frame_data_def.json', 'r') as f:
    #     frame_lines = json.load(f)
    #     for line in frame_lines:
    #         frame_data[line["frame_name"]] = line    

    if "train" not in data_files:
        raise ValueError("--do_train requires a train dataset")
    with accelerator.main_process_first():
        train_dataset = FrameRCDataset(data_files['train'], tokenizer, fe2id)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.subset(range(args.max_train_samples))
    
    if "validation" not in data_files:
        raise ValueError("--do_train requires a train dataset")
    with accelerator.main_process_first():
        eval_dataset = FrameRCDataset(data_files['validation'], tokenizer, fe2id)
        if args.max_eval_samples is not None:
            eval_dataset = eval_dataset.subset(range(args.max_eval_samples))    

    if args.do_predict:
        test_dataset = FrameRCDataset(data_files['test'], tokenizer, fe2id, args.task1_res, args.task2_res)
        if args.max_predict_samples is not None:
            test_dataset = test_dataset.subset(range(args.max_predict_samples))

    # data_collator = DataCollatorForFrameAI(tokenizer=tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )


    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("FrameSRL", experiment_config)
    # evaluate(args, accelerator, model, eval_dataset, eval_dataloader)
    train(args, accelerator, model, train_dataset, train_dataloader, optimizer, lr_scheduler, eval_dataset, eval_dataloader, tokenizer)

    if args.do_predict:
        model = BertForFrameSRL.from_pretrained(
            args.output_dir,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        model, test_dataloader = accelerator.prepare(model, test_dataloader)
        Predict(args, accelerator, model, test_dataset, test_dataloader, fe2id)

if __name__ == '__main__':
    main()