import glob
import logging
import os
import json
import time
import torch
import numpy as np
import torch.nn as nn

from transformers import WEIGHTS_NAME, AutoConfig, AutoTokenizer
from models.bert_gp import BertGPForNer,ErnieGPForNer,NezhaGPForNer,ElectraGPForNer
from processors.ner_gp import gen_dataloader,gen_test_dataloader
from metrics.gp_metric import GPEntityScore

from optimizer.gp_opt import get_optimizer
from util_func.kl_loss import compute_kl_loss
from util_func.args import get_args
from util_func.seed import set_seed
from util_func.adv import FGM
from util_func.init_logger import init_logger, logger
from util_func.progressbar import ProgressBar
from util_func.read_file import read_label_list
from util_func.data_format import gen_BIO


MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (AutoConfig, BertGPForNer, AutoTokenizer),
    'ernie':(AutoConfig, ErnieGPForNer, AutoTokenizer),
    'nezha':(AutoConfig, NezhaGPForNer, AutoTokenizer),
    'electra':(AutoConfig, ElectraGPForNer, AutoTokenizer),
}


def train(args, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloader,dev_dataloader=gen_dataloader(args.train_data_path,args.dev_data_path,tokenizer,args.train_batch_size,label_txt=args.label_txt)
    optimizer, scheduler = get_optimizer(args, model, len(train_dataloader))

    # 参数
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # 多卡训练
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # 分布式训练
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # 对抗训练
    if args.do_adv:
        fgm = FGM(model, emb_name=args.adv_name, epsilon=args.adv_epsilon)

    # 开始训练
    logger.info("***** Running training *****")
    logger.info("  Num steps = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # 是否继续训练
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    tr_loss, logging_loss = 0.0, 0.0

    if args.save_steps == -1 and args.logging_steps == -1:
        args.logging_steps = len(train_dataloader)
        args.save_steps = len(train_dataloader)

    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
    for epoch in range(int(args.num_train_epochs)):
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch)
        for step, batch in enumerate(train_dataloader):
            model.train()
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            for key in batch:
                batch[key] = batch[key].to(args.device)

            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                batch["token_type_ids"] = (batch["token_type_ids"] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**batch)

            # loss 计算
            loss = outputs["loss"]

            #rdrop
            if args.do_rdrop:
                outputs2 = model(**batch.clone())
                loss2=outputs2["loss"]
                logits = outputs['logits']
                logits2=outputs2['logits']

                ce_loss = 0.5 * (loss + loss2)
                bh = logits.shape[0] * logits.shape[1]
                kl_loss = compute_kl_loss(logits.reshape([bh, -1]), logits2.reshape([bh, -1]))
                loss = ce_loss + args.rdrop_rate * kl_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.do_adv:
                fgm.attack()
                loss_adv = model(**batch)["loss"]
                if args.n_gpu > 1:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()

            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()

            # 梯度优化
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                # evaluate
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    logger.info("\n")
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        evaluate(args, dev_dataloader, model)

                # save model
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    tokenizer.save_vocabulary(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, dev_dataloader,model, prefix=""):
    metric = GPEntityScore(args.id2label)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(dev_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    pbar = ProgressBar(n_total=len(dev_dataloader), desc="Evaluating")
    for step, batch in enumerate(dev_dataloader):
        model.eval()
        with torch.no_grad():
            for key in batch:
                batch[key] = batch[key].to(args.device)

            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                batch["token_type_ids"] = (batch['token_type_ids'] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**batch)

        tmp_eval_loss, logits = outputs['loss'], outputs['logits']

        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        label_ids = batch['label_ids'].cpu().numpy()
        pred = logits.cpu().numpy()

        metric.update(y_true=label_ids,y_pred=pred)

        pbar(step)
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    return results

def predict(args, model, tokenizer,threshold=0,prefix=""):
    # 多卡预测
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    test_dataloader,dataset=gen_test_dataloader(args.predict_data_path, tokenizer, batch_size=args.per_gpu_eval_batch_size,max_length=args.eval_max_seq_length, return_dataset=True)

    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataloader))
    logger.info("  Batch size = %d",args.per_gpu_eval_batch_size )

    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    nums = 0

    model.eval()
    with open(args.result_data_path, 'w',encoding='utf-8') as f:
        if args.save_type == 'span_csv':
            f.write('query' + '\t' + 'label'+ '\n')
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                label_offset_mapping=batch.pop('offset_mapping').cpu().numpy()
                for key in batch:
                    batch[key] = batch[key].to(args.device)

                outputs = model(**batch)

                logits = outputs['logits']
                pred= logits.cpu().numpy()
                batch_size = pred.shape[0]

                for k in range(batch_size):
                    try:

                        offset_mapping = label_offset_mapping[k].tolist()
                        entities = []

                        text = dataset[nums]
                        nums += 1

                        for category, start, end in zip(*np.where(pred[k] > threshold)):

                            entitie_ = {
                                "start": offset_mapping[start][0],
                                "end": offset_mapping[end - 1][-1],
                                "text": text[offset_mapping[start][0]:offset_mapping[end - 1][-1]],
                                "labels": args.id2label[category]
                            }

                            if entitie_['text'] == '':
                                continue

                            entities.append(entitie_)

                        #save result
                        #可以选择保存不同类似文件格式，与训练文件相同的span类型，BIO txt ，以及目前贝基那边工程接口输出的csv类型
                        if args.save_type == 'bio_txt':
                            bio_result = gen_BIO(text, entities)
                            text=list(text)
                            for id in range(len(text)):
                                f.write(text[id] + '\t' + bio_result[id] + '\n')
                            f.write('\n')
                        elif args.save_type == 'span_csv':
                            f.write(text + '\t' + json.dumps(entities, ensure_ascii=False) + '\n')

                        elif args.save_type == 'span_json':
                            v={'query': text, 'label': entities}
                            f.write(json.dumps(v, ensure_ascii=False) + '\n')
                        elif args.save_type == 'bio_csv':
                            bio_result = gen_BIO(text, entities)
                            v = {'tokens': list(text), 'labels': bio_result}
                            f.write(text + '\t' + json.dumps({text: v}, ensure_ascii=False) + '\n')
                        else:
                            pass
                    except:
                         print('error')
                         nums += 1

                pbar(step)
            logger.info("\n")

def main():
    args = get_args()
    # Set seed
    set_seed(args.seed)

    #save file and log file
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )


    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()

    label_list=read_label_list(args.label_txt)
    args.id2label={v: k for v, k in enumerate(label_list)}
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path,num_labels=len(label_list))
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path,config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    if (not args.do_train) and args.do_eval:
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_dataloader, dev_dataloader = gen_dataloader(args.train_data_path, args.dev_data_path, tokenizer,
                                                          args.train_batch_size, label_txt=args.label_txt)
        evaluate(args, dev_dataloader, model)


    if args.do_predict and args.local_rank in [-1, 0]:

        predict(args, model, tokenizer)

if __name__ == "__main__":
    main()

