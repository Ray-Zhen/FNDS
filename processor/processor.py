import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
import os
import json
# from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("FNDS.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "itc_loss_tsr": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        model.avg_sims_tool_neg.reset()
        model.var_sims_tool_neg.reset()
        model.avg_sims_tool_pos.reset()
        model.var_sims_tool_pos.reset()

        model.avg_sims_tool_neg_tsr.reset()
        model.var_sims_tool_neg_tsr.reset()
        model.avg_sims_tool_pos_tsr.reset()
        model.var_sims_tool_pos_tsr.reset()

        fn_i2t_count = 0
        fn_t2i_count = 0
        fn_i2t_tsr_count = 0
        fn_t2i_tsr_count = 0

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch, epoch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            # meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['itc_loss_tsr'].update(ret.get('itc_loss_tsr', 0), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

            # false negative 计数
            if "fn_count" in ret.keys():
                fn_i2t_count += ret["fn_count"][0]
                fn_t2i_count += ret["fn_count"][1]
                fn_i2t_tsr_count += ret["fn_count"][2]
                fn_t2i_tsr_count += ret["fn_count"][3]

        logger.info(f'false neagtive i2t count: {fn_i2t_count}')
        logger.info(f'false neagtive t2i count: {fn_t2i_count}')
        logger.info(f'false neagtive i2t tsr count: {fn_i2t_tsr_count}')
        logger.info(f'false neagtive t2i tsr count: {fn_t2i_tsr_count}')

        #########
        logger.info('\n')
        logger.info('Pos:')
        logger.info(f'Average sims: {model.avg_sims_tool_pos.avg}')
        logger.info(f'Number of Average sims: {model.avg_sims_tool_pos.count}')
        logger.info(f'Variance sims: {model.var_sims_tool_pos.var}')
        logger.info(f'Number of Variance sims: {model.var_sims_tool_pos.count}')
        logger.info("Neg:")
        logger.info(f'Average sims: {model.avg_sims_tool_neg.avg}')
        logger.info(f'Number of Average sims: {model.avg_sims_tool_neg.count}')
        logger.info(f'Variance sims: {model.var_sims_tool_neg.var}')
        logger.info(f'Number of Variance sims: {model.var_sims_tool_neg.count}')
        logger.info('\n')
        # tsr
        logger.info('Pos:')
        logger.info(f'Average sims: {model.avg_sims_tool_pos_tsr.avg}')
        logger.info(f'Number of Average sims: {model.avg_sims_tool_pos_tsr.count}')
        logger.info(f'Variance sims: {model.var_sims_tool_pos_tsr.var}')
        logger.info(f'Number of Variance sims: {model.var_sims_tool_pos_tsr.count}')
        logger.info("Neg:")
        logger.info(f'Average sims: {model.avg_sims_tool_neg_tsr.avg}')
        logger.info(f'Number of Average sims: {model.avg_sims_tool_neg_tsr.count}')
        logger.info(f'Variance sims: {model.var_sims_tool_neg_tsr.var}')
        logger.info(f'Number of Variance sims: {model.var_sims_tool_neg_tsr.count}')
        logger.info('\n')
        #########

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)


    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader, loss_type):

    logger = logging.getLogger("FNDS.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, loss_type)
    top1 = evaluator.eval(model.eval())
