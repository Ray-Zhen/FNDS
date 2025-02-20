import argparse


def get_args():
	parser = argparse.ArgumentParser(description="FNDS Args")
	######################## general settings ########################
	parser.add_argument("--local_rank", default=0, type=int)
	parser.add_argument("--name", default="baseline", help="experiment name to save")
	parser.add_argument("--output_dir", default="logs")
	parser.add_argument("--log_period", default=100)
	parser.add_argument("--eval_period", default=1)
	parser.add_argument("--val_dataset", default="test")  # use val set when evaluate, if test use test set
	parser.add_argument("--resume", default=False, action='store_true')
	parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')
	parser.add_argument("--select_ratio", default=0.3, type=float)

	######################## model general settings ########################
	parser.add_argument("--pretrain_choice", default='ViT-B/16')  # whether use pretrained model
	parser.add_argument("--temperature", type=float, default=0.02,
	                    help="initial temperature value, if 0, don't use temperature")
	parser.add_argument("--img_aug", default=False, action='store_true')
	parser.add_argument("--txt_aug", default=False, action='store_true')
	# for momentum model
	parser.add_argument("--queue_size", type=int, default=8192, help="momentum model queue size")
	parser.add_argument("--momentum_par", type=float, default=0.995, help="momentum model paramter")
	parser.add_argument('--nums_right_sims', default=200, type=int, help='number of average sims can use.')
	parser.add_argument('--match_para', default=10000, type=int, help='the probability for event 𝑐, 1/c')
	parser.add_argument('--false_margin', default=0.999, type=float, help='lambda')
	parser.add_argument('--false_num', default=100, type=int, help='false num eliminate')
	parser.add_argument('--reversed_p', default=0.8, type=float, help='reversed_p')
	parser.add_argument('--distill', default=False, action='store_true', help='momentual distillation')
	parser.add_argument('--alpha_distill', default=0.4, type=float, help='distillation parameter')
	parser.add_argument('--fnm', default=False, action='store_true', help='false negative elimination')
	parser.add_argument('--margin', default=0.2, type=float, help='LG loss margin')

	## cross modal transfomer setting
	parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
	parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
	parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
	parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
	parser.add_argument("--MLM", default=False, action='store_true',
	                    help="whether to use Mask Language Modeling dataset")

	######################## loss settings ########################
	parser.add_argument("--loss_names", default='sdm+id+mlm',
	                    help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
	parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
	parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
	parser.add_argument("--loss_type", default='fusion', help="which loss type to use ['global', 'local', 'fusion']")

	######################## vison trainsformer settings ########################
	parser.add_argument("--img_size", type=int, nargs=2, default=(384, 128))
	parser.add_argument("--stride_size", type=int, default=16)

	######################## text transformer settings ########################
	parser.add_argument("--text_length", type=int, default=77)
	parser.add_argument("--vocab_size", type=int, default=49408)

	######################## solver ########################
	parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
	parser.add_argument("--lr", type=float, default=1e-5)
	parser.add_argument("--bias_lr_factor", type=float, default=2.)
	parser.add_argument("--momentum", type=float, default=0.9)
	parser.add_argument("--weight_decay", type=float, default=4e-5)
	parser.add_argument("--weight_decay_bias", type=float, default=0.)
	parser.add_argument("--alpha", type=float, default=0.9)
	parser.add_argument("--beta", type=float, default=0.999)

	######################## scheduler ########################
	parser.add_argument("--num_epoch", type=int, default=60)
	parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))  # for step lrscheduler
	parser.add_argument("--gamma", type=float, default=0.1)  # for step lrscheduler factor
	parser.add_argument("--warmup_factor", type=float, default=0.1)
	parser.add_argument("--warmup_epochs", type=int, default=5)
	parser.add_argument("--warmup_method", type=str, default="linear", help="linear, constant")
	parser.add_argument("--lrscheduler", type=str, default="cosine", help="Only 'step', 'exp', 'poly' or 'cosine' learning rate scheduler accepted")
	parser.add_argument("--target_lr", type=float, default=0)
	parser.add_argument("--power", type=float, default=0.9)

	######################## dataset ########################
	parser.add_argument("--dataset_name", default="cuhk_peds", help="[cuhk_peds, icfg_peds, rstp_reid]")
	parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
	parser.add_argument("--num_instance", type=int, default=4)
	parser.add_argument("--root_dir", default="./data")
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--test_batch_size", type=int, default=512)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--test", dest='training', default=True, action='store_false')

	args = parser.parse_args()

	return args
