from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .TokenSelectionRepresentation import TexualEmbeddingLayer, VisualEmbeddingLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import logging
import random


class FNDS(nn.Module):
	def __init__(self, args, num_classes=11003):
		super().__init__()
		self.args = args
		self.num_classes = num_classes
		self._set_task()
		self.attn_weight = False
		self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
		                                                              args.stride_size)
		self.embed_dim = base_cfg['embed_dim']

		self.logger = logging.getLogger("FNDS.train")

		self.logit_scale = torch.ones([]) * (1 / args.temperature)
		self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
		self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

		# momentum model
		self.base_model_m, _ = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
		self.visul_emb_layer_m = VisualEmbeddingLayer(ratio=args.select_ratio)
		self.texual_emb_layer_m = TexualEmbeddingLayer(ratio=args.select_ratio)

		self.criterion = nn.CrossEntropyLoss()
		self.txt_aug = args.txt_aug

		self.fnm = args.fnm
		self.fnr = False
		if self.fnr:
			self.logger.info("Traing with FNR !")

		self.adaptive = True
		self.threshld = 0.90
		self.gmm_glob = GaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4,
		                                warm_start=True)  # warm_start调用会以上一次 fit（）的结果作为初始化参数
		self.gmm_loc = GaussianMixture(n_components=2, max_iter=50, tol=1e-2, reg_covar=5e-4, warm_start=True)

		self.queue_size = args.queue_size
		self.momentum = args.momentum_par
		self.reversed_p = args.reversed_p

		self.model_pairs = [[self.base_model, self.base_model_m], [self.visul_emb_layer, self.visul_emb_layer_m],
		                    [self.texual_emb_layer, self.texual_emb_layer_m]]

		self.copy_params()

		self.register_buffer("image_queue", torch.randn(512, self.queue_size))
		self.register_buffer("text_queue", torch.randn(512, self.queue_size))
		self.register_buffer("image_queue_tsr", torch.randn(1024, self.queue_size))
		self.register_buffer("text_queue_tsr", torch.randn(1024, self.queue_size))
		self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
		self.register_buffer("index_queue", torch.full((1, self.queue_size), -100))
		self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

		self.image_queue = F.normalize(self.image_queue, dim=0)
		self.text_queue = F.normalize(self.text_queue, dim=0)
		self.image_queue_tsr = F.normalize(self.image_queue_tsr, dim=0)
		self.text_queue_tsr = F.normalize(self.text_queue_tsr, dim=0)

	def _set_task(self):
		loss_names = self.args.loss_names
		self.current_task = [l.strip() for l in loss_names.split('+')]
		print(f'Training Model with {self.current_task} tasks')

	def encode_image(self, image):
		x, weight = self.base_model.encode_image(image)
		if self.attn_weight:
			return x[:, 0, :].float(), weight
		else:
			return x[:, 0, :].float()

	# return x.float() # for CLIP ResNet visual model

	def encode_text(self, text):
		x, weight = self.base_model.encode_text(text)
		x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
		if self.attn_weight:
			return x, weight
		else:
			return x

	def encode_image_tsr(self, image):
		x, atten_i = self.base_model.encode_image(image)
		i_tsr_f = self.visul_emb_layer(x, atten_i)
		return i_tsr_f.float()

	def encode_text_tsr(self, text):
		x, atten_t = self.base_model.encode_text(text.long())
		t_tsr_f = self.texual_emb_layer(x, text, atten_t)
		return t_tsr_f.float()

	@torch.no_grad()
	def copy_params(self):
		for model_pair in self.model_pairs:
			for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
				param_m.data.copy_(param.data)  # initialize
				param_m.requires_grad = False  # not update by gradient

	@torch.no_grad()
	def _momentum_update(self):
		for model_pair in self.model_pairs:
			for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
				param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

	@torch.no_grad()
	def _dequeue_and_enqueue(self, image_feats, text_feats, image_feats_tsr, text_feats_tsr, idxs, index):
		# gather keys before updating queue
		batch_size = image_feats.shape[0]
		ptr = int(self.queue_ptr)
		assert self.queue_size % batch_size == 0  # for simplicity

		# replace the keys at ptr (dequeue and enqueue)
		self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
		self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
		self.image_queue_tsr[:, ptr:ptr + batch_size] = image_feats_tsr.T
		self.text_queue_tsr[:, ptr:ptr + batch_size] = text_feats_tsr.T
		self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
		self.index_queue[:, ptr:ptr + batch_size] = index.T

		ptr = (ptr + batch_size) % self.queue_size  # move pointer

		self.queue_ptr[0] = ptr

	def forward(self, batch, epoch):
		ret = dict()
		images = batch['images']

		if self.txt_aug:
			caption_ids = batch['caption_ids_aug']
		else:
			caption_ids = batch['caption_ids']

		image_feats, atten_img, text_feats, atten_txt = self.base_model(images, caption_ids)
		i_feats = image_feats[:, 0, :].float()  # b, 512
		i_feats = F.normalize(i_feats, dim=-1)

		t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()  # b, 512
		t_feats = F.normalize(t_feats, dim=-1)

		i_feats_tsr = self.visul_emb_layer(image_feats, atten_img)  # b, 1024
		i_feats_tsr = F.normalize(i_feats_tsr, dim=-1)
		t_feats_tsr = self.texual_emb_layer(text_feats, caption_ids, atten_txt)  # b, 1024
		t_feats_tsr = F.normalize(t_feats_tsr, dim=-1)

		idx = batch['pids'].view(-1, 1)  # [b, 1]
		idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)  # 1, 8192+b
		pos_idx = torch.eq(idx, idx_all)
		sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

		index = batch['index'].view(-1, 1)
		index_all = torch.cat([index.t(), self.index_queue.clone().detach()], dim=1)  # 1, 8192+b

		logit_scale = self.logit_scale
		ret.update({'temperature': 1 / logit_scale})

		with torch.no_grad():
			self._momentum_update()
			image_feats_m, atten_img_m, text_feats_m, atten_txt_m = self.base_model_m(images, caption_ids)
			i_feats_m = image_feats_m[:, 0, :].float()
			i_feats_m = F.normalize(i_feats_m, dim=-1)

			t_feats_m = text_feats_m[torch.arange(text_feats_m.shape[0]), caption_ids.argmax(dim=-1)].float()
			t_feats_m = F.normalize(t_feats_m, dim=-1)

			i_feats_tsr_m = self.visul_emb_layer_m(image_feats_m, atten_img_m)
			i_feats_tsr_m = F.normalize(i_feats_tsr_m, dim=-1)
			t_feats_tsr_m = self.texual_emb_layer_m(text_feats_m, caption_ids, atten_txt_m)
			t_feats_tsr_m = F.normalize(t_feats_tsr_m, dim=-1)

			i_feats_all = torch.cat([i_feats_m.t(), self.image_queue.clone().detach()], dim=1)
			t_feats_all = torch.cat([t_feats_m.t(), self.text_queue.clone().detach()], dim=1)
			i_feats_all_tsr = torch.cat([i_feats_tsr_m.t(), self.image_queue_tsr.clone().detach()], dim=1)
			t_feats_all_tsr = torch.cat([t_feats_tsr_m.t(), self.text_queue_tsr.clone().detach()], dim=1)

		sim_i2t = i_feats @ t_feats_all
		sim_t2i = t_feats @ i_feats_all
		sim_i2t_tsr = i_feats_tsr @ t_feats_all_tsr
		sim_t2i_tsr = t_feats_tsr @ i_feats_all_tsr

		if self.fnm and epoch >= 0:
			with torch.no_grad():
				sim_neg_glob = torch.cat([sim_i2t.masked_select(~pos_idx), sim_t2i.masked_select(~pos_idx)])
				sim_neg_loc = torch.cat([sim_i2t_tsr.masked_select(~pos_idx), sim_t2i_tsr.masked_select(~pos_idx)])

				sim_neg_glob = (sim_neg_glob - sim_neg_glob.min()) / (
							sim_neg_glob.max() - sim_neg_glob.min())  # 归一化,
				sim_neg_glob = sim_neg_glob.reshape(-1, 1)
				sim_neg_loc = (sim_neg_loc - sim_neg_loc.min()) / (sim_neg_loc.max() - sim_neg_loc.min())  # 归一化
				sim_neg_loc = sim_neg_loc.reshape(-1, 1)

				self.gmm_glob.fit(sim_neg_glob.cpu().numpy())
				self.gmm_loc.fit(sim_neg_loc.cpu().numpy())

				sim_i2t_norm = (sim_i2t - sim_i2t.min()) / (sim_i2t.max() - sim_i2t.min())
				sim_t2i_norm = (sim_t2i - sim_t2i.min()) / (sim_t2i.max() - sim_t2i.min())
				sim_i2t_tsr_norm = (sim_i2t_tsr - sim_i2t_tsr.min()) / (sim_i2t_tsr.max() - sim_i2t_tsr.min())
				sim_t2i_tsr_norm = (sim_t2i_tsr - sim_t2i_tsr.min()) / (sim_t2i_tsr.max() - sim_t2i_tsr.min())

				# pred_i2t = self.gmm_pred(sim_i2t, self.gmm_glob, pos_idx)
				# pred_t2i = self.gmm_pred(sim_t2i, self.gmm_glob, pos_idx)
				# pred_i2t_tsr = self.gmm_pred(sim_i2t_tsr, self.gmm_loc, pos_idx)
				# pred_t2i_tsr = self.gmm_pred(sim_t2i_tsr, self.gmm_loc, pos_idx)
				pred_i2t, prob_i2t = self.gmm_pred(sim_i2t_norm, self.gmm_glob, pos_idx)
				pred_t2i, prob_t2i = self.gmm_pred(sim_t2i_norm, self.gmm_glob, pos_idx)
				pred_i2t_tsr, prob_i2t_tsr = self.gmm_pred(sim_i2t_tsr_norm, self.gmm_loc, pos_idx)
				pred_t2i_tsr, prob_t2i_tsr = self.gmm_pred(sim_t2i_tsr_norm, self.gmm_loc, pos_idx)

				pred_i2t_cons = pred_i2t + pred_i2t_tsr
				pred_t2i_cons = pred_t2i + pred_t2i_tsr
				# pred_cons[pred_cons == 1] += np.random.randint(0, 2, size=(((pred_cons == 1) + 0).sum(),))
				fn_idx_i2t = pred_i2t_cons == 2
				fn_idx_t2i = pred_t2i_cons == 2

				if self.fnr:  # false negative reverse
					target_i2t = fn_idx_i2t + pos_idx
					target_i2t = target_i2t / target_i2t.sum(1, keepdim=True)
					target_t2i = fn_idx_t2i + pos_idx
					target_t2i = target_t2i / target_t2i.sum(1, keepdim=True)
				else:  # adaptive fne
					if self.adaptive:

						fn_weight_i2t = torch.exp(-prob_i2t)
						fn_weight_i2t = fn_weight_i2t.masked_fill(~fn_idx_i2t, 1)
						fn_weight_i2t_tsr = torch.exp(-prob_i2t_tsr)
						fn_weight_i2t_tsr = fn_weight_i2t_tsr.masked_fill(~fn_idx_i2t, 1)

						fn_weight_t2i = torch.exp(-prob_t2i)
						fn_weight_t2i = fn_weight_t2i.masked_fill(~fn_idx_t2i, 1)
						fn_weight_t2i_tsr = torch.exp(-prob_t2i_tsr)
						fn_weight_t2i_tsr = fn_weight_t2i_tsr.masked_fill(~fn_idx_t2i, 1)

						sim_i2t = sim_i2t * fn_weight_i2t
						sim_t2i = sim_t2i * fn_weight_t2i
						sim_i2t_tsr = sim_i2t_tsr * fn_weight_i2t_tsr
						sim_t2i_tsr = sim_t2i_tsr * fn_weight_t2i_tsr
					else:  # fne
						sim_i2t[fn_idx_i2t] = -1.0e9
						sim_t2i[fn_idx_t2i] = -1.0e9
						sim_i2t_tsr[fn_idx_i2t] = -1.0e9
						sim_t2i_tsr[fn_idx_t2i] = -1.0e9

			if self.fnr:
				loss_i2t = -torch.sum(F.log_softmax(logit_scale * sim_i2t, dim=1) * target_i2t, dim=1).mean()
				loss_t2i = -torch.sum(F.log_softmax(logit_scale * sim_t2i, dim=1) * target_t2i, dim=1).mean()
				loss_i2t_tsr = -torch.sum(F.log_softmax(logit_scale * sim_i2t_tsr, dim=1) * target_i2t, dim=1).mean()
				loss_t2i_tsr = -torch.sum(F.log_softmax(logit_scale * sim_t2i_tsr, dim=1) * target_t2i, dim=1).mean()
			else:  # fne
				loss_i2t = -torch.sum(F.log_softmax(logit_scale * sim_i2t, dim=1) * sim_targets, dim=1).mean()
				loss_t2i = -torch.sum(F.log_softmax(logit_scale * sim_t2i, dim=1) * sim_targets, dim=1).mean()
				loss_i2t_tsr = -torch.sum(F.log_softmax(logit_scale * sim_i2t_tsr, dim=1) * sim_targets, dim=1).mean()
				loss_t2i_tsr = -torch.sum(F.log_softmax(logit_scale * sim_t2i_tsr, dim=1) * sim_targets, dim=1).mean()

			loss_fne = (loss_i2t + loss_t2i) / 2
			loss_fne_tsr = (loss_i2t_tsr + loss_t2i_tsr) / 2

			################## for visu ######################
			# index_i2t = []
			# index_t2i = []
			# for i in range(fn_idx_i2t.shape[0]):
			# 	fn_index = index_all[:, fn_idx_i2t[i]].view(-1).tolist()
			# 	pos_index = index_all[:, pos_idx[i]].view(-1).tolist()
			# 	query_index = index[i].tolist()[0]
			# 	prob = prob_i2t[i, :][fn_idx_i2t[i]].tolist()
			# 	# for true negative visu
			# 	tn_idx = [random.randint(0, fn_idx_i2t.shape[1]) for _ in range(5)]
			# 	tn_prob = prob_i2t[i, tn_idx].tolist()
			# 	fn_index += tn_idx
			# 	prob += tn_prob
			# 	if len(fn_index) > 0:
			# 		index_i2t.append([query_index, pos_index, fn_index, prob])
			#
			# 	prob = prob_t2i[i, :][fn_idx_t2i[i]].tolist()
			# 	# for true negative visu
			# 	tn_prob = prob_i2t[i, tn_idx].tolist()
			# 	fn_index += tn_idx
			# 	prob += tn_prob
			# 	fn_index = index_all[:, fn_idx_t2i[i]].view(-1).tolist()
			# 	if len(fn_index) > 0:
			# 		index_t2i.append([query_index, pos_index, fn_index, prob])
			#
			# ret.update({'visu': [index_i2t, index_t2i]})
			########################################

			ret.update({'itc_loss': loss_fne})
			ret.update({'itc_loss_tsr': loss_fne_tsr})
			ret.update({'fn_count': [fn_idx_i2t.sum().tolist(), fn_idx_t2i.sum().tolist(), 0, 0]})

		else:

			loss_i2t = self.criterion(logit_scale * sim_i2t, sim_targets)
			loss_t2i = self.criterion(logit_scale * sim_t2i, sim_targets)
			loss_ita = (loss_i2t + loss_t2i) / 2

			loss_i2t_tsr = self.criterion(logit_scale * sim_i2t_tsr, sim_targets)
			loss_t2i_tsr = self.criterion(logit_scale * sim_t2i_tsr, sim_targets)
			loss_ita_tsr = (loss_i2t_tsr + loss_t2i_tsr) / 2

			ret.update({'itc_loss': loss_ita})
			ret.update({'itc_loss_tsr': loss_ita_tsr})

		self._dequeue_and_enqueue(i_feats_m, t_feats_m, i_feats_tsr_m, t_feats_tsr_m, idx, index)

		return ret

	def gmm_pred(self, sim, gmm, pos_idx):
		bs = sim.shape[0]
		prob = gmm.predict_proba(sim.view(-1).cpu().numpy().reshape(-1, 1))
		prob = prob[:, gmm.means_.argmax()].reshape(bs, -1)  # prob of false negative
		prob = torch.from_numpy(prob).cuda().masked_fill(pos_idx, 0)
		pred = (prob > self.threshld) + 0
		return pred, prob


def build_model(args, num_classes=11003):
	print("gmm model")
	model = FNDS(args, num_classes)
	# covert model to fp16
	convert_weights(model)
	return model
