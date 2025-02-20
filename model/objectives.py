import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import logging


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
	"""
    Similarity Distribution Matching
    """
	batch_size = image_fetures.shape[0]
	pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
	pid_dist = pid - pid.t()
	labels = (pid_dist == 0).float()

	if image_id != None:
		# print("Mix PID and ImageID to create soft label.")
		image_id = image_id.reshape((-1, 1))
		image_id_dist = image_id - image_id.t()
		image_id_mask = (image_id_dist == 0).float()
		labels = (labels - image_id_mask) * factor + image_id_mask
		# labels = (labels + image_id_mask) / 2

	image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
	text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

	t2i_cosine_theta = text_norm @ image_norm.t()
	i2t_cosine_theta = t2i_cosine_theta.t()

	text_proj_image = logit_scale * t2i_cosine_theta
	image_proj_text = logit_scale * i2t_cosine_theta

	# normalize the true matching distribution
	labels_distribute = labels / labels.sum(dim=1)

	i2t_pred = F.softmax(image_proj_text, dim=1)
	i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
	t2i_pred = F.softmax(text_proj_image, dim=1)
	t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

	loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

	return loss


def FalseNegativeDetection(i_feats, t_feats, sims_i2t, sims_t2i, mask, avg_sims_tool_pos, var_sims_tool_pos,
                           avg_sims_tool_neg, var_sims_tool_neg, nums_right_sims, match_para, reversed_p,
						   index, index_all):
	
	average_sims_scores(i_feats, t_feats, avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg)

	prob_i2t = false_negative_prob(sims_i2t.detach(), avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg,
	                               var_sims_tool_neg, nums_right_sims, match_para)
	prob_t2i = false_negative_prob(sims_t2i.detach(), avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg,
	                               var_sims_tool_neg, nums_right_sims, match_para)

	prob_i2t = prob_i2t.masked_fill(mask.detach(), -1.0)
	prob_t2i = prob_t2i.masked_fill(mask.detach(), -1.0)

	fn_mask_i2t = prob_i2t > reversed_p
	fn_mask_t2i = prob_t2i > reversed_p

	################## for visu ######################
	index_i2t = []
	index_t2i = []

	# fn_mask_i2t_visu = prob_i2t > reversed_p or prob_i2t < 0.1
	# fn_mask_t2i_visu = prob_t2i > reversed_p or prob_t2i < 0.1
	# for i in range(fn_mask_t2i_visu.shape[0]):
	# 	fn_index = index_all[:, fn_mask_i2t_visu[i]].view(-1).tolist()
	# 	pos_index = index_all[:, mask[i]].view(-1).tolist()
	# 	query_index = index[i].tolist()[0]
	# 	prob = prob_i2t[i, :][fn_mask_i2t_visu[i]].tolist()
	# 	if len(fn_index) > 0:
	# 		index_i2t.append([query_index, pos_index, fn_index, prob])

	# 	prob = prob_t2i[i, :][fn_mask_t2i_visu[i]].tolist()
	# 	fn_index = index_all[:, fn_mask_t2i_visu[i]].view(-1).tolist()
	# 	if len(fn_index) > 0:
	# 		index_t2i.append([query_index, pos_index, fn_index, prob])
	########################################

	return fn_mask_i2t, fn_mask_t2i, prob_i2t, prob_t2i, index_i2t, index_t2i
	# return fn_mask_i2t, fn_mask_t2i, prob_i2t, prob_t2i


def FalseNegativeReverse(i_feats, t_feats, sims_i2t, sims_t2i, mask, avg_sims_tool_pos, var_sims_tool_pos,
                         avg_sims_tool_neg, var_sims_tool_neg, nums_right_sims, match_para, reversed_p):
		
	average_sims_scores(i_feats, t_feats, avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg)
	prob_i2t = false_negative_prob(sims_i2t.detach(), avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg,
	                               var_sims_tool_neg, nums_right_sims, match_para)
	prob_t2i = false_negative_prob(sims_t2i.detach(), avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg,
	                               var_sims_tool_neg, nums_right_sims, match_para)

	prob_i2t = prob_i2t.masked_fill_(mask.detach(), -1)
	prob_t2i = prob_t2i.masked_fill_(mask.detach(), -1)

	fn_mask_i2t = prob_i2t > reversed_p
	fn_mask_t2i = prob_t2i > reversed_p

	fnr_i2t_target = fn_mask_i2t + mask.detach()
	fnr_i2t_target = fnr_i2t_target / fnr_i2t_target.sum(1, keepdim=True)
	fnr_t2i_target = fn_mask_t2i + mask.detach()
	fnr_t2i_target = fnr_t2i_target / fnr_t2i_target.sum(1, keepdim=True)

	return fnr_i2t_target, fnr_t2i_target


def false_negative_prob(sims, avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg,
                        var_sims_tool_neg, nums_right_sims, match_para):
	prob = torch.zeros_like(sims)  # 初始化为0
	if avg_sims_tool_pos.count > nums_right_sims:
		mean_data_pos = avg_sims_tool_pos.avg
		var_data_pos = var_sims_tool_pos.var

		mean_data_neg = avg_sims_tool_neg.avg
		var_data_neg = var_sims_tool_neg.var

		pos_p = normal_scores(mean_data_pos, var_data_pos, sims)
		neg_p = normal_scores(mean_data_neg, var_data_neg, sims)

		prob = (pos_p / (pos_p + (match_para - 1) * neg_p))  # false negative probability

	return prob


def HardNegativeLoss(im, s, sims_i2t, sims_t2i, mask, epoch, avg_sims_tool_pos, var_sims_tool_pos,
                     avg_sims_tool_neg, var_sims_tool_neg, nums_right_sims, match_para, false_margin, false_num, logit_scale):
	diagonal_1 = sims_i2t.diag().view(sims_i2t.size(0), 1)
	d1 = diagonal_1.expand_as(sims_i2t)  # positive

	diagonal_2 = sims_t2i.diag().view(sims_t2i.size(0), 1)
	d2 = diagonal_2.expand_as(sims_t2i)

	# temperature = 0.07
	# compare every diagonal score to scores in its column
	# caption retrieval
	# cost_s = (margin + sims_i2t - d1).clamp(min=0)
	# # compare every diagonal score to scores in its row
	# # image retrieval
	# cost_im = (margin + sims_t2i - d2).clamp(min=0)
	#
	# # clear diagonals
	#
	# cost_s = cost_s.masked_fill_(mask, 0)  # mask: pos_idx
	# cost_im = cost_im.masked_fill_(mask, 0)

	# keep the maximum violating negative for each query

	# if epoch >= 1:

	###
	# cost_s_index = cost_s.max(1)[1]
	# cost_im_index = cost_im.max(1)[1]
	###

	average_sims_scores(im, s, avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg)

	index_s, match_p_i2t = choice_negative(d1.detach(), sims_i2t.detach(), mask.detach(), avg_sims_tool_pos,
	                                       var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg, nums_right_sims,
	                                       match_para, false_margin, false_num)
	index_im, match_p_t2i = choice_negative(d2.detach(), sims_t2i.detach(), mask.detach(), avg_sims_tool_pos,
	                                        var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg,
	                                        nums_right_sims, match_para, false_margin, false_num)
	#
	sims_i2t_fne = sims_i2t.gather(1, index_s)
	sims_t2i_fne = sims_t2i.gather(1, index_im)

	loss_t_2_i = F.cross_entropy(sims_t2i_fne * logit_scale, torch.zeros(sims_t2i_fne.shape[0]).to(torch.int64).cuda(),
	                             # positive 放在第一个位置
	                             reduction='mean')
	loss_i_2_t = F.cross_entropy(sims_i2t_fne * logit_scale, torch.zeros(sims_i2t_fne.shape[0]).to(torch.int64).cuda(),
	                             reduction='mean')
	
	#     cost_s = cost_s.gather(1, index_s)
	#     cost_im = cost_im.gather(1, index_im)
	# else:
	#     cost_s = cost_s.max(1)[0]
	#     cost_im = cost_im.max(1)[0]

	return (loss_t_2_i + loss_i_2_t) / 2, match_p_i2t, match_p_t2i


def choice_negative(pos_sims, sims, mask, avg_sims_tool_pos, var_sims_tool_pos, avg_sims_tool_neg, var_sims_tool_neg,
                    nums_right_sims, match_para, false_margin, false_num):
	# pos_sims [batch_size, 1]
	# sims [batch_size, memory_size]
	tau = 0.5
	# match_para = 10000
	# false_margin = 0.999
	num_sampled = sims.shape[1] - false_num

	neg_scores = torch.exp(-(sims - pos_sims) ** 2 * tau)
	# neg_scores = torch.ones_like(sims)
	# neg_scores = torch.zeros_like(sims)  # 初始化为0
	match_p_tmp = torch.zeros_like(sims)
	if avg_sims_tool_pos.count > nums_right_sims:
		mean_data_pos = avg_sims_tool_pos.avg
		var_data_pos = var_sims_tool_pos.var

		mean_data_neg = avg_sims_tool_neg.avg
		var_data_neg = var_sims_tool_neg.var

		pos_p = normal_scores(mean_data_pos, var_data_pos, sims)
		neg_p = normal_scores(mean_data_neg, var_data_neg, sims)

		match_p = (pos_p / (pos_p + (match_para - 1) * neg_p))
		# match_p_tmp = match_p
		# match_p_tmp = match_p_tmp.masked_fill_(mask, -1)
		# tmp = match_p_tmp[match_p_tmp>0.6].tolist()
		# if len(tmp) > 8:
		# 	print(tmp)	
		# 	pdb.set_trace()

		no_match_p = 1 - match_p

		match_p = torch.exp(-(match_p) * tau)

		###
		match_mask = no_match_p < false_margin  # 0.999  极不可能为false negative，easy negative

		neg_scores[match_mask] = match_p[match_mask]
	###

	neg_scores = neg_scores.masked_fill_(mask, 0)  # mask, 过滤掉positive
	# match_p_tmp = match_p_tmp.masked_fill_(mask, -1)

	# # margin_mask = pos_sims - margin
	# margin_mask = (sims <= (avg_sims_tool_pos.avg - margin))
	# neg_scores = neg_scores.masked_fill_(margin_mask, 0)
	# neg_scores_sum = neg_scores.sum(1)

	negative_idx = torch.multinomial(neg_scores, num_sampled)  # replacement=True 有放回采用

	negative_idx = torch.cat([torch.arange(negative_idx.shape[0]).view(-1, 1).cuda(), negative_idx],
	                         dim=1)  # positive放在第一个位置

	return negative_idx, match_p_tmp


def average_sims_scores(img, txt, avgsims_pos, varsims_pos, avgsims_neg, varsims_neg):
	scores = img @ txt.T
	scores_t = scores.T

	match_index_i2t = scores.sort(1, descending=True)[1]
	match_index_t2i = scores_t.sort(1, descending=True)[1]

	# clear diagonals
	mask = torch.eye(scores.size(0)) > .5
	I = mask.cuda()
	cost_s = scores.masked_fill(I, 0)
	cost_im = scores_t.masked_fill(I, 0)

	cost_s = cost_s.max(1)[0]
	cost_im = cost_im.max(1)[0]

	for i in range(scores.size(0)):
		# 保证positive位置的相似度是最大的
		if match_index_i2t[i][0].item() == i and match_index_t2i[i][0].item() == i:
			avgsims_pos.update(scores[i, i])
			###
			avgsims_neg.update(cost_s[i])
			avgsims_neg.update(cost_im[i])
		###

	for i in range(scores.size(0)):
		if match_index_i2t[i][0].item() == i and match_index_t2i[i][0].item() == i:
			varsims_pos.update(scores[i, i], avgsims_pos.avg)
			###
			varsims_neg.update(cost_s[i], avgsims_neg.avg)
			varsims_neg.update(cost_im[i], avgsims_neg.avg)
		###

def normal_scores(mean, var, x):
	normal_dist = torch.distributions.Normal(mean, torch.sqrt(var))
	output_y = normal_dist.log_prob(x).exp()

	return output_y