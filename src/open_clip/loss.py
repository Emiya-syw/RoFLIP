from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            output_dict=False,
    ):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features,
            text_features,
            logit_scale,
            logit_bias=logit_bias,
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text[:len(logits_per_image)], labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss
        
class RoFLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        hardnegative=True,
        dcm_loss_weight=0.2,
        dtm_loss_weight=0.2,
        dense_loss_weight=0.2,
        M=5,
        margin=0.2,
        alpha=0.5
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.prev_num_logits = 0
        self.labels = {}
        self.hardnegative = hardnegative
        self.dcm_loss = dcm_loss_weight != 0
        self.dtm_loss = dtm_loss_weight != 0
        self.dense = dense_loss_weight != 0
        self.dcm_loss_weight = dcm_loss_weight
        self.dtm_loss_weight = dtm_loss_weight
        self.dense_loss_weight = dense_loss_weight
        self.prev_margin = 0.1
        self.mode = "hard"
        self.margin = margin
        self.M = M
        self.alpha = alpha

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        logit_bias=None,
        features_t=None,
        features_s=None,
        resizer=None,
        output_dict=False,
        epoch=0
    ):
        device = image_features.device
        dcm_loss, dtm_loss, dense_loss = 0.0, 0.0, 0.0
        results = {}
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
            if text_features.shape[0] == 10 * image_features.shape[0]:
                caption_types = torch.tensor(([1] * image_features.shape[0] + [2] * image_features.shape[0] + [3] * image_features.shape[0] * 8) * self.world_size)
            else:
                caption_types = torch.tensor(([1] * image_features.shape[0] + [3] * image_features.shape[0]) * self.world_size)
            gt_all_text_features = all_text_features[caption_types == 1]
            dense_all_text_features = all_text_features[caption_types == 2]
            ng_all_text_features = all_text_features[caption_types == 3]
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                if self.hardnegative:
                    all_text_features = torch.cat([gt_all_text_features, ng_all_text_features])
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T
                else:
                    logits_per_image = logit_scale * all_image_features @ gt_all_text_features.T
                logits_per_text = logit_scale * gt_all_text_features @ all_image_features.T
                gt_logits_per_modal = logit_scale * all_image_features @ gt_all_text_features.T
                if self.dcm_loss:
                    ng_logits_per_modal = logit_scale * all_image_features @ ng_all_text_features.T
                    dcm_loss = self.get_dcm_loss(gt_logits_per_modal, ng_logits_per_modal, estimator=self.mode)
                if self.dtm_loss:
                    text_embedding_matrix = logit_scale * gt_all_text_features @ ng_all_text_features.T
                    gt_modal_features = logit_scale * gt_all_text_features @ gt_all_text_features.T
                    dtm_loss = self.get_dtm_loss(gt_logits_per_modal, text_embedding_matrix, gt_modal_features, estimator=self.mode, M=self.M)
                if self.dense:
                    if self.hardnegative:
                        dense_logits_per_image = logit_scale * all_image_features @ torch.cat([dense_all_text_features, ng_all_text_features]).T
                    else:
                        dense_logits_per_image = logit_scale * all_image_features @ dense_all_text_features.T
                    dense_logits_per_text = logit_scale * dense_all_text_features @ all_image_features.T
                    dense_loss = self.get_dense_loss(logits_per_image, dense_logits_per_image)
        else:
            pass

        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        results['contrastive_loss'] = total_loss
        if self.dcm_loss:
            total_loss += dcm_loss * self.dcm_loss_weight
            results['dcm'] = dcm_loss * self.dcm_loss_weight
        if self.dtm_loss:
            total_loss += dtm_loss * self.dtm_loss_weight
            results['dtm'] = dtm_loss * self.dtm_loss_weight
        if self.dense and epoch < 1:
            dense_contrastive_loss = (
                F.cross_entropy(dense_logits_per_image, labels) +
                F.cross_entropy(dense_logits_per_text, labels)
            ) / 2
            total_loss += (self.alpha * dense_contrastive_loss + (1 - self.alpha) * dense_loss) * self.dense_loss_weight
            results['dense'] = dense_loss
        return results if output_dict else total_loss

    def get_dcm_loss(self, gt_logits_per_modal, ng_logits_per_modal, tau_plus=0.1, beta=0.25, temperature=0.9, estimator='hard'):
        bs, N = ng_logits_per_modal.shape
        device = ng_logits_per_modal.device
        pos = torch.diag(gt_logits_per_modal).reshape(-1, 1)
        neg = ng_logits_per_modal
        max_logits = torch.maximum(pos, neg.max(dim=1, keepdim=True)[0])
        pos_exp = torch.exp((pos - max_logits) / temperature)
        neg_exp = torch.exp((neg - max_logits) / temperature)
        if estimator == 'easy':
            Ng = neg_exp.sum(dim=-1, keepdim=True)
        elif estimator == 'hard':
            reweight = (beta * neg_exp) / (neg_exp.mean(dim=-1, keepdim=True) + 1e-12)
            Ng = (-N * tau_plus * pos_exp + (reweight * neg_exp).sum(dim=1, keepdim=True)) / (1 - tau_plus + 1e-12)
        Ng = torch.clamp(Ng, min=torch.exp(torch.tensor(-1.0 / temperature, device=device)))
        loss = -torch.log(pos_exp / (pos_exp + Ng + 1e-12))
        return loss.mean()

    def get_dtm_loss(self, gt_logits_per_modal, embedding_matrix, gt_modal_features, M=5, lam=0.01, tau_plus=0.1, beta=0.25, temperature=0.9, estimator='hard'):
        B, N = embedding_matrix.shape
        pos_text_clean = torch.diag(gt_modal_features).view(B, 1)
        neg_logits_text = embedding_matrix
        diag_image_text = torch.diag(gt_logits_per_modal).view(B, 1)
        noise_scaled = lam * torch.randn(B, M, device=embedding_matrix.device)
        pos_text_noisy = diag_image_text.expand(B, M) + noise_scaled
        pos_text_clean = pos_text_clean / temperature
        pos_text_noisy = pos_text_noisy / temperature
        all_pos_logits = torch.cat([pos_text_clean, pos_text_noisy], dim=1)
        pos_row_max = all_pos_logits.max(dim=1, keepdim=True)[0]
        pos_shifted = all_pos_logits - pos_row_max
        pos_exp = torch.exp(pos_shifted)
        pos_exp_sum = pos_exp.sum(dim=1, keepdim=True)
        neg_row_max = neg_logits_text.max(dim=1, keepdim=True)[0]
        neg_shifted = (neg_logits_text - neg_row_max) / temperature
        neg_exp = torch.exp(neg_shifted)
        N_neg = neg_logits_text.shape[1]
        if estimator == 'easy':
            Neg = neg_exp.sum(dim=1, keepdim=True)
            Neg = torch.clamp(Neg, min=torch.exp(torch.tensor(-1.0 / temperature, device=embedding_matrix.device)))
        else:
            Debiased = (neg_exp.sum(dim=1, keepdim=True) - (N_neg * tau_plus) * pos_exp_sum) / (1.0 - tau_plus + 1e-12)
            Debiased = torch.clamp(Debiased, min=torch.exp(torch.tensor(-1.0 / temperature, device=embedding_matrix.device)))
            neg_mean = neg_exp.mean(dim=1, keepdim=True)
            reweight_neg = (beta * neg_exp) / (neg_mean + 1e-12)
            weighted_neg = (reweight_neg * neg_exp).sum(dim=1, keepdim=True)
            pos_mean = pos_exp.mean(dim=1, keepdim=True)
            reweight_pos = (beta * pos_exp) / (pos_mean + 1e-12)
            weighted_pos = (reweight_pos * pos_exp).sum(dim=1, keepdim=True)
            numerator = weighted_neg - (N_neg * tau_plus) * weighted_pos
            HardNeg = numerator / (1.0 - tau_plus + 1e-12)
            HardNeg = torch.clamp(HardNeg, min=torch.exp(torch.tensor(-1.0 / temperature, device=embedding_matrix.device)))
            Neg = HardNeg
        loss_per_sample = -torch.log(pos_exp_sum / (pos_exp_sum + Neg + 1e-12))
        return loss_per_sample.mean()

    def get_dense_loss(self, logits_per_image, dense_logits_per_image, mode="hard_rank"):
        if mode == "hard_rank":
            sparse_diag = torch.diagonal(logits_per_image, offset=0)
            dense_diag = torch.diagonal(dense_logits_per_image, offset=0)
            margin = self.margin
            target = torch.ones_like(sparse_diag)
            loss = F.margin_ranking_loss(
                input1=dense_diag,
                input2=sparse_diag,
                target=target,
                margin=margin,
                reduction="mean",
            )
        elif mode == "soft_rank":
            sparse_diag = torch.diagonal(logits_per_image, offset=0)
            dense_diag = torch.diagonal(dense_logits_per_image, offset=0)
            di = dense_diag - sparse_diag
            yi = torch.ones_like(di)
            loss = F.soft_margin_loss(input=di, target=yi, reduction='mean')
        return loss


class RoFSigLipLoss(nn.Module):
    def __init__(
        self,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        dist_impl: Optional[str] = None,
        dcm_loss=True,
        dtm_loss=True,
        hardnegative=True,
        dcm_loss_weight=0.2,
        dtm_loss_weight=0.2,
        dense=True,
        dense_loss_weight=0.2
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')
        self.prev_num_logits = 0
        self.labels = {}
        self.dcm_loss = dcm_loss
        self.dtm_loss = dtm_loss
        self.dcm_loss_weight = dcm_loss_weight
        self.dtm_loss_weight = dtm_loss_weight
        self.hardnegative = hardnegative
        self.dense = dense
        self.dense_loss_weight = dense_loss_weight

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias=None, features_t=None, features_s=None, resizer=None, output_dict=False, epoch=0):
        total_loss = 0.0
        dcm_loss, dtm_loss, dense_loss = 0.0, 0.0, 0.0
        results = {}
        if self.world_size > 1:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            if text_features.shape[0] == 10 * image_features.shape[0]:
                caption_types = torch.tensor(([1] * image_features.shape[0] + [2] * image_features.shape[0] + [3] * image_features.shape[0] * 8) * self.world_size)
            else:
                caption_types = torch.tensor(([1] * image_features.shape[0] + [3] * image_features.shape[0]) * self.world_size)
            gt_all_text_features = all_text_features[caption_types == 1]
            dense_all_text_features = all_text_features[caption_types == 2]
            ng_all_text_features = all_text_features[caption_types == 3]
            siglip_loss = self._loss(all_image_features, gt_all_text_features, logit_scale, logit_bias)
            if self.hardnegative:
                B = all_image_features.shape[0]
                for i in range(8):
                    siglip_loss += self._loss(all_image_features, ng_all_text_features[i * B:(i + 1) * B], logit_scale, logit_bias, negative_only=True)
            if self.dcm_loss:
                gt_logits_per_modal = logit_scale * all_image_features @ gt_all_text_features.T
                ng_logits_per_modal = logit_scale * all_image_features @ ng_all_text_features.T
                dcm_loss = self.get_dcm_loss(gt_logits_per_modal, ng_logits_per_modal)
            if self.dtm_loss:
                text_embedding_matrix = logit_scale * gt_all_text_features @ ng_all_text_features.T
                gt_modal_features = logit_scale * gt_all_text_features @ gt_all_text_features.T
                dtm_loss = self.get_dtm_loss(gt_logits_per_modal, text_embedding_matrix, gt_modal_features)
            if self.dense:
                if self.hardnegative:
                    dense_logits_per_image = logit_scale * all_image_features @ torch.cat([dense_all_text_features, ng_all_text_features]).T
                else:
                    dense_logits_per_image = logit_scale * all_image_features @ dense_all_text_features.T
                logits_per_image = logit_scale * all_image_features @ gt_all_text_features.T
                dense_loss = self.get_dense_loss(logits_per_image, dense_logits_per_image)
        else:
            pass
        total_loss += siglip_loss
        results['contrastive_loss'] = total_loss
        if self.dcm_loss:
            total_loss += dcm_loss * self.dcm_loss_weight
            results['dcm'] = dcm_loss
        if self.dtm_loss:
            total_loss += dtm_loss * self.dtm_loss_weight
            results['dtm'] = dtm_loss
        if self.dense:
            dense_contrastive_loss = self._loss(all_image_features, dense_all_text_features, logit_scale, logit_bias)
            total_loss += (dense_contrastive_loss + dense_loss) * self.dense_loss_weight
            results['dense'] = dense_loss
        return results if output_dict else total_loss

    def get_dcm_loss(self, gt_logits_per_modal, ng_logits_per_modal, tau_plus=0.1, beta=0.25, temperature=0.9, estimator='hard'):
        bs, N = ng_logits_per_modal.shape
        device = ng_logits_per_modal.device
        pos = torch.diag(gt_logits_per_modal).reshape(-1, 1)
        neg = ng_logits_per_modal
        max_logits = torch.maximum(pos, neg.max(dim=1, keepdim=True)[0])
        pos_exp = torch.exp((pos - max_logits) / temperature)
        neg_exp = torch.exp((neg - max_logits) / temperature)
        if estimator == 'easy':
            Ng = neg_exp.sum(dim=-1, keepdim=True)
        elif estimator == 'hard':
            reweight = (beta * neg_exp) / (neg_exp.mean(dim=-1, keepdim=True) + 1e-12)
            Ng = (-N * tau_plus * pos_exp + (reweight * neg_exp).sum(dim=1, keepdim=True)) / (1 - tau_plus + 1e-12)
        Ng = torch.clamp(Ng, min=torch.exp(torch.tensor(-1.0 / temperature, device=device)))
        loss = -torch.log(pos_exp / (pos_exp + Ng + 1e-12))
        return loss.mean()

    def get_dtm_loss(self, gt_logits_per_modal, embedding_matrix, gt_modal_features, M=5, lam=0.01, tau_plus=0.1, beta=0.25, temperature=0.9, estimator='hard'):
        B, N = embedding_matrix.shape
        pos_text_clean = torch.diag(gt_modal_features).view(B, 1)
        neg_logits_text = embedding_matrix
        diag_image_text = torch.diag(gt_logits_per_modal).view(B, 1)
        noise_scaled = lam * torch.randn(B, M, device=embedding_matrix.device)
        pos_text_noisy = diag_image_text.expand(B, M) + noise_scaled
        pos_text_clean = pos_text_clean / temperature
        pos_text_noisy = pos_text_noisy / temperature
        all_pos_logits = torch.cat([pos_text_clean, pos_text_noisy], dim=1)
        pos_row_max = all_pos_logits.max(dim=1, keepdim=True)[0]
        pos_shifted = all_pos_logits - pos_row_max
        pos_exp = torch.exp(pos_shifted)
        pos_exp_sum = pos_exp.sum(dim=1, keepdim=True)
        neg_row_max = neg_logits_text.max(dim=1, keepdim=True)[0]
        neg_shifted = (neg_logits_text - neg_row_max) / temperature
        neg_exp = torch.exp(neg_shifted)
        N_neg = neg_logits_text.shape[1]
        if estimator == 'easy':
            Neg = neg_exp.sum(dim=1, keepdim=True)
            Neg = torch.clamp(Neg, min=torch.exp(torch.tensor(-1.0 / temperature, device=embedding_matrix.device)))
        else:
            Debiased = (neg_exp.sum(dim=1, keepdim=True) - (N_neg * tau_plus) * pos_exp_sum) / (1.0 - tau_plus + 1e-12)
            Debiased = torch.clamp(Debiased, min=torch.exp(torch.tensor(-1.0 / temperature, device=embedding_matrix.device)))
            neg_mean = neg_exp.mean(dim=1, keepdim=True)
            reweight_neg = (beta * neg_exp) / (neg_mean + 1e-12)
            weighted_neg = (reweight_neg * neg_exp).sum(dim=1, keepdim=True)
            pos_mean = pos_exp.mean(dim=1, keepdim=True)
            reweight_pos = (beta * pos_exp) / (pos_mean + 1e-12)
            weighted_pos = (reweight_pos * pos_exp).sum(dim=1, keepdim=True)
            numerator = weighted_neg - (N_neg * tau_plus) * weighted_pos
            HardNeg = numerator / (1.0 - tau_plus + 1e-12)
            HardNeg = torch.clamp(HardNeg, min=torch.exp(torch.tensor(-1.0 / temperature, device=embedding_matrix.device)))
            Neg = HardNeg
        loss_per_sample = -torch.log(pos_exp_sum / (pos_exp_sum + Neg + 1e-12))
        return loss_per_sample.mean()

    def get_dense_loss(self, logits_per_image, dense_logits_per_image, mode="hard_rank"):
        if mode == "hard_rank":
            sparse_diag = torch.diagonal(logits_per_image, offset=0)
            dense_diag = torch.diagonal(dense_logits_per_image, offset=0)
            margin = 0.2
            target = torch.ones_like(sparse_diag)
            loss = F.margin_ranking_loss(
                input1=dense_diag,
                input2=sparse_diag,
                target=target,
                margin=margin,
                reduction="mean",
            )
        return loss
