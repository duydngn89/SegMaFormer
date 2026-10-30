import torch
import torch.nn.functional as F
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_and_SoftBandDice_loss(nn.Module):
    """Region Dice+BCE with a differentiable morphology-based boundary Dice term."""
    def __init__(
        self,
        bce_kwargs,
        soft_dice_kwargs,
        weight_ce=1,
        weight_dice=1,
        weight_boundary=0.1,
        boundary_dilation=3,
        boundary_smooth=1e-5,
        use_ignore_label: bool = False,
        dice_class=MemoryEfficientSoftDiceLoss,
    ):
        super().__init__()
        self.base_loss = DC_and_BCE_loss(
            bce_kwargs,
            soft_dice_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            use_ignore_label=use_ignore_label,
            dice_class=dice_class,
        )
        self.weight_boundary = float(weight_boundary)
        self.boundary_dilation = int(boundary_dilation)
        if self.boundary_dilation < 1:
            raise ValueError(
                f"boundary_dilation must be >= 1 for soft-band Dice, got {boundary_dilation}."
            )
        self.boundary_smooth = float(boundary_smooth)
        self.use_ignore_label = bool(use_ignore_label)
        self.batch_dice = soft_dice_kwargs.get("batch_dice", False)

    @staticmethod
    def _pool_fn(x: torch.Tensor):
        spatial_dims = x.ndim - 2
        if spatial_dims == 2:
            return F.max_pool2d
        if spatial_dims == 3:
            return F.max_pool3d
        raise ValueError(f"Only 2D and 3D tensors are supported, got shape {tuple(x.shape)}")

    def _soft_boundary_band(self, x: torch.Tensor):
        kernel_size = 2 * self.boundary_dilation + 1
        pool = self._pool_fn(x)
        dilation = pool(
            x,
            kernel_size=kernel_size,
            stride=1,
            padding=self.boundary_dilation,
        )
        erosion = -pool(
            -x,
            kernel_size=kernel_size,
            stride=1,
            padding=self.boundary_dilation,
        )
        return (dilation - erosion).clamp(min=0.0, max=1.0)

    def _target_regions_and_mask(self, target: torch.Tensor):
        if self.use_ignore_label:
            if target.dtype == torch.bool:
                valid_mask = ~target[:, -1:]
            else:
                valid_mask = (1 - target[:, -1:]).bool()
            target_regions = target[:, :-1]
        else:
            target_regions = target
            valid_mask = None
        return target_regions.float(), valid_mask

    def _soft_band_dice_loss(self, net_output: torch.Tensor, target: torch.Tensor):
        target_regions, valid_mask = self._target_regions_and_mask(target)

        # Keep the morphology operations in fp32 under AMP. Max-pooling remains
        # differentiable with respect to the region logits through sigmoid.
        with torch.autocast(device_type=net_output.device.type, enabled=False):
            pred_band = self._soft_boundary_band(torch.sigmoid(net_output.float()))
            target_band = self._soft_boundary_band(target_regions.float())
            if valid_mask is None:
                valid_mask = torch.ones_like(target_band[:, :1])
            else:
                valid_mask = valid_mask.float()

            reduce_axes = tuple(range(2, net_output.ndim))
            intersect = (pred_band * target_band * valid_mask).sum(dim=reduce_axes)
            pred_sum = (pred_band * valid_mask).sum(dim=reduce_axes)
            target_sum = (target_band * valid_mask).sum(dim=reduce_axes)

            if self.batch_dice:
                intersect = intersect.sum(dim=0)
                pred_sum = pred_sum.sum(dim=0)
                target_sum = target_sum.sum(dim=0)

            valid_regions = target_sum > 0
            if not valid_regions.any().item():
                return net_output.new_zeros(())

            dice = (2 * intersect + self.boundary_smooth) / torch.clamp(
                pred_sum + target_sum + self.boundary_smooth,
                min=1e-8,
            )
            return 1 - dice[valid_regions].mean()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        loss = self.base_loss(net_output, target)
        if self.weight_boundary == 0:
            return loss
        return loss + self.weight_boundary * self._soft_band_dice_loss(net_output, target)


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
    

class DC_and_topk_loss_v2(nn.Module):
    def __init__(
            self,
            soft_dice_kwargs,
            ce_kwargs,
            topk_kwargs,
            weight_ce=1,
            weight_dice=1,
            ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss,
            topk_alpha=0.3  
        ):
        """
        Dice + (alpha*TopKCE + (1-alpha)*CE)
        """

        super().__init__()

        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.topk_alpha = topk_alpha

        # normal CE (stabilizes)
        self.ce_normal = RobustCrossEntropyLoss(**ce_kwargs)

        # Top-K CE (focus on hard pixels)
        self.ce_topk = TopKLoss(**topk_kwargs)

        # Dice loss
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        if self.ignore_label is not None:
            assert target.shape[1] == 1, \
                'ignore label is not implemented for one hot encoded target variables (DC_and_CE_loss)'

            mask = (target != self.ignore_label).bool()

            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # Dice
        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0 else 0
        )

        # CE normal
        ce_normal = (
            self.ce_normal(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        # CE top-k
        ce_topk = (
            self.ce_topk(net_output, target)
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        # Mixed CE (this is the important part!)
        ce_loss = (1 - self.topk_alpha) * ce_normal + self.topk_alpha * ce_topk

        # Final loss
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_CE_and_Boundary_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        weight_boundary=0.5,
        ignore_label=None,
        boundary_dilation=3,
        use_class_weights=True,
        boundary_loss_mode="bce",
        boundary_smooth=1e-5,
        boundary_do_bg=None,
    ):
        """
        Inverse-frequency weighted Dice + CE + masked boundary loss.

        Class weights are computed per batch as
            w_c = |Omega| / (N_cls * count_c)
        and applied to Dice, CE and boundary losses. The boundary term is only
        evaluated inside a dilated class-boundary band and normalized to keep
        the scale stable.
        """
        super().__init__()

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_boundary = weight_boundary
        self.ignore_label = ignore_label
        self.boundary_dilation = boundary_dilation
        self.use_class_weights = use_class_weights

        self.batch_dice = soft_dice_kwargs.get("batch_dice", False)
        self.do_bg = soft_dice_kwargs.get("do_bg", True)
        self.smooth = soft_dice_kwargs.get("smooth", 1.0)
        self.boundary_loss_mode = str(boundary_loss_mode).strip().lower()
        if self.boundary_loss_mode not in {"bce", "dice", "soft_band_dice"}:
            raise ValueError(
                f"Unsupported boundary_loss_mode '{boundary_loss_mode}'. "
                "Expected 'bce', 'dice', or 'soft_band_dice'."
            )
        self.boundary_smooth = float(boundary_smooth)
        self.boundary_do_bg = self.do_bg if boundary_do_bg is None else bool(boundary_do_bg)

        # Reserved for compatibility with existing call sites.
        self.label_smoothing = ce_kwargs.get("label_smoothing", 0.0)

    @staticmethod
    def _pool_fn(x: torch.Tensor):
        spatial_dims = x.ndim - 2
        if spatial_dims == 2:
            return F.max_pool2d
        if spatial_dims == 3:
            return F.max_pool3d
        raise ValueError(f"Only 2D and 3D tensors are supported, got shape {tuple(x.shape)}")

    def _prepare_target(self, net_output: torch.Tensor, target: torch.Tensor):
        if target.ndim == net_output.ndim:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_and_Boundary_loss)"
            )
            target = target[:, 0]

        target = target.long()
        if self.ignore_label is not None:
            valid_mask = (target != self.ignore_label).unsqueeze(1)
            target = torch.where(valid_mask[:, 0], target, 0)
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool).unsqueeze(1)

        target_onehot = F.one_hot(target, num_classes=net_output.shape[1]).movedim(-1, 1).to(net_output.dtype)
        valid_mask = valid_mask.to(net_output.dtype)
        return target, target_onehot, valid_mask

    def _compute_class_weights(self, target_onehot: torch.Tensor, valid_mask: torch.Tensor):
        reduce_axes = (0,) + tuple(range(2, target_onehot.ndim))
        class_counts = (target_onehot * valid_mask).sum(dim=reduce_axes)
        present = class_counts > 0

        if self.use_class_weights:
            valid_voxels = valid_mask.sum()
            num_classes = target_onehot.shape[1]
            class_weights = torch.zeros_like(class_counts)
            class_weights[present] = valid_voxels / (num_classes * class_counts[present])
        else:
            class_weights = torch.ones_like(class_counts)
        return class_weights, present

    def _dice_loss(
        self,
        probs: torch.Tensor,
        target_onehot: torch.Tensor,
        valid_mask: torch.Tensor,
        class_weights: torch.Tensor,
    ):
        probs = probs if self.do_bg else probs[:, 1:]
        target_onehot = target_onehot if self.do_bg else target_onehot[:, 1:]
        class_weights = class_weights if self.do_bg else class_weights[1:]

        if class_weights.numel() == 0:
            return probs.new_zeros(())

        reduce_axes = tuple(range(2, probs.ndim))
        intersect = (probs * target_onehot * valid_mask).sum(dim=reduce_axes)
        pred_sum = (probs * valid_mask).sum(dim=reduce_axes)
        target_sum = (target_onehot * valid_mask).sum(dim=reduce_axes)

        if self.batch_dice:
            intersect = intersect.sum(dim=0)
            pred_sum = pred_sum.sum(dim=0)
            target_sum = target_sum.sum(dim=0)
            per_class_dice = (2 * intersect + self.smooth) / torch.clamp(
                pred_sum + target_sum + self.smooth,
                min=1e-8,
            )
            weighted_dice = (per_class_dice * class_weights).sum() / torch.clamp(class_weights.sum(), min=1e-8)
            return 1 - weighted_dice

        per_class_dice = (2 * intersect + self.smooth) / torch.clamp(
            pred_sum + target_sum + self.smooth,
            min=1e-8,
        )
        weighted_dice = (per_class_dice * class_weights.view(1, -1)).sum(dim=1) / torch.clamp(
            class_weights.sum(),
            min=1e-8,
        )
        return 1 - weighted_dice.mean()

    def _ce_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        class_weights: torch.Tensor,
    ):
        log_probs = torch.log_softmax(logits, dim=1)
        target_safe = target.unsqueeze(1)

        if self.label_smoothing > 0:
            target_onehot = F.one_hot(target, num_classes=logits.shape[1]).movedim(-1, 1).to(logits.dtype)
            smoothed_target = (
                (1.0 - self.label_smoothing) * target_onehot
                + self.label_smoothing / logits.shape[1]
            )
            ce_map = -(smoothed_target * log_probs).sum(dim=1, keepdim=True)
        else:
            ce_map = -log_probs.gather(1, target_safe)

        weight_map = class_weights[target].unsqueeze(1)
        weighted_ce = ce_map * weight_map * valid_mask
        return weighted_ce.sum() / torch.clamp(valid_mask.sum(), min=1.0)

    def _boundary_mask(self, target_onehot: torch.Tensor, valid_mask: torch.Tensor):
        pool = self._pool_fn(target_onehot)
        target_float = target_onehot.float()

        local_max = pool(target_float, kernel_size=3, stride=1, padding=1)
        local_min = -pool(-target_float, kernel_size=3, stride=1, padding=1)
        boundary = local_max > local_min

        if self.boundary_dilation > 0:
            kernel_size = 2 * self.boundary_dilation + 1
            boundary = pool(
                boundary.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=self.boundary_dilation,
            ) > 0

        return boundary & valid_mask.bool()

    def _soft_boundary_band(self, x: torch.Tensor):
        """Differentiable dilation-minus-erosion boundary band for 2D or 3D maps."""
        radius = max(int(self.boundary_dilation), 1)
        kernel_size = 2 * radius + 1
        pool = self._pool_fn(x)
        dilation = pool(x, kernel_size=kernel_size, stride=1, padding=radius)
        erosion = -pool(-x, kernel_size=kernel_size, stride=1, padding=radius)
        return (dilation - erosion).clamp(min=0.0, max=1.0)

    def _boundary_loss(
        self,
        probs: torch.Tensor,
        target_onehot: torch.Tensor,
        valid_mask: torch.Tensor,
        class_weights: torch.Tensor,
    ):
        probs = probs if self.boundary_do_bg else probs[:, 1:]
        target_onehot = target_onehot if self.boundary_do_bg else target_onehot[:, 1:]
        class_weights = class_weights if self.boundary_do_bg else class_weights[1:]

        if class_weights.numel() == 0:
            return probs.new_zeros(())

        boundary_mask = self._boundary_mask(target_onehot, valid_mask)
        masked_voxels = boundary_mask.sum()
        if masked_voxels.item() == 0:
            return probs.new_zeros(())

        with torch.autocast(device_type=probs.device.type, enabled=False):
            probs_fp32 = probs.float().clamp(min=1e-6, max=1 - 1e-6)
            target_fp32 = target_onehot.float()
            boundary_mask_fp32 = boundary_mask.float()
            class_weights_fp32 = class_weights.float()

            if self.boundary_loss_mode == "bce":
                # BCE on probabilities is not autocast-safe in PyTorch, so compute the
                # boundary term in fp32 while preserving the surrounding AMP training flow.
                class_weights_map = class_weights_fp32.view(1, -1, *([1] * (probs.ndim - 2)))
                bce_map = F.binary_cross_entropy(probs_fp32, target_fp32, reduction="none")
                weighted_bce = bce_map * class_weights_map * boundary_mask_fp32
                return weighted_bce.sum() / torch.clamp(masked_voxels.float(), min=1.0)

            if self.boundary_loss_mode == "soft_band_dice":
                # The prediction band stays differentiable with respect to the
                # segmentation logits through max-pooling subgradients.
                pred_boundary = self._soft_boundary_band(probs_fp32)
                target_boundary = self._soft_boundary_band(target_fp32)
                loss_mask = valid_mask.float()
            else:
                pred_boundary = probs_fp32
                target_boundary = target_fp32
                loss_mask = boundary_mask_fp32

            reduce_axes = tuple(range(2, probs.ndim))
            intersect = (pred_boundary * target_boundary * loss_mask).sum(dim=reduce_axes)
            pred_sum = (pred_boundary * loss_mask).sum(dim=reduce_axes)
            target_sum = (target_boundary * loss_mask).sum(dim=reduce_axes)

            if self.batch_dice:
                intersect = intersect.sum(dim=0)
                pred_sum = pred_sum.sum(dim=0)
                target_sum = target_sum.sum(dim=0)
                valid_classes = target_sum > 0
                if not valid_classes.any().item():
                    return probs.new_zeros(())
                per_class_dice = (2 * intersect + self.boundary_smooth) / torch.clamp(
                    pred_sum + target_sum + self.boundary_smooth,
                    min=1e-8,
                )
                weights = class_weights_fp32[valid_classes]
                weighted_dice = (
                    per_class_dice[valid_classes] * weights
                ).sum() / torch.clamp(weights.sum(), min=1e-8)
                return 1 - weighted_dice

            per_class_dice = (2 * intersect + self.boundary_smooth) / torch.clamp(
                pred_sum + target_sum + self.boundary_smooth,
                min=1e-8,
            )
            valid_classes = target_sum > 0
            weights = class_weights_fp32.view(1, -1) * valid_classes.float()
            normalizer = weights.sum(dim=1)
            valid_samples = normalizer > 0
            if not valid_samples.any().item():
                return probs.new_zeros(())
            weighted_dice = (per_class_dice * weights).sum(dim=1) / torch.clamp(
                normalizer,
                min=1e-8,
            )
            return 1 - weighted_dice[valid_samples].mean()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        target, target_onehot, valid_mask = self._prepare_target(net_output, target)
        class_weights, present = self._compute_class_weights(target_onehot, valid_mask)

        if not present.any().item():
            return net_output.new_zeros(())

        probs = softmax_helper_dim1(net_output)
        dice_loss = self._dice_loss(probs, target_onehot, valid_mask, class_weights) if self.weight_dice != 0 else 0
        ce_loss = self._ce_loss(net_output, target, valid_mask, class_weights) if self.weight_ce != 0 else 0
        boundary_loss = (
            self._boundary_loss(probs, target_onehot, valid_mask, class_weights)
            if self.weight_boundary != 0
            else 0
        )

        return (
            self.weight_dice * dice_loss
            + self.weight_ce * ce_loss
            + self.weight_boundary * boundary_loss
        )
