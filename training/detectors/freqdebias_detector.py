"""FreqDebias detector (CVPR 2025) for DeepfakeBench.

The paper has no public code and its referenced supplement is unavailable.
Details omitted from the paper are exposed in the detector YAML.
"""

import logging
import math
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from detectors import DETECTOR
from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)


def _strip_prefixes(key, prefixes):
    for prefix in prefixes:
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _selector_key(key):
    """Map both torchvision and DeepfakeBench ResNet-34 checkpoint keys."""
    key = _strip_prefixes(key, ("module.", "backbone.", "resnet."))
    sequential_stages = {
        "0.": "conv1.",
        "1.": "bn1.",
        "4.": "layer1.",
        "5.": "layer2.",
        "6.": "layer3.",
        "7.": "layer4.",
    }
    for prefix, replacement in sequential_stages.items():
        if key.startswith(prefix):
            return replacement + key[len(prefix):]
    if key.startswith("classifier_head."):
        return key.replace("classifier_head.", "fc.", 1)
    return key


class AlignmentBranch(nn.Module):
    """Align a ResNet stage to the final 512-channel resolution."""

    def __init__(self, in_channels, downsample_steps):
        super().__init__()
        layers, channels = [], in_channels
        for _ in range(downsample_steps):
            out_channels = min(channels * 2, 512)
            layers += [
                nn.Conv2d(channels, out_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            ]
            channels = out_channels
        if channels != 512:
            layers += [nn.Conv2d(channels, 512, 1, bias=False),
                       nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x):
        return self.layers(x)


class VMFClassifier(nn.Module):
    """Two-component vMF mixture classifier in stable log-density space."""

    def __init__(self, embedding_dim, num_classes=2, class_priors=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.orientation = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.raw_kappa = nn.Parameter(torch.full((num_classes,), 3.0))
        nn.init.normal_(self.orientation, std=0.02)
        priors = class_priors or [1.0 / num_classes] * num_classes
        self.register_buffer("log_prior", torch.as_tensor(priors).float().log())

    def _log_normalizer(self, kappa):
        # Uniform asymptotic log(I_v(kappa)) expansion; stable and
        # differentiable for the high Bessel order used here.
        v = self.embedding_dim / 2.0 - 1.0
        root = torch.sqrt(kappa.square() + v * v)
        log_iv = (root + v * (torch.log(kappa) - torch.log(v + root))
                  - 0.5 * torch.log(2.0 * math.pi * root))
        return (v * torch.log(kappa)
                - self.embedding_dim / 2.0 * math.log(2.0 * math.pi) - log_iv)

    def forward(self, features):
        features = F.normalize(features, dim=1)
        orientation = F.normalize(self.orientation, dim=1)
        kappa = F.softplus(self.raw_kappa) + 1e-4
        return (features @ orientation.t()) * kappa + self._log_normalizer(kappa) + self.log_prior


@DETECTOR.register_module(module_name="freqdebias")
class FreqDebiasDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = int(config.get("num_classes", 2))
        self.temperature = float(config.get("temperature", 4.0))
        self.eta = float(config.get("eta", 0.5))
        self.delta = float(config.get("delta", 0.1))
        self.mu = float(config.get("mu", 1.0))
        self.rho = float(config.get("rho", 0.1))
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier_head = nn.Linear(512, self.num_classes)
        self.alignments = nn.ModuleList([
            AlignmentBranch(64, 3), AlignmentBranch(128, 2),
            AlignmentBranch(256, 1), AlignmentBranch(512, 0),
        ])
        self.cam_classifier = nn.Linear(512, self.num_classes)
        self.cam_normalizer = nn.InstanceNorm2d(self.num_classes, affine=True)
        embedding_dim = int(config.get("vmf_embedding_dim", 256))
        self.vmf_projection = nn.Conv2d(2048, embedding_dim, 1, bias=False)
        self.vmf_classifier = VMFClassifier(
            embedding_dim, self.num_classes, config.get("class_priors"))

        self.radial_bins = int(config.get("frequency_radial_bins", 8))
        self.angular_bins = int(config.get("frequency_angular_bins", 8))
        self.frequency_clusters = int(config.get("frequency_clusters", 8))
        self.ohem_top_clusters = int(config.get("ohem_top_clusters", 3))
        self.kmeans_iterations = int(config.get("kmeans_iterations", 8))
        self.confidence_ratio = float(config.get("confidence_ratio", 0.5))
        self.amplitude_noise_std = float(config.get("amplitude_noise_std", 0.0))
        self.ohem_chunk_size = int(config.get("ohem_chunk_size", 32))
        self.standard_aug_probability = float(config.get("standard_aug_probability", 0.5))
        self.pixelation_scale = float(config.get("pixelation_scale", 0.5))
        self.selector = self._build_selector(config.get("selector_checkpoint"))
        self._segment_cache = {}

    def build_backbone(self, config):
        try:
            model = torchvision.models.resnet34(weights=None)
        except TypeError:
            model = torchvision.models.resnet34(pretrained=False)
        pretrained = config.get("pretrained")
        if pretrained and os.path.isfile(pretrained):
            state = torch.load(pretrained, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {
                _strip_prefixes(key, ("module.", "backbone.")): value
                for key, value in state.items()
                if not key.endswith("fc.weight") and not key.endswith("fc.bias")
            }
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info("Loaded ResNet-34 weights from %s (missing=%d, unexpected=%d)",
                        pretrained, len(missing), len(unexpected))
        else:
            logger.warning("FreqDebias ResNet-34 has no ImageNet initialization")
        model.fc = nn.Identity()
        return model

    def build_loss(self, config):
        return nn.CrossEntropyLoss()

    def _build_selector(self, checkpoint):
        if not checkpoint:
            logger.warning("selector_checkpoint is unset; OHEM uses detached current weights. "
                           "For the paper protocol, provide a 30-epoch FF++ checkpoint.")
            return None
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError("FreqDebias selector not found: %s" % checkpoint)
        selector = deepcopy(self.backbone)
        selector.fc = nn.Linear(512, self.num_classes)
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        selector_state = selector.state_dict()
        dropped = []
        for key, value in state.items():
            mapped_key = _selector_key(key)
            if mapped_key in selector_state and selector_state[mapped_key].shape == value.shape:
                cleaned[mapped_key] = value
            else:
                dropped.append(mapped_key)
        missing, unexpected = selector.load_state_dict(cleaned, strict=False)
        logger.info("Loaded selector from %s (missing=%d, unexpected=%d, dropped=%d)",
                    checkpoint, len(missing), len(unexpected), len(dropped))
        selector.requires_grad_(False).eval()
        return selector

    def train(self, mode=True):
        super().train(mode)
        if self.selector is not None:
            self.selector.eval()
        return self

    def _backbone_stages(self, image):
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(image))))
        f1 = self.backbone.layer1(x)
        f2 = self.backbone.layer2(f1)
        f3 = self.backbone.layer3(f2)
        f4 = self.backbone.layer4(f3)
        return [f1, f2, f3, f4]

    def features(self, data_dict):
        return self._backbone_stages(data_dict["image"])[-1]

    def classifier(self, features):
        return self.classifier_head(self.avgpool(features).flatten(1))

    def _encode(self, image, auxiliary):
        stages = self._backbone_stages(image)
        pooled = self.avgpool(stages[-1]).flatten(1)
        result = {"features": stages[-1], "feat": pooled,
                  "logits": self.classifier_head(pooled)}
        if auxiliary:
            aligned = [branch(stage) for branch, stage in zip(self.alignments, stages)]
            cam_logits = self.cam_classifier(self.avgpool(aligned[-1]).flatten(1))
            cam_maps = torch.einsum("bchw,nc->bnhw", aligned[-1], self.cam_classifier.weight)
            refined_cam = self._refined_cam(cam_maps)
            sphere_feat = self.avgpool(self.vmf_projection(torch.cat(aligned, dim=1))).flatten(1)
            result.update(cam_logits=cam_logits, refined_cam=refined_cam,
                          sphere_feat=F.normalize(sphere_feat, dim=1),
                          sphere_logits=self.vmf_classifier(sphere_feat))
        return result

    @staticmethod
    def _one_dimensional_kmeans(values, clusters, iterations):
        batch, segments = values.shape
        clusters = min(clusters, segments)
        sorted_values = values.sort(dim=1).values
        positions = torch.linspace(0, segments - 1, clusters, device=values.device).long()
        centers = sorted_values[:, positions]
        for _ in range(iterations):
            assignments = (values.unsqueeze(-1) - centers.unsqueeze(1)).abs().argmin(-1)
            one_hot = F.one_hot(assignments, clusters).to(values.dtype)
            counts = one_hot.sum(1)
            updated = (one_hot * values.unsqueeze(-1)).sum(1) / counts.clamp_min(1.0)
            centers = torch.where(counts > 0, updated, centers)
        return assignments

    def _segment_ids(self, height, width, device):
        key = (height, width, device.type, device.index)
        if key in self._segment_cache:
            return self._segment_cache[key]
        yy = torch.arange(height, device=device, dtype=torch.float32) - height // 2
        xx = torch.arange(width, device=device, dtype=torch.float32) - width // 2
        y, x = torch.meshgrid(yy, xx, indexing="ij")
        radius = torch.sqrt(x.square() + y.square())
        radius = (radius / radius.max().clamp_min(1.0) * self.radial_bins).long()
        radius = radius.clamp_max(self.radial_bins - 1)
        angle = (torch.remainder(torch.atan2(y, x), math.pi) / math.pi * self.angular_bins).long()
        ids = radius * self.angular_bins + angle.clamp_max(self.angular_bins - 1)
        self._segment_cache[key] = ids
        return ids

    def _frequency_cluster_masks(self, amplitude):
        batch, _, height, width = amplitude.shape
        ids = self._segment_ids(height, width, amplitude.device)
        segment_count = self.radial_bins * self.angular_bins
        flat_ids = ids.flatten()
        log_amplitude = torch.log1p(amplitude).mean(1).flatten(1)
        one_hot = F.one_hot(flat_ids, segment_count).to(log_amplitude.dtype)
        means = log_amplitude @ one_hot / one_hot.sum(0).clamp_min(1.0)
        assignments = self._one_dimensional_kmeans(
            means, self.frequency_clusters, self.kmeans_iterations)
        pixel_clusters = assignments.gather(1, flat_ids.expand(batch, -1))
        return F.one_hot(pixel_clusters, self.frequency_clusters).permute(0, 2, 1).reshape(
            batch, self.frequency_clusters, height, width).to(amplitude.dtype)

    def _selector_logits(self, images):
        logits, selector = [], self.selector
        was_training = self.backbone.training
        if selector is None:
            self.backbone.eval()
        with torch.no_grad():
            for chunk in images.split(self.ohem_chunk_size):
                if selector is not None:
                    logits.append(selector(chunk))
                else:
                    feature = self._backbone_stages(chunk)[-1]
                    logits.append(self.classifier_head(self.avgpool(feature).flatten(1)))
        if selector is None and was_training:
            self.backbone.train()
        return torch.cat(logits)

    def _standard_augmentation(self, images):
        if images.numel() == 0:
            return images
        batch = len(images)
        contrast_mask = torch.rand(batch, device=images.device) < self.standard_aug_probability
        contrast = torch.empty(batch, 1, 1, 1, device=images.device).uniform_(0.7, 1.3)
        contrasted = (((images + 1.0) * 0.5 - 0.5) * contrast + 0.5).clamp(0, 1) * 2 - 1
        augmented = torch.where(contrast_mask[:, None, None, None], contrasted, images)
        pixel_mask = torch.rand(batch, device=images.device) < self.standard_aug_probability
        h, w = images.shape[-2:]
        size = (max(1, int(h * self.pixelation_scale)), max(1, int(w * self.pixelation_scale)))
        pixelated = F.interpolate(augmented, size, mode="bilinear", align_corners=False)
        pixelated = F.interpolate(pixelated, (h, w), mode="nearest")
        return torch.where(pixel_mask[:, None, None, None], pixelated, augmented)

    @torch.no_grad()
    def _fo_mixup(self, images, labels):
        synthesized = images.clone()
        fake_indices = torch.nonzero(labels == 1, as_tuple=False).flatten()
        if len(fake_indices) < 2:
            return self._standard_augmentation(synthesized)
        fake = images[fake_indices]
        pixels = ((fake.float() + 1.0) * 0.5).clamp(0, 1)
        spectrum = torch.fft.fftshift(torch.fft.fft2(pixels, dim=(-2, -1)), dim=(-2, -1))
        amplitude, phase = spectrum.abs(), torch.angle(spectrum)
        masks = self._frequency_cluster_masks(amplitude)
        filtered_amp = amplitude[:, None] * masks[:, :, None]
        filtered_spec = torch.polar(filtered_amp, phase[:, None].expand_as(filtered_amp))
        filtered = torch.fft.ifft2(torch.fft.ifftshift(filtered_spec, dim=(-2, -1)),
                                  dim=(-2, -1)).real.clamp(0, 1)
        filtered = filtered.reshape(-1, *pixels.shape[1:]) * 2 - 1
        targets = torch.ones(len(filtered), dtype=torch.long, device=images.device)
        losses = F.cross_entropy(self._selector_logits(filtered), targets, reduction="none")
        losses = losses.reshape(len(fake), self.frequency_clusters)
        top_count = min(self.ohem_top_clusters, self.frequency_clusters)
        top_clusters = losses.topk(top_count, dim=1).indices
        column = torch.randint(top_count, (len(fake), 1), device=images.device)
        chosen = top_clusters.gather(1, column).squeeze(1)
        mask = masks[torch.arange(len(fake), device=images.device), chosen][:, None]
        partner_amp = amplitude[torch.randperm(len(fake), device=images.device)]
        xi = torch.rand(len(fake), 1, 1, 1, device=images.device)
        mixed_amp = amplitude * mask + ((1 - xi) * amplitude + xi * partner_amp) * (1 - mask)
        if self.amplitude_noise_std > 0:
            mixed_amp *= 1 + torch.randn_like(mixed_amp) * self.amplitude_noise_std
        mixed_spec = torch.polar(mixed_amp.clamp_min(0), phase)
        mixed = torch.fft.ifft2(torch.fft.ifftshift(mixed_spec, dim=(-2, -1)),
                               dim=(-2, -1)).real.clamp(0, 1)
        synthesized[fake_indices] = mixed.to(images.dtype) * 2 - 1
        return self._standard_augmentation(synthesized)

    def _confidence_indices(self, logits):
        probability = torch.softmax(logits.detach(), 1).clamp_min(1e-8)
        entropy = -(probability * probability.log()).sum(1)
        keep = max(1, int(math.ceil(len(logits) * self.confidence_ratio)))
        return entropy.topk(keep, largest=False).indices

    def _refined_cam(self, cam_maps):
        batch, classes, height, width = cam_maps.shape
        values = cam_maps.detach().reshape(batch * classes, -1)
        assignments = self._one_dimensional_kmeans(values, 2, self.kmeans_iterations)
        centers = []
        for cluster in range(2):
            membership = assignments == cluster
            centers.append((values * membership).sum(1) / membership.sum(1).clamp_min(1))
        high_cluster = torch.stack(centers, 1).argmax(1)
        mask = assignments == high_cluster[:, None]
        masked = cam_maps * mask.reshape(batch, classes, height, width).to(cam_maps.dtype)
        return self.cam_normalizer(masked)

    def _js_divergence(self, first_logits, second_logits):
        first = torch.softmax(first_logits / self.temperature, -1).clamp_min(1e-8)
        second = torch.softmax(second_logits / self.temperature, -1).clamp_min(1e-8)
        mean = 0.5 * (first + second)
        return 0.5 * ((first * (first.log() - mean.log())).sum(-1)
                      + (second * (second.log() - mean.log())).sum(-1)).mean()

    def forward(self, data_dict, inference=False):
        image = data_dict["image"]
        auxiliary = self.training and not inference
        original = self._encode(image, auxiliary=auxiliary)
        output = {"cls": original["logits"],
                  "prob": torch.softmax(original["logits"], 1)[:, 1],
                  "feat": original["feat"]}
        if auxiliary:
            synthesized = self._encode(self._fo_mixup(image, data_dict["label"]), auxiliary=True)
            output.update(original_aux=original, synthesized_aux=synthesized,
                          confidence_indices=self._confidence_indices(synthesized["logits"]))
        return output

    def get_losses(self, data_dict, pred_dict):
        labels = data_dict["label"]
        # DeepfakeBench records a loss during validation as well. Auxiliary
        # branches are deliberately absent in inference, as required by the
        # paper, so validation reports only the main classifier CE.
        if "original_aux" not in pred_dict:
            classification = self.loss_func(pred_dict["cls"], labels)
            return {"overall": classification, "cls": classification}
        original, synthesized = pred_dict["original_aux"], pred_dict["synthesized_aux"]
        selected = pred_dict["confidence_indices"]
        selected_labels = labels[selected]
        cls_original = self.loss_func(original["logits"], labels)
        cls_synthetic = self.loss_func(synthesized["logits"][selected], selected_labels)
        original_soft = torch.softmax(original["logits"][selected] / self.temperature, 1)
        synthetic_log = torch.log_softmax(synthesized["logits"][selected] / self.temperature, 1)
        prediction_cr = F.kl_div(synthetic_log, original_soft, reduction="batchmean") * self.temperature**2
        loss_cls = cls_original + cls_synthetic + prediction_cr
        loss_cam = 0.5 * (self.loss_func(original["cam_logits"], labels)
                          + self.loss_func(synthesized["cam_logits"][selected], selected_labels))
        original_cam = original["refined_cam"][selected].flatten(2)
        synthetic_cam = synthesized["refined_cam"][selected].flatten(2)
        loss_attention = self._js_divergence(original_cam, synthetic_cam)
        loss_sphere_cls = 0.5 * (self.loss_func(original["sphere_logits"], labels)
                                 + self.loss_func(synthesized["sphere_logits"][selected], selected_labels))
        original_log = torch.log_softmax(original["sphere_logits"][selected], 1)
        synthetic_log = torch.log_softmax(synthesized["sphere_logits"][selected], 1)
        symmetric_kl = 0.5 * (
            F.kl_div(original_log, synthetic_log.exp(), reduction="none").sum(1)
            + F.kl_div(synthetic_log, original_log.exp(), reduction="none").sum(1))
        loss_sphere = (symmetric_kl / (1 + symmetric_kl)).mean()
        overall = (loss_cls + self.eta * loss_cam + self.delta * loss_attention
                   + self.mu * loss_sphere_cls + self.rho * loss_sphere)
        return {"overall": overall, "cls": loss_cls, "cam": loss_cam,
                "attention": loss_attention, "sphere_cls": loss_sphere_cls,
                "sphere": loss_sphere}

    def get_train_metrics(self, data_dict, pred_dict):
        auc, eer, acc, ap = calculate_metrics_for_train(
            data_dict["label"].detach(), pred_dict["cls"].detach())
        return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}
