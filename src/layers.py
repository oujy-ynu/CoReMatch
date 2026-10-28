"""Shared-anchor representation and cross-graph comparison layers for CoReMatch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


def _entmax15_forward(inputs, dim=-1):
    """Closed-form 1.5-entmax forward probability projection."""
    if inputs.size(dim) == 0:
        raise ValueError("entmax15 cannot normalize an empty dimension.")

    # The closed-form thresholding algorithm for alpha=1.5 operates on x / 2.
    values = inputs / 2.0
    values = values - values.max(dim=dim, keepdim=True).values
    sorted_values, _ = torch.sort(values, dim=dim, descending=True)

    rank_shape = [1] * inputs.dim()
    rank_shape[dim] = inputs.size(dim)
    ranks = torch.arange(
        1,
        inputs.size(dim) + 1,
        device=inputs.device,
        dtype=inputs.dtype,
    ).view(rank_shape)

    cumulative_mean = sorted_values.cumsum(dim) / ranks
    cumulative_square_mean = sorted_values.square().cumsum(dim) / ranks
    variance_sum = ranks * (
        cumulative_square_mean - cumulative_mean.square()
    )
    threshold_delta = ((1.0 - variance_sum) / ranks).clamp_min(0.0)
    thresholds = cumulative_mean - torch.sqrt(threshold_delta)

    support_size = (thresholds <= sorted_values).sum(
        dim=dim,
        keepdim=True,
    ).clamp_min(1)
    threshold = thresholds.gather(dim, support_size - 1)
    probabilities = (values - threshold).clamp_min(0.0).square()

    # Correct floating-point drift from the probability simplex.
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(
        torch.finfo(probabilities.dtype).eps
    )


class _Entmax15Function(torch.autograd.Function):
    """1.5-entmax with its analytical Jacobian-vector product."""

    @staticmethod
    def forward(ctx, inputs, dim):
        probabilities = _entmax15_forward(inputs, dim=dim)
        ctx.dim = int(dim)
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx, grad_output):
        (probabilities,) = ctx.saved_tensors

        # The active-support derivative is sqrt(p), subject to the simplex constraint.
        support_derivative = torch.sqrt(probabilities)
        grad_input = grad_output * support_derivative
        correction = grad_input.sum(
            dim=ctx.dim,
            keepdim=True,
        ) / support_derivative.sum(
            dim=ctx.dim,
            keepdim=True,
        )
        grad_input = grad_input - correction * support_derivative
        return grad_input, None


def entmax15(inputs, dim=-1):
    """Sparse 1.5-entmax with analytical gradients."""
    return _Entmax15Function.apply(inputs, dim)


class SharedAnchorSparseAggregation(nn.Module):
    """Shared-Anchor Sparse Aggregation of node values into shared anchor slots.

    Shared anchors provide queries; nodes provide keys and values. Entmax-1.5
    assigns sparse aggregation weights. Linear projections map queries, keys,
    values, and aggregated node information into the anchor feature space.
    """

    def __init__(
        self,
        embed_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.dropout = float(dropout)

        if self.embed_dim <= 0:
            raise ValueError("Sparse-attention dimensions must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Sparse-attention dropout must be in [0, 1).")

        self.scale = self.embed_dim ** -0.5
        self.query_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.output_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for projection in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
        ):
            nn.init.xavier_uniform_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        *,
        projected_query=None,
        key_padding_mask_verified=False,
    ):
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("Sparse attention expects batch-first 3D tensors.")
        if key.shape[:2] != value.shape[:2]:
            raise ValueError("Sparse-attention key/value shapes must align.")
        if (
            query.size(-1) != self.embed_dim
            or key.size(-1) != self.embed_dim
            or value.size(-1) != self.embed_dim
        ):
            raise ValueError("Sparse-attention embedding dimensions do not match.")

        if projected_query is None:
            projected_query = self.query_projection(query)
        elif (projected_query.shape != query.shape
              or projected_query.device != query.device
              or projected_query.dtype != query.dtype):
            raise ValueError("Precomputed anchor queries must match query shape, device and dtype.")
        projected_key = self.key_projection(key)
        projected_value = self.value_projection(value)
        aggregation_scores = torch.matmul(
            projected_query,
            projected_key.transpose(-1, -2),
        ) * self.scale

        expanded_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != key.shape[:2]:
                raise ValueError(
                    "key_padding_mask must have shape [batch, key_length]."
                )
            expanded_mask = key_padding_mask.to(torch.bool)[:, None, :]
            # The model's CPU input preparation checks nonempty graph batches.
            # Keep validation for standalone calls and unverified input paths.
            if not key_padding_mask_verified and torch.any(key_padding_mask.all(dim=-1)):
                raise ValueError(
                    "Every sparse-attention sample needs at least one valid key."
                )

        # A finite sentinel avoids inf-inf inside the entmax threshold.
        if expanded_mask is not None:
            aggregation_scores = aggregation_scores.masked_fill(
                expanded_mask,
                -1.0e4,
            )
        attention = entmax15(
            aggregation_scores.float(),
            dim=-1,
        ).to(aggregation_scores.dtype)

        if expanded_mask is not None:
            attention = attention.masked_fill(expanded_mask, 0.0)
            attention = attention / attention.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(torch.finfo(attention.dtype).eps)

        dropped_attention = F.dropout(
            attention,
            p=self.dropout,
            training=self.training,
        )
        context = torch.matmul(
            dropped_attention,
            projected_value,
        )
        output = self.output_projection(context)

        return output, attention


class SharedAnchorRepresentationBlock(nn.Module):
    """Construct shared-anchor representations with ASA and SASA at one GIN scale."""

    def __init__(
        self,
        input_dim,
        anchor_count,
        anchor_dim,
        num_heads,
        ffn_ratio=2,
        dropout=0.0,
        layer_scale_init=0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.anchor_count = int(anchor_count)
        self.anchor_dim = int(anchor_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        ffn_ratio = int(ffn_ratio)

        if self.input_dim <= 0 or self.anchor_count <= 0 or self.anchor_dim <= 0:
            raise ValueError("Anchor dimensions and anchor_count must be positive.")
        if self.num_heads <= 0 or self.anchor_dim % self.num_heads != 0:
            raise ValueError("anchor_dim must be divisible by a positive attention_heads.")
        if ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be positive.")
        if self.dropout < 0.0:
            raise ValueError("attention_dropout must be non-negative.")

        ffn_dim = self.anchor_dim * ffn_ratio
        self.input_projection = nn.Linear(self.input_dim, self.anchor_dim)
        self.input_norm = nn.LayerNorm(self.anchor_dim)
        self.node_encoder = nn.TransformerEncoderLayer(
            d_model=self.anchor_dim,
            nhead=self.num_heads,
            dim_feedforward=ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Shared anchors followed by Anchor Self-Attention (ASA).
        self.shared_anchors = nn.Parameter(
            torch.empty(1, self.anchor_count, self.anchor_dim)
        )
        nn.init.xavier_uniform_(self.shared_anchors)
        self.anchor_self_attention_input_norm = nn.LayerNorm(self.anchor_dim)
        self.anchor_self_attention = nn.MultiheadAttention(
            embed_dim=self.anchor_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.anchor_self_attention_output_norm = nn.LayerNorm(self.anchor_dim)

        # Shared-Anchor Sparse Aggregation.
        self.shared_anchor_sparse_aggregation = SharedAnchorSparseAggregation(
            embed_dim=self.anchor_dim,
            dropout=self.dropout,
        )
        self.anchor_ffn_norm = nn.LayerNorm(self.anchor_dim)
        self.anchor_ffn = nn.Sequential(
            nn.Linear(self.anchor_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(ffn_dim, self.anchor_dim),
            nn.Dropout(self.dropout),
        )
        self.output_norm = nn.LayerNorm(self.anchor_dim)
        self.anchor_self_attention_scale = nn.Parameter(torch.tensor(float(layer_scale_init)))
        self.anchor_ffn_scale = nn.Parameter(
            torch.tensor(float(layer_scale_init))
        )

        # Cache updated shared anchors and projected queries for inference.
        # Non-persistent buffers are excluded from the model state dictionary.
        self.register_buffer("_cached_updated_anchors", None, persistent=False)
        self.register_buffer("_cached_anchor_query", None, persistent=False)
        self._anchor_cache_signature = None
        self._anchor_cache_batch_sizes = []

    def clear_parameter_state(self):
        self._cached_updated_anchors = None
        self._cached_anchor_query = None
        self._anchor_cache_signature = None
        self._anchor_cache_batch_sizes = []

    def train(self, mode=True):
        self.clear_parameter_state()
        return super().train(mode)

    def _apply(self, fn):
        self.clear_parameter_state()
        return super()._apply(fn)

    def _update_shared_anchors(self, batch_size):
        anchors = self.shared_anchors.expand(batch_size, -1, -1)
        normalized_anchors = self.anchor_self_attention_input_norm(anchors)
        update, _ = self.anchor_self_attention(
            normalized_anchors, normalized_anchors, normalized_anchors,
            need_weights=False,
        )
        return self.anchor_self_attention_output_norm(
            anchors + self.anchor_self_attention_scale * update
        )

    def _constant_anchor_query(self, batch_size):
        parameters = (
            self.shared_anchors,
            self.anchor_self_attention_scale,
            *self.anchor_self_attention_input_norm.parameters(),
            *self.anchor_self_attention.parameters(),
            *self.anchor_self_attention_output_norm.parameters(),
            *self.shared_anchor_sparse_aggregation.query_projection.parameters(),
        )
        # Detect in-place parameter updates and replaced Parameters as well as
        # load_state_dict/train/device/dtype changes handled by the hooks below.
        signature = tuple((id(parameter), parameter._version) for parameter in parameters)
        if self._cached_anchor_query is None or signature != self._anchor_cache_signature:
            self.clear_parameter_state()
            self._anchor_cache_signature = signature
        if batch_size not in self._anchor_cache_batch_sizes:
            # Use the requested batch shape to avoid FP32 rounding changes
            # from computing B=1 and broadcasting the result.
            # Shared anchors require only one cached row per batch size.
            anchors = self._update_shared_anchors(batch_size)
            query = self.shared_anchor_sparse_aggregation.query_projection(anchors)
            anchors, query = anchors[:1].detach().clone(), query[:1].detach().clone()
            if self._cached_anchor_query is None or len(self._anchor_cache_batch_sizes) >= 8:
                self._cached_updated_anchors, self._cached_anchor_query = anchors, query
                self._anchor_cache_batch_sizes = [batch_size]
            else:
                self._cached_updated_anchors = torch.cat((self._cached_updated_anchors, anchors), dim=0)
                self._cached_anchor_query = torch.cat((self._cached_anchor_query, query), dim=0)
                self._anchor_cache_batch_sizes.append(batch_size)
        index = self._anchor_cache_batch_sizes.index(batch_size)
        return (
            self._cached_updated_anchors[index:index+1].expand(batch_size, -1, -1),
            self._cached_anchor_query[index:index+1].expand(batch_size, -1, -1),
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.clear_parameter_state()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, node_features, batch_index, *, nonempty_graphs_verified=False, dense_layout=None):
        if node_features.dim() != 2 or node_features.size(1) != self.input_dim:
            raise ValueError(
                "Shared anchor representation block expected node features [N, {}], got {}.".format(
                    self.input_dim,
                    tuple(node_features.shape),
                )
            )
        if batch_index.numel() == 0:
            raise ValueError("Shared anchor representation block received a batch with no nodes.")

        projected = self.input_norm(self.input_projection(node_features))
        if dense_layout is None:
            dense_nodes, node_mask = to_dense_batch(projected, batch_index)
        else:
            flat_index, node_mask, batch_size, max_nodes = dense_layout
            dense_nodes = projected.new_zeros((batch_size * max_nodes, self.anchor_dim))
            dense_nodes.index_copy_(0, flat_index, projected)
            dense_nodes = dense_nodes.view(batch_size, max_nodes, self.anchor_dim)
        nodes = self.node_encoder(
            dense_nodes,
            src_key_padding_mask=~node_mask,
        )

        use_constants = (
            not self.training
            and not torch.is_grad_enabled() and not torch.is_autocast_enabled()
            and not torch.is_autocast_cpu_enabled()
        )
        if use_constants:
            anchors, projected_query = self._constant_anchor_query(dense_nodes.size(0))
        else:
            anchors = self._update_shared_anchors(dense_nodes.size(0))
            projected_query = None

        # Shared-Anchor Sparse Aggregation using the updated shared anchors.
        anchor_representations, _ = self.shared_anchor_sparse_aggregation(
            query=anchors,
            key=nodes,
            value=nodes,
            key_padding_mask=~node_mask,
            projected_query=projected_query,
            key_padding_mask_verified=bool(use_constants and nonempty_graphs_verified),
        )
        anchor_representations = anchor_representations + self.anchor_ffn_scale * self.anchor_ffn(
            self.anchor_ffn_norm(anchor_representations)
        )
        anchor_representations = F.normalize(self.output_norm(anchor_representations), p=2, dim=-1)
        return anchor_representations


class SharedAnchorRepresentation(nn.Module):
    """Shared Anchor Representation (SAR) at every GIN scale."""

    def __init__(
        self,
        input_dims,
        anchor_count,
        anchor_dim,
        num_heads,
        ffn_ratio=2,
        dropout=0.0,
        layer_scale_init=0.1,
    ):
        super().__init__()
        input_dims = [int(dim) for dim in input_dims]
        self.anchor_count = int(anchor_count)
        anchor_dim = int(anchor_dim)
        self.anchor_dim = anchor_dim
        self.scale_blocks = nn.ModuleList(
            [
                SharedAnchorRepresentationBlock(
                    input_dim=input_dim,
                    anchor_count=self.anchor_count,
                    anchor_dim=anchor_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                )
                for input_dim in input_dims
            ]
        )

    def forward(self, layer_features, batch_index, *, nonempty_graphs_verified=False, dense_layout=None):
        if len(layer_features) != len(self.scale_blocks):
            raise ValueError(
                "Expected {} GNN feature scales, got {}.".format(
                    len(self.scale_blocks), len(layer_features)
                )
            )
        return [
            block(node_features, batch_index, nonempty_graphs_verified=nonempty_graphs_verified,
                  dense_layout=dense_layout)
            for block, node_features in zip(self.scale_blocks, layer_features)
        ]


class ScaleWiseAttentionPooling(nn.Module):
    """Pool anchors with a learned graph context at each structural scale."""

    def __init__(
        self,
        anchor_dim,
        num_scales,
        anchor_count,
        pool_dim=32,
    ):
        super().__init__()
        self.anchor_dim = int(anchor_dim)
        self.num_scales = int(num_scales)
        self.anchor_count = int(anchor_count)
        self.pool_dim = int(pool_dim)
        if min(
            self.anchor_dim,
            self.num_scales,
            self.anchor_count,
            self.pool_dim,
        ) <= 0:
            raise ValueError("Graph-level pooling dimensions must be positive.")

        self.anchor_projection = nn.Linear(self.anchor_dim, self.pool_dim)
        self.scale_fusion = nn.Sequential(
            nn.Linear(self.num_scales * self.pool_dim, self.pool_dim),
            nn.GELU(),
            nn.LayerNorm(self.pool_dim),
        )
        self.context_weights = nn.ParameterList([
            nn.Parameter(torch.empty(self.pool_dim, self.pool_dim))
            for _ in range(self.num_scales)
        ])
        for weight in self.context_weights:
            nn.init.xavier_uniform_(weight)
        self.register_buffer('_packed_context_weights', None, persistent=False)
        self._context_signature = None

    def clear_parameter_state(self):
        self._packed_context_weights = None
        self._context_signature = None

    def train(self, mode=True):
        self.clear_parameter_state()
        return super().train(mode)

    def _apply(self, fn):
        self.clear_parameter_state()
        return super()._apply(fn)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.clear_parameter_state()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def project_anchor_grid(self, anchor_scales):
        if len(anchor_scales) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} anchor scales, got {len(anchor_scales)}."
            )
        expected_tail = (self.anchor_count, self.anchor_dim)
        batch_size = anchor_scales[0].size(0)
        for anchors in anchor_scales:
            if anchors.dim() != 3 or tuple(anchors.shape[1:]) != expected_tail:
                raise ValueError(
                    "Scale-wise pooling expected each scale as [B, {}, {}], got {}.".format(
                        self.anchor_count,
                        self.anchor_dim,
                        tuple(anchors.shape),
                    )
                )
            if anchors.size(0) != batch_size:
                raise ValueError("All anchor scales must have one batch size.")
        return self.anchor_projection(torch.stack(anchor_scales, dim=1))

    def forward(self, anchor_scales):
        grid = self.project_anchor_grid(anchor_scales)
        if not self.training and not torch.is_grad_enabled():
            signature = tuple((id(p), p._version) for p in self.context_weights)
            if self._packed_context_weights is None or signature != self._context_signature:
                self._packed_context_weights = torch.stack(list(self.context_weights)).detach()
                self._context_signature = signature
            means = grid.mean(dim=2).transpose(0, 1)
            contexts = torch.tanh(torch.bmm(means, self._packed_context_weights)).transpose(0, 1)
            coefficients = torch.sigmoid((grid * contexts[:, :, None, :]).sum(dim=-1))
            pooled = (coefficients[..., None] * grid).sum(dim=2)
            return self.scale_fusion(pooled.flatten(start_dim=1))
        pooled_scales = []
        for scale_index, weight in enumerate(self.context_weights):
            anchors = grid[:, scale_index]
            context = torch.tanh(anchors.mean(dim=1) @ weight)
            coefficients = torch.sigmoid((anchors * context[:, None, :]).sum(dim=-1))
            pooled_scales.append((coefficients[..., None] * anchors).sum(dim=1))
        pooled = torch.stack(pooled_scales, dim=1)
        return self.scale_fusion(pooled.flatten(start_dim=1))


class AnchorLevelComparator(nn.Module):
    """Anchor-Level Comparison (ALC) using differences and Hadamard products."""

    def __init__(self, anchor_dim, num_scales, anchor_count):
        super().__init__()
        self.anchor_dim = int(anchor_dim)
        self.num_scales = int(num_scales)
        self.anchor_count = int(anchor_count)
        self.feature_dim = 2 * self.anchor_dim
        self.scale_comparators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.feature_dim),
                    nn.Linear(self.feature_dim, 16),
                    nn.GELU(),
                    nn.Linear(16, 1),
                )
                for _ in range(self.num_scales)
            ]
        )
        self._packed_signature = None
        for name in ("_packed_weight_1", "_packed_bias_1", "_packed_weight_2", "_packed_bias_2"):
            self.register_buffer(name, None, persistent=False)

    def clear_parameter_state(self):
        self._packed_signature = None
        self._packed_weight_1 = self._packed_bias_1 = None
        self._packed_weight_2 = self._packed_bias_2 = None

    def train(self, mode=True):
        self.clear_parameter_state()
        return super().train(mode)

    def _apply(self, fn):
        self.clear_parameter_state()
        return super()._apply(fn)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.clear_parameter_state()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _pack_linear_parameters(self):
        parameters = tuple(p for encoder in self.scale_comparators
                           for layer in (encoder[1], encoder[3]) for p in layer.parameters())
        signature = tuple((id(p), p._version) for p in parameters)
        if self._packed_signature != signature:
            self._packed_weight_1 = torch.stack([encoder[1].weight for encoder in self.scale_comparators]).detach()
            self._packed_bias_1 = torch.stack([encoder[1].bias for encoder in self.scale_comparators]).detach()
            self._packed_weight_2 = torch.stack([encoder[3].weight for encoder in self.scale_comparators]).detach()
            self._packed_bias_2 = torch.stack([encoder[3].bias for encoder in self.scale_comparators]).detach()
            self._packed_signature = signature

    def _batched_forward(self, anchor_scales_1, anchor_scales_2):
        self._pack_linear_parameters()
        left, right = torch.stack(anchor_scales_1), torch.stack(anchor_scales_2)
        features = torch.cat((torch.abs(left - right), left * right), dim=-1)
        # Apply each scale's affine LayerNorm with its own parameters before
        # batching the linear transformations.
        normalized = torch.stack([encoder[0](features[i])
                                  for i, encoder in enumerate(self.scale_comparators)])
        shape = normalized.shape
        hidden = torch.baddbmm(self._packed_bias_1[:, None],
            normalized.reshape(self.num_scales, -1, self.feature_dim),
            self._packed_weight_1.transpose(1, 2))
        hidden = F.gelu(hidden)
        scores = torch.baddbmm(self._packed_bias_2[:, None], hidden,
                              self._packed_weight_2.transpose(1, 2))
        return scores.view(self.num_scales, shape[1], self.anchor_count).permute(1, 0, 2).reshape(shape[1], -1)

    def forward(self, anchor_scales_1, anchor_scales_2):
        if len(anchor_scales_1) != self.num_scales or len(anchor_scales_2) != self.num_scales:
            raise ValueError(f"Anchor-level comparison requires {self.num_scales} anchor scales.")
        discrepancies = []
        expected_tail = (self.anchor_count, self.anchor_dim)
        for left, right in zip(anchor_scales_1, anchor_scales_2):
            if left.shape != right.shape or tuple(left.shape[1:]) != expected_tail:
                raise ValueError(
                    f"Anchor-level comparator expected [B, {self.anchor_count}, {self.anchor_dim}]."
                )
        if (not self.training and not torch.is_grad_enabled()
                and not torch.is_autocast_enabled() and not torch.is_autocast_cpu_enabled()):
            return self._batched_forward(anchor_scales_1, anchor_scales_2)
        for encoder, left, right in zip(
            self.scale_comparators,
            anchor_scales_1,
            anchor_scales_2,
        ):
            if left.shape != right.shape or tuple(left.shape[1:]) != expected_tail:
                raise ValueError(
                    f"Anchor-level comparator expected [B, {self.anchor_count}, "
                    f"{self.anchor_dim}]."
                )
            features = torch.cat(
                (torch.abs(left - right), left * right),
                dim=-1,
            )
            discrepancies.append(encoder(features).squeeze(-1))
        return torch.cat(discrepancies, dim=1)


class NeuralTensorNetwork(nn.Module):
    """Model graph-level relations with an NTN and GELU."""

    def __init__(self, anchor_dim, relation_channels=16):
        super().__init__()
        anchor_dim = int(anchor_dim)
        relation_channels = int(relation_channels)
        self.tensor = nn.Parameter(
            torch.empty(relation_channels, anchor_dim, anchor_dim)
        )
        nn.init.xavier_uniform_(self.tensor)
        self.ordered_linear = nn.Linear(
            2 * anchor_dim,
            relation_channels,
            bias=False,
        )
        self.relation_bias = nn.Parameter(torch.zeros(relation_channels))

    def forward(self, anchors_1, anchors_2):
        bilinear = torch.einsum(
            "bqd,kde,bqe->bqk",
            anchors_1,
            self.tensor,
            anchors_2,
        )
        ordered_pair = torch.cat((anchors_1, anchors_2), dim=-1)
        return F.gelu(
            bilinear
            + self.ordered_linear(ordered_pair)
            + self.relation_bias.view(1, 1, -1)
        )
