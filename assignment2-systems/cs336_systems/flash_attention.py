"""
An implementation of Flash Attention2 from the paper

Forward and Backward Passes for Flash Attention implement in PyTorch
"""

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx  # type: ignore
from einops import einsum, rearrange, reduce
from jaxtyping import Float
import math
import triton
import triton.language as tl


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def _flash_attention_single_batch(
        Q: Float[Tensor, "n_q d"],
        K: Float[Tensor, "n_k d"],
        V: Float[Tensor, "n_v d"],
        is_causal: bool = False,
        B_q: int = 16,
        B_k: int = 16,
    ) -> tuple[Tensor, Tensor]:
        """
        Flash attention for a single batch element.

        The tile sizes B_q and B_k refer to the number of queries and keys processed
        in each tile, creating square blocks of size B_q × B_k in the attention matrix.

        Args:
            Q: Query tensor of shape (n_q, d)
            K: Key tensor of shape (n_k, d)
            V: Value tensor of shape (n_v, d)
            is_causal: Whether to apply causal masking
            B_q: Tile size for queries (number of queries per tile)
            B_k: Tile size for keys (number of keys per tile)

        Returns:
            output: Attention output of shape (n_q, d)
            L: Log-sum-exp values of shape (n_q,)
        """
        n_q, d = Q.shape
        n_k, _ = K.shape
        n_v, _ = V.shape

        T_q = math.ceil(n_q / B_q)  # number of query tiles
        T_k = math.ceil(n_k / B_k)  # number of key and value tiles

        # Initialize output and L tensors
        output = torch.zeros((n_q, d), device=Q.device)
        L = torch.zeros((n_q,), device=Q.device)

        for i in range(T_q):
            # load Q_i tile
            Q_i = Q[i * B_q : min((i + 1) * B_q, n_q), :]  # (B_q, d)
            actual_B_q = Q_i.shape[0]  # actual size might be smaller for last tile

            # Initialize O_i = B_q x d
            O_i = torch.zeros((actual_B_q, d), device=Q.device)

            # Initialize l_i = B_q
            l_i = torch.zeros((actual_B_q,), device=Q.device)

            # Initialize m_i = B_q with -inf
            m_i = torch.full((actual_B_q,), -float("inf"), device=Q.device)

            for j in range(T_k):
                # load K_j, V_j tiles
                K_j = K[j * B_k : min((j + 1) * B_k, n_k), :]  # (B_k, d)
                V_j = V[j * B_k : min((j + 1) * B_k, n_v), :]  # (B_k, d)
                actual_B_k = K_j.shape[0]  # actual size might be smaller for last tile

                # Compute pre-softmax scores S_ij = Q_i K_j^T/sqrt(d) shape (B_q, B_k).
                S_ij = einsum(Q_i, K_j, "q d, k d -> q k") / math.sqrt(d)

                # Apply causal mask if needed
                if is_causal:
                    # Create causal mask for this tile
                    q_indices = torch.arange(i * B_q, min((i + 1) * B_q, n_q), device=Q.device)
                    k_indices = torch.arange(j * B_k, min((j + 1) * B_k, n_k), device=Q.device)
                    causal_mask = q_indices[:, None] >= k_indices[None, :]
                    causal_mask = causal_mask[:actual_B_q, :actual_B_k]
                    S_ij = S_ij.masked_fill(~causal_mask, -float("inf"))

                # Compute m_ij = max over rows of S_ij shape (B_q,)
                # m_ij = max(m_ij-1, rowmax(S_ij))
                row_max_S_ij = reduce(S_ij, "q k -> q", "max")
                previous_m_i = m_i
                m_i = torch.max(previous_m_i, row_max_S_ij)

                # Compute p_ij = exp(S_ij - m_i) shape (B_q, B_k)
                P_ij = torch.exp(S_ij - rearrange(m_i, "q -> q 1"))

                # Compute l_ij which is the softmax denominator
                # l_ij = exp(m_ij-1 - m_ij) * l_ij-1 + sum_rows(P_ij)
                l_i = torch.exp(previous_m_i - m_i) * l_i + reduce(P_ij, "q k -> q", "sum")

                # Compute O_ij = diag(exp(m_ij-1 - m_ij)) O_ij-1 + P_ij V_ja
                scale = torch.exp(previous_m_i - m_i)
                # making use of broadcasting
                O_i = rearrange(scale, "q -> q 1") * O_i + einsum(P_ij, V_j, "q k, k d -> q d")

            # Divide O_i by l_i to get the final output for this tile
            # O_i = diag(1/l_Tk) O_iTk
            O_i = rearrange(1.0 / l_i, "q -> q 1") * O_i

            # Compute the final LSE = m_i + log(l_i)
            lse_i = m_i + torch.log(l_i)

            # Write O_i and lse_i back to O and L
            output[i * B_q : min((i + 1) * B_q, n_q), :] = O_i
            L[i * B_q : min((i + 1) * B_q, n_q)] = lse_i

        return output, L

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        Q: Float[Tensor, "batch n_q d"],
        K: Float[Tensor, "batch n_k d"],
        V: Float[Tensor, "batch n_v d"],
        is_causal: bool = False,
    ) -> Tensor:
        batch, n_q, d = Q.shape
        _, n_k, _ = K.shape
        _, n_v, __ = V.shape

        # Initialize output and L tensors
        output = torch.zeros((batch, n_q, d), device=Q.device)
        L = torch.zeros((batch, n_q), device=Q.device)

        # Process each batch independently
        for b in range(batch):
            batch_output, batch_L = FlashAttention._flash_attention_single_batch(Q[b], K[b], V[b], is_causal)
            output[b] = batch_output
            L[b] = batch_L

        ctx.save_for_backward(Q, K, V, output, L)

        return output

    @staticmethod
    @torch.compile
    def backward(
        ctx: FunctionCtx,
        grad_output: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        Q, K, V, output, L = ctx.saved_tensors
        dO = grad_output

        batch, nq, d = Q.shape
        scale = 1.0 / math.sqrt(d)

        # D = sum(do * O)
        D = einsum(dO, output, "b q d, b q d -> b q")

        # Recompute S = QK^T/sqrt(d)
        S = einsum(Q, K, "b q d, b k d -> b q k") * scale  # (b, n_q, n_k)

        # recompute P  = exp(S - L)
        P = torch.exp(S - rearrange(L, "b q -> b q 1"))  # (b, n_q, n_k)

        # dV = P^T dO
        dV = einsum(P, dO, "b q k, b q d -> b k d")

        # dP = dO V^T
        dP = einsum(dO, V, "b q d, b k d -> b q k")

        # dS = P * (dP - D)  (elementwise)
        dS = P * (dP - D.unsqueeze(-1))

        # dQ = dS K / sqrt(d)
        dQ = einsum(dS, K, "b q k, b k d -> b q d") * scale

        dK = einsum(rearrange(dS, "b q k -> b k q"), Q, "b k q, b q d -> b k d") * scale

        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr,  # pointer to Q matrix
    K_ptr,  # pointer to K matrix
    V_ptr,  # pointer to V matrix
    O_ptr,  # pointer to output matrix
    L_ptr,  # pointer to log-sum-exp vector
    stride_qb,  # batch stride for Q
    stride_qq,  # query stride for Q
    stride_qd,  # feature stride for Q
    stride_kb,  # batch stride for K
    stride_kk,  # key stride for K
    stride_kd,  # feature stride for K
    stride_vb,  # batch stride for V
    stride_vk,  # key stride for V
    stride_vd,  # feature stride for V
    stride_ob,  # batch stride for output
    stride_oq,  # query stride for output
    stride_od,  # feature stride for output
    stride_lb,  # batch stride for L
    stride_lq,  # query stride for L
    N_QUERIES,  # number of queries
    N_KEYS,  # number of keys
    scale,  # scaling factor 1/sqrt(d)
    D: tl.constexpr,  # feature dimension
    Q_TILE_SIZE: tl.constexpr,  # tile size for queries = B_q
    K_TILE_SIZE: tl.constexpr,  # tile size for keys = B_k
    is_causal: tl.constexpr,  # whether to apply causal masking
):
    # program indices
    query_tile_idx = tl.program_id(1)  # which query tile
    batch_idx = tl.program_id(0)  # which batch element

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_idx * stride_qb,  # start pointer
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_idx * stride_kb,  # start pointer
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),  # transposed
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )  # we want to work with K.T

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_idx * stride_vb,  # start pointer
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_idx * stride_ob,  # start pointer
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_idx * stride_lb,  # start pointer
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    M = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)  # running max for log-sum-exp

    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)  # number of key/value tiles

    # load Q tile
    Q_block = tl.load(Q_block_ptr)  # (B_q, d)

    # load O, L. we have initialized them to zero in python
    O_block = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # (B_q, d)
    L = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # (B_q,)

    # inner loop over key/value tiles
    for j in range(T_k):
        # load K, V tiles
        K_block = tl.load(K_block_ptr)  # (B_k, d)
        V_block = tl.load(V_block_ptr)  # (B_k, d)

        # Compute pre-softmax scores S_ij = Q_i K_j^T/sqrt(d) shape (B_q, B_k).
        S = tl.dot(Q_block, K_block) * scale  # (B_q, B_k)

        # Apply causal mask if needed
        if is_causal:
            q_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_idx * Q_TILE_SIZE  # (B_q,)
            k_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # (B_k,)
            causal_mask = q_indices[:, None] < k_indices[None, :]  # (B_q, B_k)
            # add -1e6 to masked positions for numerical stability
            S = tl.where(causal_mask, -1e6, S)

        # compute M_j
        M_new = tl.maximum(M, tl.max(S, axis=1))  # (B_q,)

        # Compute P_ij = exp(S - M_j)
        P = tl.exp(S - tl.expand_dims(M_new, 1))  # (B_q, B_k)

        # Compute l_j = rowsum(P_ij) + exp(M_j-1 - M_j) * l_j-1
        L_new = tl.sum(P, axis=1) + tl.exp(M - M_new) * L  # (B_q,)

        # Compute O_ij = P_ij V_j + diag(exp(M_j-1 - M_j)) O_ij-1
        O_block = tl.dot(P.to(V_ptr.dtype.element_ty), V_block) + tl.exp(M - M_new)[:, None] * O_block  # (B_q, d)

        # advance K, V block pointers to next tile
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        # update M, L for next iteration
        M = M_new
        L = L_new

    # Compute O_i = diag(1/l_Tk) O_iTk
    O_block = O_block * tl.expand_dims(1.0 / L, 1)  # (B_q, d)

    # Compute the final LSE = m_i + log(l_i)
    L = M + tl.log(L)  # (B_q,)

    # store O_i and L_i back to O and L
    tl.store(O_block_ptr, O_block.to(O_ptr.dtype.element_ty))
    tl.store(L_block_ptr, L)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        Q: Float[Tensor, "batch n_q d"],
        K: Float[Tensor, "batch n_k d"],
        V: Float[Tensor, "batch n_v d"],
        is_causal: bool = False,
    ):
        batch, n_q, d = Q.shape
        _, n_k, _ = K.shape
        _, n_v, __ = V.shape

        # Use constants for tile sizes - these must be compile-time constants
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        T_q = math.ceil(n_q / Q_TILE_SIZE)  # number of query tiles

        # grid size = (batch, T_q)
        grid = (batch, T_q)

        output_tensor = torch.zeros((batch, n_q, d), device=Q.device, dtype=torch.float32)
        L = torch.zeros((batch, n_q), device=Q.device, dtype=torch.float32)
        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            output_tensor,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            output_tensor.stride(0),
            output_tensor.stride(1),
            output_tensor.stride(2),
            L.stride(0),
            L.stride(1),
            N_QUERIES=n_q,
            N_KEYS=n_k,
            scale=1.0 / math.sqrt(d),
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, output_tensor, L)
        ctx.is_causal = is_causal
        return output_tensor

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_output: Tensor,
        grad_L: Tensor,
    ) -> tuple[None, None, None, None]:
        # TODO: implement backward pass; currently returns no gradients
        raise NotImplementedError()
