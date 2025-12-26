import logging
from typing import Hashable, List, Optional, Tuple

import numpy as np
import torch
from arctic_inference.suffix_decoding import SuffixDecodingCache, SuffixDecodingDraft
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.suffix_info import SuffixVerifyInput

USE_FULL_MASK = True

logger = logging.getLogger(__name__)


class SuffixCache(SuffixDecodingCache):
    def batch_get(
        self,
        batch_tokens: List[List[int]],
        req_ids: List[Hashable] = None,
        draft_token_num: int = 8,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch version of speculate for compatibility with N-gram interface.

        Args:
            batch_tokens: List of token sequences, one per request
            req_ids: List of request IDs for accessing cached prompts
            draft_token_num: Number of draft tokens to generate per request
            max_spec_factor: Maximum speculation factor
            max_spec_offset: Maximum speculation offset
            min_token_prob: Minimum token probability threshold

        Returns:
            Tuple of (draft_tokens, tree_mask) as numpy arrays
            - draft_tokens: shape (bs * draft_token_num,)
            - tree_mask: shape (bs * draft_token_num * draft_token_num,)
        """
        bs = len(batch_tokens)
        if req_ids is None:
            req_ids = [None] * bs

        all_draft_tokens = []
        all_masks = []

        for i, tokens in enumerate(batch_tokens):
            req_id = req_ids[i]

            # Get speculative tokens using suffix cache
            if len(tokens) == 0 or req_id is None or req_id not in self.active_requests:
                # Empty pattern or request not active, return default zeros
                result = SuffixDecodingDraft()
            else:
                # Prepare context (limit to max_tree_depth)
                context = (
                    tokens
                    if len(tokens) <= self.max_tree_depth
                    else tokens[-self.max_tree_depth :]
                )

                try:
                    # Use speculate method with req_id (only works for active requests)
                    result = self.speculate(
                        req_id=req_id,
                        context=context,
                        max_spec_tokens=draft_token_num,
                        max_spec_factor=max_spec_factor,
                        max_spec_offset=max_spec_offset,
                        min_token_prob=min_token_prob,
                        use_tree_spec=False,  # Use simple chain for compatibility
                    )
                except Exception as e:
                    # If speculation fails, return empty result
                    result = SuffixDecodingDraft()

            # Pad or truncate to draft_token_num
            draft_tokens = result.token_ids[:draft_token_num]
            parents = result.parents[:draft_token_num] if result.parents else []

            if len(tokens) > 0:
                last_token = tokens[-1]
                # Prepend last_token and adjust parents
                if len(draft_tokens) == 0:
                    # No speculation results
                    draft_tokens = [last_token]
                    parents = [-1]
                else:
                    # Have speculation results, but still prepend last_token
                    # When prepending last_token at position 0, ALL parent indices shift by +1
                    # because all speculation tokens move one position to the right
                    # Example: original parents [-1, 0, 1, 2] becomes [-1, 0, 1, 2, 3] after prepending
                    #   - prepended token at pos 0: parent=-1 (root)
                    #   - speculation token 0 (now at pos 1): parent=0 (points to prepended token)
                    #   - speculation token 1 (now at pos 2): parent=1 (points to spec token 0)
                    #   - speculation token 2 (now at pos 3): parent=2 (points to spec token 1)
                    parents = [-1] + [p + 1 for p in parents]
                    draft_tokens = [last_token] + draft_tokens
                    # Truncate to draft_token_num
                    draft_tokens = draft_tokens[:draft_token_num]
                    parents = parents[:draft_token_num]

            draft_tokens += [0] * (draft_token_num - len(draft_tokens))

            # Build tree mask from parents
            mask = self._build_tree_mask_from_parents(parents, draft_token_num)

            all_draft_tokens.extend(draft_tokens)
            all_masks.append(mask)

        # Flatten masks to match N-gram format: (bs * draft_token_num * draft_token_num,)
        draft_tokens_array = np.array(all_draft_tokens, dtype=np.int64)
        mask_array = np.array(all_masks, dtype=np.int64).reshape(-1)

        return draft_tokens_array, mask_array

    def _build_tree_mask_from_parents(
        self, parents: List[int], draft_token_num: int
    ) -> np.ndarray:
        """
        Build tree attention mask from parent indices.

        This implementation follows the N-gram C++ logic:
        - Token i can see all tokens that its parent can see
        - Token i can also see itself

        Args:
            parents: List of parent indices for each token
            draft_token_num: Size of the mask matrix

        Returns:
            Attention mask of shape (draft_token_num, draft_token_num)
            mask[i, j] = 1 means token i can attend to token j
        """
        mask = np.zeros((draft_token_num, draft_token_num), dtype=np.int64)

        # Pad parents if needed
        parents_padded = parents + [0] * (draft_token_num - len(parents))

        # Initialize: first token (index 0) can see itself
        mask[0, 0] = 1

        for i in range(1, draft_token_num):  # Start from 1, as mask[0,0] already set
            parent_idx = parents_padded[i]

            if parent_idx != -1 and parent_idx < draft_token_num:
                # Key: Inherit visibility from parent
                # Copy the first (parent_idx + 1) elements from parent's row
                # This matches the N-gram C++ logic: memcpy(&info.mask[i * n], &info.mask[prevs[i] * n], prevs[i] + 1)
                mask[i, : parent_idx + 1] = mask[parent_idx, : parent_idx + 1]

            # Each token can always see itself
            mask[i, i] = 1

        return mask


class SuffixWorker:

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens

        self.suffix_cache_max_depth = server_args.speculative_suffix_cache_max_depth
        self.suffix_max_spec_factor = server_args.speculative_suffix_max_spec_factor
        self.suffix_max_spec_offset = server_args.speculative_suffix_max_spec_offset
        self.suffix_min_token_prob = server_args.speculative_suffix_min_token_prob

        self.suffix_max_cached_requests = (
            server_args.speculative_suffix_max_cached_requests
        )

        self.max_batch_size = target_worker.max_running_requests
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        self._init_preallocated_tensors()

        self.suffix_cache = SuffixCache(
            max_tree_depth=self.suffix_cache_max_depth,
            max_cached_requests=self.suffix_max_cached_requests,
        )

    def clear_cache_pool(self):
        """Clear the suffix cache pool"""
        self.suffix_cache = SuffixCache(
            max_tree_depth=self.suffix_cache_max_depth,
            max_cached_requests=self.suffix_max_cached_requests,
        )

    def _efficient_concat_last_n(self, seq1: List[int], seq2: List[int], n: int):
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]

        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    def _init_preallocated_tensors(self):
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        self.draft_tokens_batch = []
        self.tree_mask_batch = []
        self.retrieve_indexes_batch = []
        self.retrive_next_token_batch = []
        self.retrive_next_sibling_batch = []
        self.positions_batch = []

        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrive_next_token_batch.append(self.retrive_next_token[:bs, :])
            self.retrive_next_sibling_batch.append(self.retrive_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )

    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare draft tokens using suffix cache.

        This method mimics the logic in generate_draft_token_ids_suffix
        to compute max_spec_tokens dynamically for each request.
        """
        bs = batch.batch_size()

        batch_tokens = []
        req_ids = []

        for req in batch.reqs:
            # Get the pattern tokens (last N tokens)
            check_token = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.suffix_cache_max_depth
            )
            batch_tokens.append(check_token)
            req_ids.append(req.rid)

        # Call batch_get with all necessary parameters
        req_drafts, mask = self.suffix_cache.batch_get(
            batch_tokens=batch_tokens,
            req_ids=req_ids,
            draft_token_num=self.draft_token_num,
            max_spec_factor=self.suffix_max_spec_factor,
            max_spec_offset=self.suffix_max_spec_offset,
            min_token_prob=self.suffix_min_token_prob,
        )

        total_draft_token_num = len(req_drafts)

        # Check if speculative decoding is needed; here we always enforce it
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        if batch.forward_mode.is_extend():
            return

        bs = batch.batch_size()

        retrive_index = self.retrieve_indexes_batch[bs]
        retrive_next_token = self.retrive_next_token_batch[bs]
        retrive_next_sibling = self.retrive_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        req_drafts, mask = self._prepare_draft_tokens(batch)
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable
            retrive_index,  # mutable
            retrive_next_token,  # mutable
            retrive_next_sibling,  # mutable
            bs,
            self.draft_token_num,
        )

        # NOTE: QLEN_MASK is faster than FULL_MASK, but requires corresponding changes in flashinfer.
        # Testing shows about 8% performance improvement (the effect is roughly proportional to batch size).
        if USE_FULL_MASK:
            tree_mask = []
            mask = mask.reshape(
                batch.batch_size(), self.draft_token_num, self.draft_token_num
            )
            for i, req in enumerate(batch.reqs):
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                req_mask = torch.ones((self.draft_token_num, seq_len - 1)).cuda()
                req_mask = torch.cat(
                    (req_mask, torch.from_numpy(mask[i]).cuda()), dim=1
                ).to(torch.bool)
                tree_mask.append(req_mask.flatten())
            tree_mask = torch.cat(tree_mask, dim=0)

        batch.spec_algorithm = SpeculativeAlgorithm.SUFFIX
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = SuffixVerifyInput(
            draft_tokens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            self.draft_token_num,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        self._prepare_for_speculative_decoding(batch)
        model_worker_batch = batch.get_model_worker_batch()
        num_accepted_tokens = 0

        if model_worker_batch.forward_mode.is_target_verify():
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )
            verify_input = model_worker_batch.spec_info
            logits_output, next_token_ids, num_accepted_tokens = verify_input.verify(
                batch, logits_output, self.page_size
            )
            # Extract the actually accepted tokens for cache update
            sampled_token_ids = self._extract_accepted_tokens(batch, verify_input)
            self._update_suffix_cache(batch, sampled_token_ids)
            batch.forward_mode = ForwardMode.DECODE

        else:
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            logits_output, next_token_ids, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _extract_accepted_tokens(
        self, batch: ScheduleBatch, spec_info
    ) -> list[list[int]]:
        """Extract the actually accepted tokens from verify results

        Similar to the original implementation, we need to get the tokens that were
        accepted during verification from the spec_info.
        """
        bs = batch.batch_size()
        sampled_token_ids = []

        accept_length_cpu = spec_info.accept_length.cpu().tolist()

        # Get the verified tokens (these are the tokens that passed verification)
        verified_id_cpu = spec_info.verified_id.cpu().tolist()

        offset = 0
        for i in range(bs):
            # accept_length[i] is the number of draft tokens accepted
            # We need to include the bonus token as well (+1)
            sample_len = accept_length_cpu[i] + 1
            sample_ids = verified_id_cpu[offset : offset + sample_len]
            sampled_token_ids.append(sample_ids)
            offset += sample_len

        return sampled_token_ids

    def _update_suffix_cache(
        self, batch: ScheduleBatch, sampled_token_ids: list[list[int]]
    ) -> None:
        """Update suffix cache with accepted tokens from verification"""
        seen_req_ids = set()

        for i, sampled_ids in enumerate(sampled_token_ids):
            req_id = batch.reqs[i].rid
            seen_req_ids.add(req_id)

            if not sampled_ids:
                continue

            if req_id not in self.suffix_cache.active_requests:
                prompt_token_ids = batch.reqs[i].origin_input_ids
                self.suffix_cache.start_request(req_id, prompt_token_ids)

                # IMPORTANT: Also cache all previously generated output tokens
                # (before the current sampled_ids) to maintain continuity
                output_ids = batch.reqs[i].output_ids
                # Calculate how many tokens were generated before this round
                previous_output_len = len(output_ids) - len(sampled_ids)
                if previous_output_len > 0:
                    previous_tokens = output_ids[:previous_output_len]
                    self.suffix_cache.add_active_response(req_id, previous_tokens)

            self.suffix_cache.add_active_response(req_id, sampled_ids)

        # Stop requests that are not seen (request completed)
        for req_id in list(self.suffix_cache.active_requests):
            if req_id not in seen_req_ids:
                self.suffix_cache.stop_request(req_id)
