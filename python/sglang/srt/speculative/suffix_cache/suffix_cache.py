# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np

from sglang.srt.speculative.suffix_cache._C import SuffixTree, Candidate


@dataclass
class SuffixSpecResult:
    """
    A dataclass representing the result of a speculation using SuffixDecoding.

    Attributes:
        token_ids (List[int]): List of token IDs in the speculation result.
        parents (List[int]): List of parent indices for each token used to
            encode the tree structure. The parent token of token_ids[i] is
            token_ids[parents[i]].
        probs (List[float]): List of estimated probabilities for each token.
        score (float): The overall score of the suffix match computed as the
            sum of the estimated probabilities of each speculated token.
        match_len (int): The length of the pattern match that yielded this
            speculation result.
    """
    token_ids: List[int] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)
    probs: List[float] = field(default_factory=list)
    score: float = 0.0
    match_len: int = 0

    @staticmethod
    def from_candidate(candidate: Candidate) -> SuffixSpecResult:
        return SuffixSpecResult(
            token_ids=candidate.token_ids,
            parents=candidate.parents,
            probs=candidate.probs,
            score=candidate.score,
            match_len=candidate.match_len,
        )


class SuffixCache:

    def __init__(self, max_depth: int = 64):
        self._max_depth = max_depth
        self._suffix_tree = SuffixTree(max_depth)
        self._prompt_trees = {}
        self._req_to_seq_id = {}

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def has_cached_prompt(self, req_id: Hashable) -> bool:
        return req_id in self._prompt_trees

    def cached_prompt_ids(self) -> List[Hashable]:
        return list(self._prompt_trees.keys())

    def cache_prompt(self, req_id: Hashable, prompt_token_ids: Sequence[int]):
        """
        Cache a prompt for a specific request ID. Future speculations for the
        same request may also source draft tokens from this prompt.

        Args:
            req_id (Hashable): The request identifier. Must be a hashable value
                that uniquely identifies the request.
            prompt_token_ids (Sequence[int]): A sequence of token IDs
                representing the prompt to be cached.

        Raises:
            ValueError: If a prompt already exists for the given request ID.

        Note:
            The caller should evict the cached prompt using `evict_prompt` once
            the prompt is no longer needed (i.e. the request is completed).
        """
        if req_id in self._prompt_trees:
            raise ValueError(f"Prompt already exists for request '{req_id}'")
        self._prompt_trees[req_id] = SuffixTree(self._max_depth)
        self._prompt_trees[req_id].extend(0, prompt_token_ids)

    def evict_prompt(self, req_id: Hashable):
        """
        Evicts a prompt from the cache for a specific request.

        Args:
            req_id (Hashable): The unique identifier for the request whose
                prompt should be evicted.

        Raises:
            ValueError: If no prompt exists for the given request identifier.
        """
        if req_id not in self._prompt_trees:
            raise ValueError(f"Prompt does not exist for request '{req_id}'")
        del self._prompt_trees[req_id]

    def _get_or_assign_seq_id(self, req_id: Hashable) -> int:
        if req_id not in self._req_to_seq_id:
            self._req_to_seq_id[req_id] = len(self._req_to_seq_id)
        return self._req_to_seq_id[req_id]

    def update_response(
        self,
        req_id: Hashable,
        token_ids: Union[int | Sequence[int]],
    ):
        """
        Update the cached response for a given request by adding token(s) to
        its end. It does not rely on the prompt being cached for the request,
        and its lifetime does not depend on the prompt's existence. Once the
        response is updated, the new tokens can be used for future speculations
        for all requests.

        Args:
            req_id (Hashable): The unique identifier for the request.
            token_ids (Union[int, Sequence[int]]): Either a single token ID
                (int) or a sequence of token IDs to be appended to the response
                for the given request.

        Notes:
            - If req_id doesn't exist, a new empty sequence will be initialized.
            - If token_ids is a single integer, it's added as a single token.
            - If token_ids is a sequence, all tokens in the sequence are added.
        """
        seq_id = self._get_or_assign_seq_id(req_id)
        if isinstance(token_ids, int):
            self._suffix_tree.append(seq_id, token_ids)
            if req_id in self._prompt_trees:
                self._prompt_trees[req_id].append(0, token_ids)
        else:
            self._suffix_tree.extend(seq_id, token_ids)
            if req_id in self._prompt_trees:
                self._prompt_trees[req_id].extend(0, token_ids)

    def speculate(
        self,
        req_id: Hashable,
        pattern: Sequence[int],
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
        use_cached_prompt: bool = True,
    ) -> SuffixSpecResult:
        """
        Speculates and returns the most likely continuation of a given token
        pattern using the request-specific prompt cache (if available) and the
        global cache of previous responses.

        Args:
            req_id (Hashable): The unique identifier for the request.
            pattern (Sequence[int]): The sequence of token IDs to match and
                continue from.
            max_spec_tokens (int): Maximum number of tokens to speculate. If 0,
                uses the cache's max_depth.
            max_spec_factor (float): Factor that limits speculation based on
                matched pattern length.
            min_token_prob (float): Minimum estimated probability threshold for
                candidate tokens.
            use_tree_spec (bool): If True, uses tree-based speculation.
            use_cached_prompt (bool): If True, uses the cached prompt for the
                request in addition to the global cache of previous responses.

        Returns:
            The speculation result containing the most likely continuation
            tokens, their probabilities, and overall score.

        Raises:
            ValueError: If the prompt doesn't exist for the given req_id when
                use_cached_prompt is True, or if the pattern is invalid.
        """
        if use_cached_prompt and req_id not in self._prompt_trees:
            raise ValueError(f"Prompt does not exist for request '{req_id}'")
        if not pattern:
            raise ValueError("Pattern must not be empty")

        if max_spec_tokens is None:
            max_spec_tokens = self.max_depth

        if len(pattern) > self._max_depth:
            pattern = pattern[-self._max_depth :]

        if use_cached_prompt:
            prompt_tree = self._prompt_trees[req_id]
            candidate = prompt_tree.speculate(
                pattern,
                max_spec_tokens,
                max_spec_factor,
                max_spec_offset,
                min_token_prob,
                use_tree_spec)
            result = SuffixSpecResult.from_candidate(candidate)
        else:
            result = SuffixSpecResult()

        candidate = self._suffix_tree.speculate(
            pattern,
            max_spec_tokens,
            max_spec_factor,
            max_spec_offset,
            min_token_prob,
            use_tree_spec)
        if candidate.score > result.score:
            result = SuffixSpecResult.from_candidate(candidate)
        return result

    def batch_get(
        self,
        batch_tokens: List[List[int]],
        req_ids: List[Hashable] = None,
        draft_token_num: int = 8,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_cached_prompt: bool = True,
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
            use_cached_prompt: Whether to use cached prompt for speculation

        Returns:
            Tuple of (draft_tokens, tree_mask) as numpy arrays
            - draft_tokens: shape (bs * draft_token_num,)
            - tree_mask: shape (bs * draft_token_num * draft_token_num,)
        """
        bs = len(batch_tokens)
        if req_ids is None:
            req_ids = [None] * bs
            use_cached_prompt = False

        all_draft_tokens = []
        all_masks = []

        for i, tokens in enumerate(batch_tokens):
            req_id = req_ids[i]

            # Get speculative tokens using suffix cache
            if len(tokens) == 0:
                # Empty pattern, return default zeros
                result = SuffixSpecResult()
            else:
                # Prepare pattern (limit to max_depth)
                pattern = tokens if len(tokens) <= self._max_depth else tokens[-self._max_depth:]

                try:
                    # Use speculate method with req_id for prompt cache
                    result = self.speculate(
                        req_id=req_id,
                        pattern=pattern,
                        max_spec_tokens=draft_token_num,
                        max_spec_factor=max_spec_factor,
                        max_spec_offset=max_spec_offset,
                        min_token_prob=min_token_prob,
                        use_tree_spec=False,  # Use simple chain for compatibility
                        use_cached_prompt=use_cached_prompt and (req_id is not None),
                    )
                except Exception as e:
                    # If speculation fails, return empty result
                    result = SuffixSpecResult()

            # Pad or truncate to draft_token_num
            draft_tokens = result.token_ids[:draft_token_num]
            parents = result.parents[:draft_token_num] if result.parents else []

            # IMPORTANT: Following N-gram's pattern, ALWAYS prepend the last token from pattern
            # as the first draft token (with parent=-1), regardless of speculation results.
            # This ensures continuity and matches N-gram's behavior.
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
            mask = self._build_tree_mask_from_parents(
                parents,
                draft_token_num
            )

            all_draft_tokens.extend(draft_tokens)
            all_masks.append(mask)

        # Flatten masks to match N-gram format: (bs * draft_token_num * draft_token_num,)
        draft_tokens_array = np.array(all_draft_tokens, dtype=np.int64)
        mask_array = np.array(all_masks, dtype=np.int64).reshape(-1)

        return draft_tokens_array, mask_array

    def _build_tree_mask_from_parents(
        self,
        parents: List[int],
        draft_token_num: int
    ) -> np.ndarray:
        """
        Build tree attention mask from parent indices.

        This implementation follows the N-gram C++ logic:
        - Token i can see all tokens that its parent can see
        - Token i can also see itself

        Reference: ngram.cpp fillResult() lines 43-51

        Args:
            parents: List of parent indices for each token
            draft_token_num: Size of the mask matrix

        Returns:
            Attention mask of shape (draft_token_num, draft_token_num)
            mask[i, j] = 1 means token i can attend to token j
        """
        mask = np.zeros((draft_token_num, draft_token_num), dtype=np.int64)

        # Pad parents if needed
        # Use parent=0 (root token) for padding to match N-gram C++ logic (ngram.cpp:40)
        parents_padded = parents + [0] * (draft_token_num - len(parents))

        # Initialize: first token (index 0) can see itself
        mask[0, 0] = 1

        for i in range(1, draft_token_num):  # Start from 1, as mask[0,0] already set
            parent_idx = parents_padded[i]

            if parent_idx != -1 and parent_idx < draft_token_num:
                # Key: Inherit visibility from parent
                # Copy the first (parent_idx + 1) elements from parent's row
                # This matches the N-gram C++ logic: memcpy(&info.mask[i * n], &info.mask[prevs[i] * n], prevs[i] + 1)
                mask[i, :parent_idx + 1] = mask[parent_idx, :parent_idx + 1]

            # Each token can always see itself
            mask[i, i] = 1

        return mask
