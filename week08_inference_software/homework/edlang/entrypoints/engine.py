import torch
from typing import List, Dict, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from dataclasses import dataclass

from edlang.entrypoints.config import EngineConfig


@dataclass
class Request:
    request_id: int
    prompt: str
    max_new_tokens: int
    current_len: int = 0
    sampling_params: Optional[Dict[str, Any]] = None  # Bonus Part

    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    generated_tokens: Optional[List[int]] = None
    generated_text: Optional[str] = None
    num_generated: int = 0
    is_finished: bool = False

    arrival_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None


@dataclass
class BatchResult:
    request_ids: List[int]
    new_tokens: List[List[int]]
    finished: List[bool]


class InferenceEngine:
    def __init__(self, engine_config: EngineConfig):
        self.model_config = engine_config.model_config

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=self.model_config.torch_dtype,
            device_map=self.model_config.device,
        )
        self.model.eval()
    
    def _should_finish_on_token(self, token_id: int, request: Request) -> bool:
        sampling_params = request.sampling_params or {}
        ignore_eos = sampling_params.get("ignore_eos_token", False)
        eos_token_id = sampling_params.get("eos_token_id", self.tokenizer.eos_token_id)

        if ignore_eos:
            return False

        if eos_token_id is None:
            return False

        return token_id == eos_token_id

    @torch.no_grad()
    def prefill(self, requests: List[Request]) -> BatchResult:
        """
        Prefill phase: tokenize prompts, run through model, generate first token.
        
        Steps:
        1. Tokenize prompts and create batch
        2. Forward pass with use_cache=True to get logits and KV cache
        3. Generate first token for each request (greedy: argmax)
        4. Save request state (input_ids, attention_mask, past_key_values)
        5. Check if finished (EOS token or max_new_tokens reached)
        
        Note: Use attention_mask to get real prompt length (without padding).
        """
        if not requests:
            return BatchResult(request_ids=[], new_tokens=[], finished=[])

        device = next(self.model.parameters()).device
        prompts = [request.prompt for request in requests]

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.max_prompt_length,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

        batch_request_ids = []
        batch_new_tokens = []
        batch_finished = []

        for i, request in enumerate(requests):
            real_prompt_len = int(attention_mask[i].sum().item())

            next_token_logits = outputs.logits[i, real_prompt_len - 1, :]
            next_token = self._sample(next_token_logits, request)

            request.input_ids = input_ids[i:i + 1, :real_prompt_len].detach().clone()
            request.attention_mask = attention_mask[i:i + 1, :real_prompt_len].detach().clone()
            request.current_len = real_prompt_len
            request.past_key_values = self._get_past_for_request(
                outputs.past_key_values,
                i,
                real_seq_len=real_prompt_len,
            )

            request.generated_tokens = [next_token]
            request.num_generated = 1
            request.is_finished = (
                request.num_generated >= request.max_new_tokens
                or self._should_finish_on_token(next_token, request)
            )

            batch_request_ids.append(request.request_id)
            batch_new_tokens.append([next_token])
            batch_finished.append(request.is_finished)

        return BatchResult(
            request_ids=batch_request_ids,
            new_tokens=batch_new_tokens,
            finished=batch_finished,
        )

    @torch.no_grad()
    def decode(self, requests: List[Request]) -> BatchResult:
        """
        Decode phase: generate next token for each active request using KV cache.
        
        Steps:
        1. Filter active (non-finished) requests
        2. Prepare batched KV cache with RIGHT padding
        3. Create batch from last generated tokens
        4. Build attention_mask accounting for different sequence lengths
        5. Forward pass with past_key_values and cache_position
        6. Generate next token (greedy: argmax)
        7. Update request state
        
        Note: Use RIGHT padding for KV cache. Handle finished requests separately.
        """
        if not requests:
            return BatchResult(request_ids=[], new_tokens=[], finished=[])
        
        active_requests = [request for request in requests if not request.is_finished]

        if not active_requests:
            return BatchResult(
                request_ids=[request.request_id for request in requests],
                new_tokens=[[] for _ in requests],
                finished=[True for _ in requests],
            )
        
        device = next(self.model.parameters()).device
        past_key_values = self._prepare_past_key_values_batch(active_requests)

        input_ids = torch.tensor(
            [[request.generated_tokens[-1]] for request in active_requests],
            dtype=torch.long,
            device=device,
        )

        max_past_len = max(request.current_len for request in active_requests)

        attention_mask = torch.zeros(
            (len(active_requests), max_past_len + 1),
            dtype=torch.long,
            device=device,
        )

        for i, request in enumerate(active_requests):
            attention_mask[i, :request.current_len + 1] = 1

        cache_position = torch.tensor(
            [request.current_len for request in active_requests],
            dtype=torch.long,
            device=device,
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )

        active_results = {}
        for i, request in enumerate(active_requests):
            next_token_logits = outputs.logits[i, -1, :]
            next_token = self._sample(next_token_logits, request)

            request.generated_tokens.append(next_token)
            request.num_generated += 1
            request.current_len += 1
            request.past_key_values = self._get_past_for_request(
                outputs.past_key_values,
                i,
                real_seq_len=request.current_len,
            )

            one = torch.ones(
                (1, 1),
                dtype=request.attention_mask.dtype,
                device=request.attention_mask.device,
            )
            request.attention_mask = torch.cat([request.attention_mask, one], dim=1)

            request.is_finished = (
                request.num_generated >= request.max_new_tokens
                or self._should_finish_on_token(next_token, request)
            )

            active_results[request.request_id] = ([next_token], request.is_finished)

        request_ids = []
        new_tokens = []
        finished = []

        for request in requests:
            request_ids.append(request.request_id)
            if request.request_id in active_results:
                token_list, is_finished = active_results[request.request_id]
                new_tokens.append(token_list)
                finished.append(is_finished)
            else:
                new_tokens.append([])
                finished.append(True)

        return BatchResult(
            request_ids=request_ids,
            new_tokens=new_tokens,
            finished=finished,
        )

    def _get_past_for_request(
        self,
        past_key_values,
        request_idx: int,
        real_seq_len: Optional[int] = None,
    ):
        if past_key_values is None:
            return None

        new_cache = DynamicCache()
        for layer_idx in range(self.model.config.num_hidden_layers):
            key   = past_key_values.key_cache[layer_idx][request_idx:request_idx+1]
            value = past_key_values.value_cache[layer_idx][request_idx:request_idx+1]

            if real_seq_len is not None and key.shape[2] > real_seq_len:
                key   = key[:, :, :real_seq_len, :]
                value = value[:, :, :real_seq_len, :]

            new_cache.update(key, value, layer_idx)
        return new_cache

    def _prepare_past_key_values_batch(self, requests: List[Request]):
        """
        Prepare batched KV cache from requests with RIGHT padding.
        
        Combines KV cache from different requests into one batch. Since requests
        may have different sequence lengths, add RIGHT padding to max_seq_len.
        """
        if not requests:
            return None

        max_seq_len = max(request.current_len for request in requests)
        batch_cache = DynamicCache()

        for layer_idx in range(self.model.config.num_hidden_layers):
            layer_keys = []
            layer_values = []

            for request in requests:
                key = request.past_key_values.key_cache[layer_idx]
                value = request.past_key_values.value_cache[layer_idx]

                pad_len = max_seq_len - key.shape[2]
                if pad_len > 0:
                    key_pad = torch.zeros(
                        (1, key.shape[1], pad_len, key.shape[3]),
                        dtype=key.dtype,
                        device=key.device,
                    )
                    value_pad = torch.zeros(
                        (1, value.shape[1], pad_len, value.shape[3]),
                        dtype=value.dtype,
                        device=value.device,
                    )
                    key = torch.cat([key, key_pad], dim=2)
                    value = torch.cat([value, value_pad], dim=2)

                layer_keys.append(key)
                layer_values.append(value)

            batch_key = torch.cat(layer_keys, dim=0)
            batch_value = torch.cat(layer_values, dim=0)
            batch_cache.update(batch_key, batch_value, layer_idx)

        return batch_cache

    def _sample(self, tokens_dist: torch.Tensor, request: Request) -> int:
        sampling_params = request.sampling_params or {}
        logits = tokens_dist.clone()
        eos_token_id = sampling_params.get("eos_token_id", self.tokenizer.eos_token_id)
        ignore_eos = sampling_params.get("ignore_eos_token", False)

        if ignore_eos and eos_token_id is not None and 0 <= eos_token_id < logits.numel():
            logits[eos_token_id] = float("-inf")

        do_sample = sampling_params.get("do_sample", False)
        temperature = float(sampling_params.get("temperature", 1.0))

        if not do_sample or temperature <= 0:
            return int(torch.argmax(logits).item())
        
        logits = logits / max(temperature, 1e-5)

        top_k = sampling_params.get("top_k", None)
        if top_k is not None and 0 < top_k < logits.numel():
            top_values, _ = torch.topk(logits, top_k)
            threshold = top_values[-1]
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, float("-inf")),
                logits,
            )
        
        top_p = sampling_params.get("top_p", None)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_mask = cumulative_probs > top_p
            sorted_mask[1:] = sorted_mask[:-1].clone()
            sorted_mask[0] = False

            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

            filtered_logits = torch.full_like(logits, float("-inf"))
            filtered_logits.scatter_(0, sorted_indices, sorted_logits)
            logits = filtered_logits

        probs = torch.softmax(logits, dim=-1)

        if torch.isnan(probs).any() or probs.sum() <= 0:
            return int(torch.argmax(tokens_dist).item())

        token = torch.multinomial(probs, num_samples=1)
        return int(token.item())

    def get_generated_text(self, request: Request) -> str:
        if not request.generated_tokens:
            return request.prompt

        full_ids = request.input_ids[0].tolist() + request.generated_tokens
        return self.tokenizer.decode(full_ids, skip_special_tokens=True)
