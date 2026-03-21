from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

import sys
import os

from edlang.entrypoints.engine import Request, InferenceEngine, BatchResult
from edlang.managers.metric_manager import MetricManager, METRIC_SHOW_PERIOD


@dataclass
class SchedulerConfig:
    max_batch_size: int = 8
    max_waiting_requests: int = 100
    prefill_timeout_ms: float = 50.0
    enable_metrics: bool = False

    policy_name: str = "baseline"
    max_prefill_per_step: int = 4


class EDLangScheduler:

    def __init__(
        self,
        engine: InferenceEngine,
        config: Optional[SchedulerConfig] = None,
    ):
        self.engine = engine
        self.config = config or SchedulerConfig()

        self.waiting_queue = deque()
        self.active_requests = []

        self.next_request_id = 0
        self.metrics_manager = MetricManager(enable_metrics=self.config.enable_metrics)

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ):
        request = Request(
            request_id=self.next_request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        self.waiting_queue.append(request)
        self.next_request_id += 1

        self.metrics_manager.update_waiting_queue_num(len(self.waiting_queue))
        self.metrics_manager.register_request_arrival(request)

        return request.request_id

    def step(self):
        decode_result = None
        prefill_result = None

        if not self.waiting_queue and not self.active_requests:
            self.metrics_manager.update_waiting_queue_num(0)
            self.metrics_manager.update_active_requests_num(0)
            self.metrics_manager.set_no_work()
            return None

        prefill_batch_size = self._decide_prefill_batch_size()

        if prefill_batch_size > 0 and self.waiting_queue:
            prefill_result = self._prefill_step()
        else:
            decode_result = self._decode_step()

        self.metrics_manager.update_waiting_queue_num(len(self.waiting_queue))
        self.metrics_manager.update_active_requests_num(len(self.active_requests))

        if (
            self.config.enable_metrics
            and time.time() - self.metrics_manager.time >= METRIC_SHOW_PERIOD
        ):
            stage = "prefill" if prefill_result is not None else "decode"
            self.metrics_manager.show_metrics(stage)
            self.metrics_manager.time = time.time()

        return prefill_result if prefill_result is not None else decode_result

    def _decode_step(self):
        active = [req for req in self.active_requests if not req.is_finished]

        if not active:
            return None

        start_time = time.time()
        batch_result = self.engine.decode(active)
        end_time = time.time()

        self.metrics_manager.record_decode(active, batch_result, start_time, end_time)
        self.metrics_manager.update_active_requests_num(len(self.active_requests))

        return batch_result

    def _prefill_step(self):
        if not self.waiting_queue:
            return None

        batch_size = min(
            self._decide_prefill_batch_size(),
            len(self.waiting_queue),
            self.config.max_batch_size,
        )

        if batch_size <= 0:
            return None

        batch_requests = [self.waiting_queue.popleft() for _ in range(batch_size)]

        start_time = time.time()
        batch_result = self.engine.prefill(batch_requests)
        end_time = time.time()

        self.active_requests.extend(batch_requests)

        self.metrics_manager.record_prefill(batch_requests, batch_result, start_time, end_time)
        self.metrics_manager.update_waiting_queue_num(len(self.waiting_queue))
        self.metrics_manager.update_active_requests_num(len(self.active_requests))

        return batch_result

    def _decide_prefill_batch_size(self):
        num_active = len([r for r in self.active_requests if not r.is_finished])
        num_waiting = len(self.waiting_queue)

        if num_waiting == 0:
            return 0

        free_slots = max(0, self.config.max_batch_size - num_active)
        policy = self.config.policy_name

        if policy == "baseline":
            if num_active > 0:
                return 0
            return 1

        if policy == "fill_free_slots":
            if free_slots == 0:
                return 0
            return min(num_waiting, free_slots)

        if policy == "capped_aggressive":
            if free_slots == 0:
                return 0
            return min(num_waiting, free_slots, self.config.max_prefill_per_step)

        if policy == "timeout_hybrid":
            if num_active == 0:
                return min(num_waiting, self.config.max_batch_size, self.config.max_prefill_per_step)

            oldest_request = self.waiting_queue[0]
            if oldest_request.arrival_time is None:
                return 0

            wait_ms = (time.time() - oldest_request.arrival_time) * 1000.0

            if wait_ms >= self.config.prefill_timeout_ms and free_slots > 0:
                return min(num_waiting, free_slots, self.config.max_prefill_per_step)

            return 0

        return 0

    def get_finished_requests(self) -> List[Request]:
        finished = [req for req in self.active_requests if req.is_finished]
        self.active_requests = [req for req in self.active_requests if not req.is_finished]
        self.metrics_manager.update_active_requests_num(len(self.active_requests))
        return finished

    def get_metric_manager(self):
        return self.metrics_manager

    def clear(self):
        self.waiting_queue = deque()
        self.active_requests = []
        self.next_request_id = 0
        self.metrics_manager = MetricManager(enable_metrics=self.config.enable_metrics)
