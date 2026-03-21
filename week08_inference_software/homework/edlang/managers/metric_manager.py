import time
import torch


METRIC_SHOW_PERIOD = 3.0


class MetricManager:
    def __init__(self, enable_metrics: bool = False):
        self.enable_metrics = enable_metrics

        self.waiting_queue_num = 0
        self.active_requests_num = 0

        self.throughput_tokens_per_second = 0.0
        self.ttft_ms = 0.0
        self.tpot_ms = 0.0
        self.rps = 0.0

        self.end_to_end_latency_ms = 0.0
        self.queue_wait_ms = 0.0
        self.gpu_utilization = 0.0
        self.memory_allocated_gb = 0.0
        self.memory_peak_gb = 0.0

        self.time = time.time()
        self.start_time = time.time()

        self.completed_requests = 0

        self.ttft_values = []
        self.tpot_values = []
        self.e2e_values = []
        self.queue_wait_values = []

        self.gpu_busy_time = 0.0
    
    def calculate_throughtput_tokens_per_second(self, tokens_num: int, time_s: float):
        if not self.enable_metrics:
            return 0.0

        if time_s <= 0:
            self.throughput_tokens_per_second = 0.0
        else:
            self.throughput_tokens_per_second = tokens_num / time_s

        return self.throughput_tokens_per_second

    def update_waiting_queue_num(self, num: int):
        self.waiting_queue_num = num

    def update_active_requests_num(self, num: int):
        self.active_requests_num = num

    def set_no_work(self):
        if not self.enable_metrics:
            return

        self.throughput_tokens_per_second = 0.0
        self.gpu_utilization = 0.0
    
    def register_request_arrival(self, request):
        if not self.enable_metrics:
            return
        request.arrival_time = time.time()
    
    def _update_gpu_and_memory(self, elapsed: float):
        if not self.enable_metrics:
            return

        self.gpu_busy_time += elapsed
        total_time = time.time() - self.start_time

        if total_time > 0:
            self.gpu_utilization = min(100.0, 100.0 * self.gpu_busy_time / total_time)

        if torch.cuda.is_available():
            self.memory_allocated_gb = torch.cuda.memory_allocated() / 1e9
            self.memory_peak_gb = torch.cuda.max_memory_allocated() / 1e9

    def record_prefill(self, requests, batch_result, start_time: float, end_time: float):
        if not self.enable_metrics:
            return

        elapsed = end_time - start_time
        tokens_num = sum(len(tokens) for tokens in batch_result.new_tokens)

        self.calculate_throughtput_tokens_per_second(tokens_num, elapsed)
        self._update_gpu_and_memory(elapsed)

        for request, new_tokens, finished in zip(requests, batch_result.new_tokens, batch_result.finished):
            if request.arrival_time is not None:
                queue_wait_ms = (start_time - request.arrival_time) * 1000.0
                self.queue_wait_values.append(queue_wait_ms)
                self.queue_wait_ms = sum(self.queue_wait_values) / len(self.queue_wait_values)

            if len(new_tokens) > 0 and request.first_token_time is None and request.arrival_time is not None:
                request.first_token_time = end_time
                ttft = (request.first_token_time - request.arrival_time) * 1000.0
                self.ttft_values.append(ttft)
                self.ttft_ms = sum(self.ttft_values) / len(self.ttft_values)

            if finished and request.finish_time is None and request.arrival_time is not None:
                request.finish_time = end_time
                self.completed_requests += 1

                e2e = (request.finish_time - request.arrival_time) * 1000.0
                self.e2e_values.append(e2e)
                self.end_to_end_latency_ms = sum(self.e2e_values) / len(self.e2e_values)

        total_elapsed = max(end_time - self.start_time, 1e-9)
        self.rps = self.completed_requests / total_elapsed

    def record_decode(self, requests, batch_result, start_time: float, end_time: float):
        if not self.enable_metrics:
            return

        elapsed = end_time - start_time
        tokens_num = sum(len(tokens) for tokens in batch_result.new_tokens)

        self.calculate_throughtput_tokens_per_second(tokens_num, elapsed)
        self._update_gpu_and_memory(elapsed)

        if tokens_num > 0:
            tpot = elapsed * 1000.0 / tokens_num
            self.tpot_values.append(tpot)
            self.tpot_ms = sum(self.tpot_values) / len(self.tpot_values)

        for request, finished in zip(requests, batch_result.finished):
            if finished and request.finish_time is None and request.arrival_time is not None:
                request.finish_time = end_time
                self.completed_requests += 1

                e2e = (request.finish_time - request.arrival_time) * 1000.0
                self.e2e_values.append(e2e)
                self.end_to_end_latency_ms = sum(self.e2e_values) / len(self.e2e_values)

        total_elapsed = max(end_time - self.start_time, 1e-9)
        self.rps = self.completed_requests / total_elapsed

    def show_metrics(self, stage: str):
        metrix_output = f"""
{stage}
- Throughput tokens per second: {self.throughput_tokens_per_second:.3f}
- TTFT: {self.ttft_ms:.3f} ms
- TPOT: {self.tpot_ms:.3f} ms
- RPS: {self.rps:.3f}
- Waiting queue number: {self.waiting_queue_num}
- Active requests number: {self.active_requests_num}
- End-to-End latency: {self.end_to_end_latency_ms:.3f} ms
- Queue wait time: {self.queue_wait_ms:.3f} ms
- GPU utilization: {self.gpu_utilization:.3f}%
- GPU memory allocated: {self.memory_allocated_gb:.3f} GB
- GPU memory peak: {self.memory_peak_gb:.3f} GB"""
        print("-" * 20 + metrix_output + "\n" + "-" * 20)
