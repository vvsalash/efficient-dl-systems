import json
import time
import torch
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class _ScheduleState:
    phase: str
    cycle_idx: int
    pos_in_cycle: int
    step: int


class Profile:
    def __init__(
        self,
        model: torch.nn.Module,
        name: str = "model",
        schedule: Optional[Dict[str, int]] = None,
    ) -> None:
        self.name_map = self._build_name_map(model, name)
        self.model = model

        schedule = schedule or {"wait": 0, "warmup": 0, "active": 1, "repeat": 1}
        self.wait = int(schedule.get("wait", 0))
        self.warmup = int(schedule.get("warmup", 0))
        self.active = int(schedule.get("active", 1))
        repeat = schedule.get("repeat", 1)
        self.repeat = None if repeat is None else int(repeat)

        assert self.wait >= 0 and self.warmup >= 0 and self.active >= 0
        assert (self.wait + self.warmup + self.active) > 0, "schedule cycle length must be > 0"
        if self.repeat is not None:
            assert self.repeat >= 0
        
        self._cycle_length = self.wait + self.warmup + self.active
        self._global_step = 0
        self._state = self._compute_state(self._global_step)

        self._spans: List[Dict[str, Any]] = []

        self._fw_inflight: Dict[torch.nn.Module, Dict[str, Any]] = {}
        self._bw_inflight: Dict[torch.nn.Module, Dict[str, Any]] = {}

        self.events: List[Dict[str, Any]] = []

        self._handles: List[Any] = []
        self._enabled = False

        self._cuda_zero: Optional[torch.cuda.Event] = None
        self._has_cuda = torch.cuda.is_available()

        self._pid = os.getpid()
        self._tid_forward = 0
        self._tid_backward = 1
    

    def _compute_state(self, step: int) -> _ScheduleState:
        if self.repeat is not None and self.repeat == 0:
            return _ScheduleState(
                phase="done",
                cycle_idx=0,
                pos_in_cycle=0,
                step=step
            )
        
        cycle_idx = step // self._cycle_length
        pos = step % self._cycle_length

        if self.repeat is not None and cycle_idx >= self.repeat:
            return _ScheduleState(
                phase="done",
                cycle_idx=cycle_idx,
                pos_in_cycle=pos,
                step=step
            )
        
        if pos < self.wait:
            phase = "wait"
        elif pos < self.wait + self.warmup:
            phase = "warmup"
        else:
            phase = "active"

        return _ScheduleState(
            phase=phase,
            cycle_idx=cycle_idx,
            pos_in_cycle=pos,
            step=step
        )
    

    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = module.__class__.__name__
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map


    def _is_leaf(self, module):
        return len(list(module.children())) == 0
    
    
    @property
    def phase(self) -> str:
        return self._state.phase
    

    @property
    def step_num(self) -> int:
        return self._state.step
    

    def _should_log(self) -> bool:
        return self._enabled and (self._state.phase == "active")
    

    def _new_cuda_event(self) -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)


    def _forward_pre_hook(self, module, inputs):
        if not self._should_log():
            return
        
        start_event = self._new_cuda_event()
        start_event.record()

        self._fw_inflight[module] = {
            "start": start_event
        }


    def _forward_post_hook(self, module, inputs, outputs):
        if not self._should_log():
            return
        
        inflight = self._fw_inflight.pop(module, None)
        if inflight is None:
            return
        
        end_event = self._new_cuda_event()
        end_event.record()

        self._spans.append(
            {
                "name": self.name_map.get(module, module.__class__.__name__),
                "kind": "forward",
                "step": self._state.step,
                "phase": self._state.phase,
                "pid": self._pid,
                "tid": self._tid_forward,
                "start": inflight["start"],
                "end": end_event,
            }
        )

    def _backward_pre_hook(self, module, grad_output):
        if not self._should_log():
            return
        
        start_event = self._new_cuda_event()
        start_event.record()

        self._bw_inflight[module] = {
            "start": start_event
        }

    def _backward_post_hook(self, module, grad_input, grad_output):
        if not self._should_log():
            return
        
        inflight = self._bw_inflight.pop(module, None)
        if inflight is None:
            return
        
        end_event = self._new_cuda_event()
        end_event.record()

        self._spans.append(
            {
                "name": self.name_map.get(module, module.__class__.__name__),
                "kind": "backward",
                "step": self._state.step,
                "phase": self._state.phase,
                "pid": self._pid,
                "tid": self._tid_backward,
                "start": inflight["start"],
                "end": end_event,
            }
        )

    def __enter__(self):
        self._enabled = True

        if self._has_cuda:
            self._cuda_zero = self._new_cuda_event()
            self._cuda_zero.record()
        else:
            self._cuda_zero = None
        

        for module in self.model.modules():
            if not self._is_leaf(module):
                continue

            self._handles.append(module.register_forward_pre_hook(self._forward_pre_hook))
            self._handles.append(module.register_forward_hook(self._forward_post_hook))
            self._handles.append(module.register_full_backward_pre_hook(self._backward_pre_hook))
            self._handles.append(module.register_full_backward_hook(self._backward_post_hook))
        
        return self

 
    def __exit__(self, type, value, traceback):
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()

        self._enabled = False

        self._fw_inflight.clear()
        self._bw_inflight.clear()


    def step(self):
        self._global_step += 1
        self._state = self._compute_state(self._global_step)
        return self._state
    

    def summary(self):
        print("Summary:")
        for s in self._spans[:20]:
            print({key: value for key, value in s.items() if key not in ("start", "end")})
        if len(self._spans) > 20:
            print(f"... total spans = {len(self._spans)}")
    

    def _finalize_to_trace_events(self) -> List[Dict[str, Any]]:
        if not self._spans:
            return []
        
        trace: List[Dict[str, Any]] = []

        if self._has_cuda:
            torch.cuda.synchronize()
            
            for span in self._spans:
                start_event = span["start"]
                end_event = span["end"]

                ts = self._cuda_zero.elapsed_time(start_event) * 1000.0
                dur = start_event.elapsed_time(end_event) * 1000.0

                trace.append(
                    {
                        "name": span["name"],
                        "ph": "X",
                        "ts": ts,
                        "dur": dur,
                        "pid": span["pid"],
                        "tid": span["tid"],
                        "args": {
                            "step": span["step"],
                            "phase": span["phase"],
                            "kind": span["kind"],
                        },
                    }
                )

        return trace

    def to_perfetto(self, path="trace.json"):
        trace_events = self._finalize_to_trace_events()
        self.events = trace_events

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "traceEvents": trace_events,
            "displayTimeUnit": "us",
        }
        with open(path, "w") as file:
            json.dump(payload, file)
