# API Request Schema

Current server endpoints in [server/app.py](/hackathon/cuda/server/app.py):

## `GET /health`

Response:

```json
{
  "ok": "true"
}
```

## `POST /reset`

Request body:

```json
{
  "task": "softmax_m4096_n256",
  "seed": 0
}
```

Fields:
- `task`: optional string task id
- `seed`: optional integer seed

Response shape:

```json
{
  "observation": {
    "type": "reset",
    "task_id": "softmax_m4096_n256",
    "family": "softmax",
    "M": 4096,
    "N": 256,
    "dtype": "fp16",
    "mode": "self_improving",
    "tried_config_ids": [49, 53],
    "tried_latencies_ms": [0.0072, 0.0071],
    "best_so_far_ms": 0.0071,
    "steps_remaining": 6
  },
  "reward": 0.0,
  "done": false,
  "state": {
    "episode_id": "softmax_m4096_n256:0:1",
    "step_count": 0,
    "task_id": "softmax_m4096_n256",
    "family": "softmax",
    "mode": "self_improving",
    "tried_config_ids": [49, 53]
  },
  "info": {
    "best_so_far_ms": 0.0071,
    "oracle_best_ms": 0.0070,
    "current_regret": 0.01,
    "segment_key": "softmax_M4096_Nsmall_Dfp16",
    "training_status": "idle"
  }
}
```

## `POST /step`

Two request forms are supported.

Discrete config form:

```json
{
  "config_id": 0,
  "source": null
}
```

Continuous action form:

```json
{
  "x": [0.0, 0.0, 0.0],
  "source": null
}
```

Fields:
- `config_id`: optional integer config id
- `x`: optional 3-vector continuous action
- `source`: optional full kernel source string

One of `config_id` or `x` must be provided.

Response shape:

```json
{
  "observation": {
    "type": "step",
    "task_id": "softmax_m4096_n256",
    "family": "softmax",
    "mode": "self_improving",
    "tried_config_ids": [49, 53, 0],
    "tried_latencies_ms": [0.0072, 0.0071, 0.0070],
    "best_so_far_ms": 0.0070,
    "steps_remaining": 5,
    "last_trial": {
      "config_id": 0,
      "config": {
        "config_id": 0,
        "family": "softmax",
        "task_id": "softmax_m4096_n256",
        "block_size": 256,
        "num_warps": 1,
        "num_stages": 1
      },
      "latency_ms": 0.0070,
      "score": 4.95,
      "duplicate": false
    }
  },
  "reward": 0.02,
  "done": false,
  "state": {
    "episode_id": "softmax_m4096_n256:0:1",
    "step_count": 1,
    "task_id": "softmax_m4096_n256",
    "family": "softmax",
    "mode": "self_improving",
    "tried_config_ids": [49, 53, 0]
  },
  "info": {
    "best_so_far_ms": 0.0070,
    "oracle_best_ms": 0.0070,
    "current_regret": 0.0,
    "live_regret": 0.0,
    "segment_key": "softmax_M4096_Nsmall_Dfp16",
    "training_status": "idle",
    "adapter_version": "base"
  }
}
```

## `GET /state`

Response:

```json
{
  "episode_id": "softmax_m4096_n256:0:1",
  "step_count": 1,
  "task_id": "softmax_m4096_n256",
  "family": "softmax",
  "mode": "self_improving",
  "tried_config_ids": [49, 53, 0]
}
```
