# %%
import numpy as np
import pandas as pd
import tabulate
from datetime import timedelta

from prima.configuration.utils import (
    bound_norm_random,
    config_sim,
    time_step,
    env_trigger,
)
from prima.configuration import Experiment


# %%
seeds = {
    "z": np.random.RandomState(1),
    "a": np.random.RandomState(2),
    "b": np.random.RandomState(3),
    "c": np.random.RandomState(4),
}

# %%
# Policies per Mechanism
def p1m1(_g, step, sH, s):
    return {"param1": 1}


def p2m1(_g, step, sH, s):
    return {"param1": 1, "param2": 4}


def p1m2(_g, step, sH, s):
    return {"param1": "a", "param2": 2}


def p2m2(_g, step, sH, s):
    return {"param1": "b", "param2": 4}


def p1m3(_g, step, sH, s):
    return {"param1": ["c"], "param2": np.array([10, 100])}


def p2m3(_g, step, sH, s):
    return {"param1": ["d"], "param2": np.array([20, 200])}


# %%
import sys

from decorator import decorator, decorate

# from loguru import logger
# from logging import StreamHandler


# logger.configure(
#     handlers=[
#         dict(
#             sink=sys.stderr,
#             format="[green]{time:YYYY-MM-DD at HH:mm:ss}[green] | [cyan]{level}[cyan] | {message}",
#             enqueue=True,
#             serialize=True,
#         ),
#     ],
# )


# %%


# %%
def _print_param(func, *args, **kwargs):
    print(kwargs)
    return func(*args, **kwargs)


@decorator
def print_parameters(func, *args, **kwargs):
    # logger.info("Parameters: {}".format(kw))
    return func(*args, **kwargs)


# %%

# Internal States per Mechanism
@print_parameters
def s1m1(_g, step, sH, s, _input):
    y = "s1"
    x = s["s1"] + 1
    return (y, x)


def s2m1(_g, step, sH, s, _input):
    y = "s2"
    x = _input["param2"]
    return (y, x)


def s1m2(_g, step, sH, s, _input):
    y = "s1"
    x = s["s1"] + 1
    return (y, x)


def s2m2(_g, step, sH, s, _input):
    y = "s2"
    x = _input["param2"]
    return (y, x)


def s1m3(_g, step, sH, s, _input):
    y = "s1"
    x = s["s1"] + 1
    return (y, x)


def s2m3(_g, step, sH, s, _input):
    y = "s2"
    x = _input["param2"]

    return (y, x)


def policies(_g, step, sH, s, _input):
    y = "policies"
    x = _input
    return (y, x)


# %%


# Exogenous States
proc_one_coef_A = 0.7
proc_one_coef_B = 1.3


def es3(_g, step, sH, s, _input):
    y = "s3"
    x = s["s3"] * bound_norm_random(seeds["a"], proc_one_coef_A, proc_one_coef_B)
    return (y, x)


def es4(_g, step, sH, s, _input):
    y = "s4"
    y = "s4"
    x = s["s4"] * bound_norm_random(seeds["b"], proc_one_coef_A, proc_one_coef_B)
    return (y, x)


def update_timestamp(_g, step, sH, s, _input):
    y = "timestamp"
    return y, time_step(
        dt_str=s[y],
        dt_format="%Y-%m-%d %H:%M:%S",
        _timedelta=timedelta(days=0, minutes=0, seconds=1),
    )


# %%


# Genesis States
genesis_states = {
    "s1": 0.0,
    "s2": 0.0,
    "s3": 1.0,
    "s4": 1.0,
    "timestamp": "2018-10-01 15:16:24",
}


# Environment Process
trigger_timestamps = [
    "2018-10-01 15:16:25",
    "2018-10-01 15:16:27",
    "2018-10-01 15:16:29",
]
env_processes = {
    "s3": [lambda _g, x: 5],
    "s4": env_trigger(3)(
        trigger_field="timestamp",
        trigger_vals=trigger_timestamps,
        funct_list=[lambda _g, x: 10],
    ),
}


# %%

partial_states = [
    {
        "policies": {"b1": p1m1, "b2": p2m1},
        "variables": {
            "s1": s1m1,
            "s2": s2m1,
            "s3": es3,
            "s4": es4,
            "timestamp": update_timestamp,
        },
    },
    {
        "policies": {"b1": p1m2, "b2": p2m2},
        "variables": {
            "s1": s1m2,
            "s2": s2m2,
            # "s3": es3p1,
            # "s4": es4p2,
        },
    },
    {
        "policies": {"b1": p1m3, "b2": p2m3},
        "variables": {
            "s1": s1m3,
            "s2": s2m3,
            # "s3": es3p1,
            # "s4": es4p2,
        },
    },
]


sim_config = config_sim(
    {
        "N": 2,
        "T": range(1),
    }
)

exp = Experiment()
exp.append_model(
    model_id="sys_model_A",
    sim_configs=sim_config,
    initial_state=genesis_states,
    env_processes=env_processes,
    partial_state_update_blocks=partial_states,
    policy_ops=[lambda a, b: a + b],
)

# %%
from prima.engine import ExecutionMode, ExecutionContext

exec_mode = ExecutionMode()
local_mode_ctx = ExecutionContext(context=exec_mode.single_mode)


# %%
from prima.engine import Executor

simulation = Executor(exec_context=local_mode_ctx, configs=exp.configs)

# %%
raw_system_events, tensor_field, sessions = simulation.execute()

simulation_result = pd.DataFrame(raw_system_events)

# %%
print(tabulate.tabulate(simulation_result, tablefmt="fancy_grid"))
