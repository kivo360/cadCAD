# %%
from prima._imports import *
from prima import init, ParitalState, Experiment
from prima.engine import Executor, ExecutionMode, ExecutionContext
import numpy as np
import pandas as pd
import tabulate

# %%
seeds = {
    "z": np.random.RandomState(1),
    "a": np.random.RandomState(2),
    "b": np.random.RandomState(3),
    "c": np.random.RandomState(4),
}

# %%
class GenesisState(BaseModel):
    hello: int = 1
    state_1: float = 0.0
    state_2: float = 0.0
    state_3: float = 1.0
    state_4: float = 1.0

    timestamp: str = "2018-10-01 15:16:24"


class ModelParams(BaseModel):
    alpha: int = [1, 10]
    beta: int = [2, 8]
    gamma: int = [3, 8]
    omega: int = [4, 2, 3]


class SystemParams(BaseModel):
    length: List[int] = Field([1], alias="T")
    run_count: int = Field(1, alias="N")
    model_params: ModelParams = Field(ModelParams(), alias="M")

    def dict(self, **kwargs) -> DictAny:
        kwargs.update(dict(by_alias=True))
        return super().dict(**kwargs)


class PolicyVarSpace(BaseModel):
    param1: Optional[List[int]] = list(range(4, 10))
    param2: Optional[List[int]] = list(range(4, 10))

    class Config:
        Extra = Extra.allow


# %%
model = init(GenesisState(), SystemParams(length=range(1)), PolicyVarSpace())
model.model_id = "sys_model_A"

# %%
@model.transition("state_1", "state_change_1")
def state_change_1(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    statespace: GenesisState,
    aggregate: DictAny,
    **kwargs,
) -> Tuple[str, Any]:
    statespace.state_1 += 1
    return statespace.state_1


@model.transition("state_2", "state_change_2")
def state_change_2(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    statespace: GenesisState,
    aggregate: DictAny,
    **kwargs,
) -> Tuple[str, Any]:
    statespace.state_2 += 1
    return statespace.state_2


@model.transition("state_3", "state_change_3")
def state_change_3(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    statespace: GenesisState,
    aggregate: DictAny,
    **kwargs,
) -> Tuple[str, Any]:
    statespace.state_3 += 1
    return statespace.state_3


@model.transition("state_4", "state_change_4")
def state_change_4(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    statespace: GenesisState,
    aggregate: DictAny,
    **kwargs,
) -> Tuple[str, Any]:
    statespace.state_4 += 1
    return statespace.state_4


@model.transition("timestamp", "update_timestamp")
def update_timestamp(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    statespace: GenesisState,
    aggregate: DictAny,
    **kwargs,
) -> Tuple[str, Any]:
    statespace.state_3 += 1
    return statespace.state_3


# conf =


# model.


# conf
# model.config
model.simulation_config

# %%


@model.policy("policy_1", "policy_1")
def policy_1(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    prior_state: GenesisState,
) -> DictAny:
    return {"param1": 1}


@model.policy("policy_1", "policy_2")
def policy_2(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    prior_state: GenesisState,
) -> DictAny:
    return {"param1": 1, "param2": 4}


@model.policy("policy_1", "policy_2_1")
def policy_2_1(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    prior_state: GenesisState,
) -> DictAny:
    return {"param1": "a", "param2": 2}


@model.policy("policy_1", "policy_2_2")
def policy_2_2(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    prior_state: GenesisState,
) -> DictAny:
    return {"param1": "b", "param2": 4}


@model.policy("policy_1", "policy_1_3")
def policy_1_3(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    prior_state: GenesisState,
) -> DictAny:
    return {"param1": ["c"], "param2": np.array([10, 100])}


@model.policy("policy_1", "policy_2_3")
def policy_2_3(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    prior_state: GenesisState,
) -> DictAny:
    return {"param1": ["d"], "param2": np.array([20, 200])}


# %%
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


# def update_timestamp(_g, step, sH, s, _input):
#     y = "timestamp"
#     return y, time_step(
#         dt_str=s[y],
#         dt_format="%Y-%m-%d %H:%M:%S",
#         _timedelta=timedelta(days=0, minutes=0, seconds=1),
#     )


# %%
@model.transition("state_3", "edro_state_3")
def edro_state_3(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    statespace: GenesisState,
    aggregate: DictAny,
    **kwargs,
) -> Tuple[str, Any]:
    statespace.state_3 = statespace.state_3 * bound_norm_random(
        seeds["a"], proc_one_coef_A, proc_one_coef_B
    )
    return statespace.state_3


# %%
@model.transition("state_4", "edro_state_4")
def edro_state_4(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    statespace: GenesisState,
    aggregate: DictAny,
    **kwargs,
) -> Tuple[str, Any]:
    statespace.state_4 = statespace.state_4 * bound_norm_random(
        seeds["c"], proc_one_coef_A, proc_one_coef_B
    )
    return statespace.state_4


# %%
model.add_partial(
    part_state=ParitalState(
        policies={"b1": policy_1, "b2": policy_2},
        variables={
            "state_1": state_change_1,
            "state_2": state_change_2,
            "state_3": state_change_3,
            "state_4": state_change_4,
            "timestamp": update_timestamp,
        },
    )
)

# %%
model.add_partial(
    part_state=ParitalState(
        policies={"b1": policy_2_1, "b2": policy_2_2},
        variables={
            "state_1": state_change_1,
            "state_2": state_change_2,
        },
    )
)

# %%
model.add_partial(
    part_state=ParitalState(
        policies={"b1": policy_1_3, "b2": policy_2_3},
        variables={
            "state_1": state_change_3,
            "state_2": state_change_4,
        },
    )
)

# %%
model.partials

# %%
model.get_model()

# %%
experiment = Experiment()

# %%
experiment.append_model(**model.get_model())


# %%
exec_mode = ExecutionMode()
local_mode_ctx = ExecutionContext(context=exec_mode.single_mode)


# %%
from prima.engine import Executor

simulation = Executor(exec_context=local_mode_ctx, configs=experiment.configs)

# %%
# %%
raw_system_events, tensor_field, sessions = simulation.execute()

simulation_result = pd.DataFrame(raw_system_events)
# pd.DataFrame(raw_system_events)
# logger.info(raw_system_events)
print(tabulate.tabulate(simulation_result, headers="keys", tablefmt="psql"))
# %%
