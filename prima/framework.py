from ._imports import *


def filter_poly_val(item: Tuple[str, Any]) -> bool:
    key, value = item
    if key in ["policies", "variables"] and isinstance(value, dict):
        return True
    return False


class FoundationBase(BaseModel, abc.ABC):

    __genisis__: ClassVar[DictAny] = {}
    __transitions__: ClassVar[DictAny] = {}
    __policies__: ClassVar[DictAny] = {}
    __processes__: ClassVar[DictAny] = {}
    __partial_blocks__: ClassVar[DictAny] = {}

    class Config:
        extra = Extra.allow


def create_decorator(
    *,
    pre: OptCany = None,
    post: OptCany = None,
) -> Callable[..., Tuple[str, Any]]:
    @decorator
    def inner_decorator(func, *args, **kwargs):
        logger.info("Decorating {}".format(func.__name__))
        args, kwargs = pre(*args, **kwargs) if pre is not None else (args, kwargs)

        response = func(*args, **kwargs)
        response = post(response, fn=func) if post is not None else response

        return response

    inner = inner_decorator
    return inner


class ParitalState(BaseModel):
    policies: DictAny = {}
    variables: DictAny = {}

    def add_policies(self, policies: DictAny) -> None:
        self.policies.update(policies)

    def add_variables(self, variables: DictAny) -> None:
        self.variables.update(variables)

    # Set attribute like a dict
    def __add__(self, other: dict):
        if isinstance(other, dict) and other.keys() in {"policies", "variables"}:
            self.policies.update(other.get("policies", {}))
            self.variables.update(other.get("variables", {}))
            return self
        return super().__add__(other)


class Foundation(GenericModel, FoundationBase, Generic[StateSP, SysParams, PolicySP]):
    if TYPE_CHECKING:
        __genisis__: ClassVar[DictAny] = {}
        __transitions__: ClassVar[DictAny] = {}
        __policies__: ClassVar[DictAny] = {}
        # Environment processes
        __processes__: ClassVar[DictAny] = {}
        __partial_blocks__: ClassVar[DictAny] = {}

    model_id: str = "foundation"
    state_space: StateSP
    system_params: SysParams
    policy_space: PolicySP

    partial_states: List[ParitalState] = []

    @root_validator
    def extract_initial_states(cls, values: DictAny) -> DictAny:
        # cls.__genisis__ = values.get("state_space", {})
        return values

    def input_preprocess(self, *args, **kwargs) -> Tuple[List, DictAny]:
        if not len(args) >= 5:
            return args, kwargs
        logger.info(args[2])
        space_type = type(self.state_space)
        args = list(args)
        args[0] = self.system_params
        # args[2] = parse_obj_as(List[space_type], args[2][0])
        args[3] = space_type(**args[3])
        return args, kwargs

    def transition(
        self, state_name: str, fn_name: str, **kwargs
    ) -> Callable[
        ...,
        Callable[[SysParams, int, List[StateSP], StateSP, DictAny], Tuple[str, Any]],
    ]:
        def pre_process(*args, fn: CallAny = None, **kwargs) -> Tuple[List, DictAny]:
            args, kwargs = self.input_preprocess(*args, **kwargs)
            return tuple(args), kwargs

        def post_process(
            response: Any, *args, fn: CallAny, **kwargs
        ) -> Tuple[str, Any]:
            return (state_name, response)

        decorated = create_decorator(post=post_process, pre=pre_process)
        self.__transitions__[fn_name] = decorated
        return decorated

    def policy(self, policy: str, fn_name: str, **kwargs) -> None:
        def pre_process(*args, **kwargs) -> Tuple[List, DictAny]:
            args, kwargs = self.input_preprocess(*args, **kwargs)
            # args[4] = type(self.state_space)(**args[4])
            return tuple(args), kwargs

        def post_process(response: Any, *args, **kwargs) -> Tuple[str, Any]:
            if isinstance(response, BaseModel):
                return response.dict(exclude_none=True, exclude_defaults=True)
            return response

        decorated = create_decorator(post=post_process, pre=pre_process)
        self.__policies__[fn_name] = decorated
        return decorated

    # A context manager for a block of code
    def __enter__(self):
        return self

    # exit context
    def __exit__(self, exc_type, exc_value, traceback):
        self.leave_context()

    def leave_context(self):
        pass

    @property
    def state_dict(self):
        return self.state_space.dict(exclude_none=True)

    @property
    def config(self):
        return self.system_params.dict(exclude_none=True, by_alias=True)

    @property
    def simulation_config(self):
        return config_sim(self.config)

    @property
    def partials(self):
        return [part.dict() for part in self.partial_states]

    @property
    def processes(self):
        """The processes property."""
        # Using placeholder to get something working.
        trigger_timestamps = [
            "2018-10-01 15:16:25",
            "2018-10-01 15:16:27",
            "2018-10-01 15:16:29",
        ]

        return {
            "s3": [lambda _g, x: 5],
            "s4": env_trigger(3)(
                trigger_field="timestamp",
                trigger_vals=trigger_timestamps,
                funct_list=[lambda _g, x: 10],
            ),
        }

    def add_partial(self, part_state: ParitalState):
        self.partial_states.append(part_state)

    def get_model(self):
        return dict(
            model_id=self.model_id,
            sim_configs=self.config,
            initial_state=self.state_dict,
            env_processes=self.processes,
            partial_state_update_blocks=self.partials,
            policy_ops=[lambda a, b: a + b],
        )


def create_foundation(
    state_space: StateSP, system_params: SysParams, policy_space: PolicySP
) -> Foundation[StateSP, SysParams, PolicySP]:
    return Foundation[StateSP, SysParams, PolicySP](
        state_space=state_space, system_params=system_params, policy_space=policy_space
    )


class GenesisState(BaseModel):
    hello: int = 1


class ModelParams(BaseModel):
    alpha: int = 1
    beta: int = 2
    gamma: int = 3
    omega: int = 4


class SystemParams(BaseModel):
    length: int = Field(1, alias="T")
    run_count: int = Field(1, alias="N")
    model_params: ModelParams = Field(ModelParams(), alias="M")

    def dict(self, **kwargs) -> DictAny:
        kwargs.update(dict(by_alias=True))
        return super().dict(kwargs)


class PolicyVarSpace(BaseModel):
    param1: Optional[int] = None
    param2: Optional[int] = None

    class Config:
        Extra = Extra.allow


foundation = create_foundation(GenesisState(), SystemParams(), PolicyVarSpace())


# NOTE: Making sure this function can have parameters. Append func to an internal class that handles registry.
@foundation.transition("bonding", "example_state_change")
def example_state_change(
    params: ModelParams,
    substep: int,
    history: List[GenesisState],
    statespace: GenesisState,
    aggregate: DictAny,
    **kwargs,
) -> Tuple[str, Any]:

    return statespace.hello
