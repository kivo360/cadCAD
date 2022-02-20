import abc
from ast import alias
import inspect
from pydantic import BaseModel, Field, parse_obj_as, Extra
from pydantic.generics import GenericModel
from typing import (
    Callable,
    Any,
    Literal,
    TypeVar,
    Generic,
    Type,
    Union,
    List,
    Dict,
    Tuple,
    Optional,
    TYPE_CHECKING,
    ClassVar,
)
from types import FunctionType
from inflection import parameterize, dasherize
from decorator import decorator
from loguru import logger
from typing import TypeVar, Generic
from enum import Enum
from pathlib import Path
from toolz.curried import curry

ROOT = Path(__file__).parent


CallAny = Callable[..., Any]
OptCany = Optional[CallAny]
TupleAny = Tuple[str, Any]
StateSP = TypeVar("StateSP")
SysParams = TypeVar("SysParams")
PolicySP = TypeVar("PolicySP")
DictAny = Dict[str, Any]


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

        args, kwargs = (
            pre(*args, fn=func, **kwargs) if pre is not None else (args, kwargs)
        )

        response = func(*args, **kwargs)
        response = post(response, fn=func) if post is not None else response

        return response

    inner = inner_decorator
    return inner


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

    def input_preprocess(self, *args, **kwargs) -> Tuple[List, DictAny]:
        if not len(args) >= 5:
            return args, kwargs

        space_type = type(self.state_space)
        args = list(args)
        args[0] = self.system_params
        args[2] = parse_obj_as(List[space_type], args[2])
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


class UpdateRules(BaseModel):
    policies: DictAny
    variables: DictAny

    def add_policies(self, policies: DictAny) -> None:
        self.policies.update(policies)

    def add_variables(self, variables: DictAny) -> None:
        self.variables.update(variables)


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


response = example_state_change({}, 0, [], {}, {})
logger.warning(response)
