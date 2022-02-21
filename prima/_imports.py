# A tacky tactic I picked up to reduce the number of imports to manually do.
# Can use autoflake expand_star_imports to get the specific imports.
# isort: off
from auto_all import end_all, start_all

start_all(globals())

# isort: on
import abc
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from decorator import decorator
from loguru import logger
from pydantic import BaseModel, Extra, Field, parse_obj_as, root_validator
from pydantic.generics import GenericModel

ROOT = Path(__file__).parent


CallAny = Callable[..., Any]
OptCany = Optional[CallAny]
TupleAny = Tuple[str, Any]
StateSP = TypeVar("StateSP")
SysParams = TypeVar("SysParams")
PolicySP = TypeVar("PolicySP")
DictAny = Dict[str, Any]


from prima.configuration.utils import (
    bound_norm_random,
    config_sim,
    time_step,
    env_trigger,
)

from toolz import keyfilter

# import cytoolz

from cytoolz.dicttoolz import itemfilter
from datetime import timedelta

# isort: off
end_all(globals())
# isort: on
