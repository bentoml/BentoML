from typing import Literal
import numpy as np
from pandas._libs.tslibs.nattype import NaTType
from pandas._libs.tslibs.offsets import BaseOffset
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import Frequency, Timezone

INVALID_FREQ_ERR_MSG: str
DIFFERENT_FREQ: str

class IncompatibleFrequency(ValueError): ...

def periodarr_to_dt64arr(periodarr: np.ndarray, freq: int) -> np.ndarray: ...
def period_asfreq_arr(
    arr: np.ndarray, freq1: int, freq2: int, end: bool
) -> np.ndarray: ...
def get_period_field_arr(field: str, arr: np.ndarray, freq: int) -> np.ndarray: ...
def from_ordinals(values: np.ndarray, freq: Frequency) -> np.ndarray: ...
def extract_ordinals(values: np.ndarray, freq: Frequency | int) -> np.ndarray: ...
def extract_freq(values: np.ndarray) -> BaseOffset: ...
def period_asfreq(ordinal: int, freq1: int, freq2: int, end: bool) -> int: ...
def period_ordinal(
    y: int, m: int, d: int, h: int, min: int, s: int, us: int, ps: int, freq: int
) -> int: ...
def freq_to_dtype_code(freq: BaseOffset) -> int: ...
def validate_end_alias(how: str) -> Literal["E", "S"]: ...

class Period:
    ordinal: int
    freq: BaseOffset
    def __new__(
        cls,
        value=...,
        freq=...,
        ordinal=...,
        year=...,
        month=...,
        quarter=...,
        day=...,
        hour=...,
        minute=...,
        second=...,
    ) -> Period | NaTType: ...
    @classmethod
    def now(cls, freq=...) -> Period: ...
    def strftime(self, fmt: str) -> str: ...
    def to_timestamp(
        self,
        freq: str | BaseOffset | None = ...,
        how: str = ...,
        tz: Timezone | None = ...,
    ) -> Timestamp: ...
    def asfreq(self, freq, how=...) -> Period: ...
    @property
    def freqstr(self) -> str: ...
    @property
    def is_leap_year(self) -> bool: ...
    @property
    def daysinmonth(self) -> int: ...
    @property
    def days_in_month(self) -> int: ...
    @property
    def qyear(self) -> int: ...
    @property
    def quarter(self) -> int: ...
    @property
    def day_of_year(self) -> int: ...
    @property
    def weekday(self) -> int: ...
    @property
    def day_of_week(self) -> int: ...
    @property
    def week(self) -> int: ...
    @property
    def weekofyear(self) -> int: ...
    @property
    def second(self) -> int: ...
    @property
    def minute(self) -> int: ...
    @property
    def hour(self) -> int: ...
    @property
    def day(self) -> int: ...
    @property
    def month(self) -> int: ...
    @property
    def year(self) -> int: ...
    @property
    def end_time(self) -> Timestamp: ...
    @property
    def start_time(self) -> Timestamp: ...
    def __sub__(self, other) -> Period | BaseOffset: ...
    def __add__(self, other) -> Period: ...
