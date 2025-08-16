"""
该模块基于 Pydantic v2 的验证机制，提供一组通用的转换器与检查器工厂函数，
以 BeforeValidator / AfterValidator 装饰器形式封装，可便捷地应用于字段验证。
还扩展了 BaseModel，支持在空值时自动回退到字段默认值。

主要功能：
- 格式化 ValidationError 供前端或日志使用
- 构建通用的字段转换器（单值 / 列表）
- 构建通用的字段检查器（单值 / 列表）
- 扩展 BaseModel，提供默认值回退逻辑
"""

from functools import partial
from typing import Any, Callable, Iterable, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
)
from pydantic_core import PydanticCustomError, PydanticUndefined


def format_validation_error(exc: ValidationError) -> list[dict[str, str]]:
    """
    将 Pydantic 的 ValidationError 转换为结构化的错误列表。

    Args:
        exc (ValidationError): Pydantic 抛出的验证异常对象。

    Returns:
        list[dict[str, str]]: 每个错误包含字段路径、提示信息、错误类型、原始输入值。
    """
    details = []
    for err in exc.errors():
        details.append(
            {
                "field": ".".join(map(str, err["loc"])),
                "message": err["msg"],
                "type": err["type"],
                "input": err["input"],
            }
        )
    return details


def convert(
    func: Callable[..., Any], ignore_none: bool = True, **kwargs: Any
) -> BeforeValidator:
    """
    构造一个单值转换器，在字段值进入验证流程前执行自定义转换逻辑。

    Args:
        func (Callable): 用户自定义的转换函数。
        ignore_none (bool): 当值为 None 时是否跳过转换。
        **kwargs: 传递给 func 的额外参数。

    Returns:
        BeforeValidator: 可应用于 Pydantic 字段的转换器。
    """
    partial_func = partial(func, **kwargs)

    def validator(value: Any) -> Any:
        if ignore_none and value is None:
            return value
        try:
            return partial_func(value)
        except Exception:
            return value

    return BeforeValidator(validator)


def convert_list(
    func: Callable[..., Any], ignore_none: bool = True, **kwargs: Any
) -> BeforeValidator:
    """
    构造一个列表转换器，在字段值进入验证流程前对列表内每个元素进行转换。

    Args:
        func (Callable): 用户自定义的转换函数。
        ignore_none (bool): 当值为 None 时是否跳过转换。
        **kwargs: 传递给 func 的额外参数。

    Returns:
        BeforeValidator: 可应用于 Pydantic 字段（Iterable 类型）的转换器。
    """
    partial_func = partial(func, **kwargs)

    def validator(values: Iterable[Any]) -> Iterable[Any]:
        if ignore_none and values is None:
            return values
        try:
            return [partial_func(value) for value in values]
        except Exception:
            return values

    return BeforeValidator(validator)


def check(
    func: Callable[..., Any],
    ignore_none: bool = True,
    check_result: bool = False,
    description: str | None = None,
    **kwargs: Any,
) -> AfterValidator:
    """
    构造一个单值检查器，在字段值通过基础验证后执行自定义检查逻辑。

    Args:
        func (Callable): 检查函数。
        ignore_none (bool): 当值为 None 时是否跳过检查。
        check_result (bool): 是否检查函数返回值为真，若为假则报错。
        description (str | None): 错误提示语，可覆盖默认提示。
        **kwargs: 传递给 func 的额外参数。

    Returns:
        AfterValidator: 可应用于 Pydantic 字段的检查器。
    """
    partial_func = partial(func, **kwargs)

    def validator(value: Any) -> Any:
        if ignore_none and value is None:
            return value
        try:
            result = partial_func(value)
            if check_result and not result:
                raise ValueError(
                    description if description else f"{func.__name__} failed"
                )
        except Exception as ex:
            raise PydanticCustomError("Check failed", "{str_ex}", {"str_ex": str(ex)})
        return value

    return AfterValidator(validator)


def check_list(
    func: Callable[..., Any],
    ignore_none: bool = True,
    check_result: bool = False,
    description: str | None = None,
    **kwargs: Any,
) -> AfterValidator:
    """
    构造一个列表检查器，对 Iterable 中的每个值执行自定义检查。

    Args:
        func (Callable): 检查函数。
        ignore_none (bool): 当值为 None 时是否跳过检查。
        check_result (bool): 是否检查函数返回值为真，若为假则报错。
        description (str | None): 错误提示语，可覆盖默认提示。
        **kwargs: 传递给 func 的额外参数。

    Returns:
        AfterValidator: 可应用于 Pydantic Iterable 字段的检查器。
    """
    partial_func = partial(func, **kwargs)

    def validator(values: Iterable[Any]) -> Iterable[Any]:
        if ignore_none and values is None:
            return values
        try:
            for value in values:
                result = partial_func(value)
                if check_result and not result:
                    raise ValueError(
                        description if description else f"{func.__name__} failed"
                    )
        except Exception as ex:
            raise PydanticCustomError("Check failed", "{str_ex}", {"str_ex": str(ex)})
        return values

    return AfterValidator(validator)


class BaseModelEx(BaseModel):
    """
    扩展版 BaseModel：
    - 在字段值为空（空序列、空集合、空字符串、None）时，自动回退到字段默认值（若有）
    - 支持在配置中控制 validate_default，决定默认值是否经过验证
    """

    @field_validator("*", mode="wrap")
    @classmethod
    def use_default_value(
        cls: type[Self],
        value: Any,
        validator: ValidatorFunctionWrapHandler,
        info: ValidationInfo,
        /,
    ) -> Any:
        if value in ([], {}, (), set(), "", None):
            if info and info.field_name:
                field_info = cls.model_fields.get(info.field_name)
                if field_info:
                    default = field_info.get_default(call_default_factory=True)
                    if default is not PydanticUndefined:
                        if info.config and info.config.get("validate_default"):
                            return validator(default)
                        return default
        return validator(value)
