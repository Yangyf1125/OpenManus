from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# 基础工具类
class BaseTool(ABC, BaseModel):
    name: str # 工具名称
    description: str # 工具描述
    parameters: Optional[dict] = None # 工具参数定义（可选，通常是 JSON Schema 格式）

    class Config: # 允许 Pydantic 字段类型为任意类型
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        # 让工具实例可以像函数一样被异步调用。
        # 实际会调用 execute 方法。
        # 用法举例：await tool(param1=xxx)
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        # 工具的核心执行逻辑，子类必须实现。
        # 每个具体工具都要重写这个方法，定义自己的功能。

    def to_param(self) -> Dict:
        """Convert tool to function call format."""
        # 把工具信息转换为函数调用格式（通常用于 LLM 工具调用接口）。
        # 返回一个包含工具名称、描述和参数定义的字典。

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

#工具结果类
class ToolResult(BaseModel):
    """Represents the result of a tool execution."""
    # 表示工具执行的结果。
    output: Any = Field(default=None) # 工具的输出内容，可以是任意类型（如字符串、字典等）
    error: Optional[str] = Field(default=None) # 错误信息（如果工具执行失败，可以在这里写明原因）
    base64_image: Optional[str] = Field(default=None) # 可选的 base64 编码图片（比如浏览器截图等）
    system: Optional[str] = Field(default=None) # 系统级信息（可选，用于特殊场景）

    class Config: # 允许字段为任意类型
        arbitrary_types_allowed = True

    def __bool__(self):  # 只要 ToolResult 的任意一个字段有值（非 None/非空），就认为它为 True
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
    # 支持 ToolResult + ToolResult 的合并操作。
    # 合并规则：
    # - output、error、system 字段：如果两个都有值就拼接，否则取有值的那个。
    # - base64_image 字段：如果两个都有值就报错（不能拼接图片），否则取有值的那个。
    # 用法举例
    #     r1 = ToolResult(output="A", error=None)
    #     r2 = ToolResult(output="B", error="fail")
    #     r3 = r1 + r2
    #     # r3.output == "AB"
    #     # r3.error == "fail"

        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field # 字符串拼接
                raise ValueError("Cannot combine tool results")
            return field or other_field # 只要有一个有值就用它

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self):
        # 如果有错误信息，优先显示错误，否则显示输出内容
        # result1 = ToolResult(output="执行成功")
        # print(result1)  # 输出：执行成功

        # result2 = ToolResult(error="参数错误")
        # print(result2)  # 输出：Error: 参数错误
        return f"Error: {self.error}" if self.error else self.output

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        # 返回一个新的 ToolResult 实例，指定字段用传入的值替换，其他字段保持不变。
        # 用法举例：
        #     r1 = ToolResult(output="ok", error=None)
        #     r2 = r1.replace(error="fail")
        #     # r2.output == "ok"
        #     # r2.error == "fail"


        # return self.copy(update=kwargs)  这一行原本就有
        # 先把原对象转成字典，再用新的字段覆盖，最后用 type(self) 创建新对象
        return type(self)(**{**self.dict(), **kwargs})


class CLIResult(ToolResult):
    """A ToolResult that can be rendered as a CLI output."""
    #用于表示可以在命令行（CLI）上友好展示的工具执行结果

class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""
    # 用于专门表示工具执行失败的结果
