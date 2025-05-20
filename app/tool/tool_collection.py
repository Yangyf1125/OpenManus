"""Collection classes for managing multiple tools."""
from typing import Any, Dict, List

from app.exceptions import ToolError
from app.logger import logger
from app.tool.base import BaseTool, ToolFailure, ToolResult


# 工具集合类，用于管理和操作多个工具（BaseTool）
class ToolCollection:
    """A collection of defined tools."""

    class Config:
        arbitrary_types_allowed = True  # 允许任意类型字段（兼容 Pydantic）

    def __init__(self, *tools: BaseTool):
        # 初始化时可以传入任意数量的工具
        self.tools = tools # 工具元组
        self.tool_map = {tool.name: tool for tool in tools} # 名称到工具的映射，便于查找

    def __iter__(self):
        # 让 ToolCollection 可以像列表一样被遍历
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        # 把所有工具的信息转成 LLM 可识别的参数格式（如 OpenAI function call）
        return [tool.to_param() for tool in self.tools]

    # 用于按名称调用集合中的某个工具并返回执行结果
    # 用法类似result = await tool_collection.execute(name="python_execute", tool_input={"code": "print(1+1)"})
    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] = None #tool_input是参数
    ) -> ToolResult:
        # 根据工具名称查找工具对象
        tool = self.tool_map.get(name)
        if not tool:
             # 如果找不到，返回失败结果
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            # 调用工具（支持异步），传入参数
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            # 如果工具执行时抛出 ToolError，返回失败结果
            return ToolFailure(error=e.message)

    # 依次执行集合中的所有工具，并返回每个工具的执行结果列表
    async def execute_all(self) -> List[ToolResult]:
        """Execute all tools in the collection sequentially."""
        results = []
        for tool in self.tools:
            try:
                # 调用每个工具（不带参数），并收集结果
                result = await tool()
                results.append(result)
            except ToolError as e:
                # 如果工具执行出错，收集失败结果
                results.append(ToolFailure(error=e.message))
        return results

    def get_tool(self, name: str) -> BaseTool:
        return self.tool_map.get(name)
        # 根据工具名称返回对应的工具对象。
        # 如果找不到，返回 None。
        # 用法举例：
        #     tool = tool_collection.get_tool("python_execute")
        #     if tool:
        #         await tool(code="print(1+1)")

    def add_tool(self, tool: BaseTool):
        """Add a single tool to the collection.

        If a tool with the same name already exists, it will be skipped and a warning will be logged.
        """
        # 向集合中添加一个工具。
        # 如果已有同名工具，则跳过并记录警告日志。
        # 返回自身，方便链式调用。
        # 用法： collection.add_tool(PythonExecute())
        if tool.name in self.tool_map:
            logger.warning(f"Tool {tool.name} already exists in collection, skipping")
            return self  # 已有同名工具，跳过

        self.tools += (tool,) # 把新工具加到元组末尾
        self.tool_map[tool.name] = tool # 更新名称到工具的映射
        return self # 支持链式调用

    def add_tools(self, *tools: BaseTool):
        """Add multiple tools to the collection.

        If any tool has a name conflict with an existing tool, it will be skipped and a warning will be logged.
        """
        # 批量添加多个工具到集合中。
        # 如果有同名工具，会跳过并记录警告日志。
        # 返回自身，方便链式调用。
        # 用法举例：
        #     collection.add_tools(PythonExecute(), BrowserUseTool())


        for tool in tools:
            self.add_tool(tool) # 复用单个添加方法，自动去重
        return self
