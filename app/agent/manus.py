from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
# 搜索引擎本地化
from app.tool.baidu_search import BaiduSearch
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor


class Manus(ToolCallAgent):
    """A versatile general-purpose agent with support for both local and MCP tools."""
    # 你的 Manus 类是一个通用型智能体，继承自 ToolCallAgent，支持本地工具和 MCP（多通道处理）远程工具

    # 基本信息
    name: str = "Manus"
    description: str = "A versatile agent that can solve various tasks using multiple tools including MCP-based tools"

    # 系统提示词和下一步提示词
    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    # 最大观察长度和最大步数
    max_observe: int = 10000
    max_steps: int = 20

    # MCP 客户端，用于远程工具调用
    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # 可用工具集合（本地工具）
    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BaiduSearch(),
            BrowserUseTool(),
            StrReplaceEditor(),
            AskHuman(),
            Terminate(),
        )
    )

    # 特殊工具名称（如 Terminate）
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    # 浏览器上下文辅助对象
    browser_context_helper: Optional[BrowserContextHelper] = None

    # Track connected MCP servers
    # 已连接的 MCP 服务器
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command

    # 初始化标志
    _initialized: bool = False

    @model_validator(mode="after") #这是 Pydantic v2 的模型校验器（@model_validator(mode="after")），会在 Manus 实例创建并字段赋值后自动调用。
    def initialize_helper(self) -> "Manus":
        """Initialize basic components synchronously."""
        # 同步初始化 browser_context_helper，即为 Manus agent 创建一个浏览器上下文辅助对象，方便后续浏览器相关工具的使用。
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    @classmethod
    # 是 Manus 智能体的异步工厂方法，用于正确初始化一个 Manus 实例。
    # 使用例：manus = await Manus.create()
    async def create(cls, **kwargs) -> "Manus":
        """Factory method to create and properly initialize a Manus instance."""
        instance = cls(**kwargs) # 1. 创建 Manus 实例（会自动初始化本地工具等）
        await instance.initialize_mcp_servers() # 2. 异步初始化 MCP 服务器连接（远程工具）
        instance._initialized = True # 3. 标记已初始化
        return instance # 4. 返回已初始化的实例

    #用于初始化并连接所有在配置中声明的 MCP 服务器，以便 Manus 智能体可以动态使用远程工具
    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers."""
        # 遍历配置文件中所有的 MCP 服务器配置
        for server_id, server_config in config.mcp_config.servers.items():
            try:
                # 如果服务器类型是 sse（流式接口，通常是 http）
                if server_config.type == "sse":
                    if server_config.url:
                        # 连接 sse 类型的 MCP 服务器
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            f"Connected to MCP server {server_id} at {server_config.url}"
                        )
                # 如果服务器类型是 stdio（通过命令行进程通信）
                elif server_config.type == "stdio":
                    if server_config.command:
                        # 连接 stdio 类型的 MCP 服务器
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                        logger.info(
                            f"Connected to MCP server {server_id} using command {server_config.command}"
                        )
            except Exception as e:
                # 如果连接过程中出错，记录错误日志，但不中断整个流程
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: List[str] = None,
    ) -> None:
        """Connect to an MCP server and add its tools."""

    # 连接一个 MCP 服务器，并把该服务器的工具加入到 Manus 智能体的可用工具集合中。

    # 参数说明：
    # - server_url: 服务器的地址（如果是 stdio 类型，这里其实是命令行）
    # - server_id: 服务器的唯一标识（用于区分不同服务器）
    # - use_stdio: 是否用 stdio 方式连接（True=命令行，False=流式接口）
    # - stdio_args: 命令行参数列表（仅 stdio 模式下用）

    # 用法举例：
    # await self.connect_mcp_server("http://localhost:8000/mcp", "my_server")
    # await self.connect_mcp_server("python my_tool.py", "my_stdio", use_stdio=True, stdio_args=["--port", "9000"])

    # 简要说明：

    # 支持两种远程工具服务器连接方式：流式接口（SSE）和命令行（stdio）。
    # 连接成功后，会把该服务器的所有工具加入到 Manus 智能体的工具集合中，供后续自动调用。
    # connected_servers 用于追踪当前已连接的所有 MCP 服务器，方便后续管理和断开。

        if use_stdio:
            # 通过命令行方式连接 MCP 服务器
            await self.mcp_clients.connect_stdio(
                server_url, stdio_args or [], server_id
            )
            # 记录已连接服务器
            self.connected_servers[server_id or server_url] = server_url
        else:
            # 通过 SSE（流式接口）方式连接 MCP 服务器
            await self.mcp_clients.connect_sse(server_url, server_id)
            # 记录已连接服务器
            self.connected_servers[server_id or server_url] = server_url

        # 只把本次新连接服务器的工具加入到可用工具集合
        # Update available tools with only the new tools from this server
        new_tools = [
            tool for tool in self.mcp_clients.tools if tool.server_id == server_id
        ]
        self.available_tools.add_tools(*new_tools)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """Disconnect from an MCP server and remove its tools."""

        # 断开与 MCP 服务器的连接，并移除其工具。
        # 参数说明：
        # - server_id: 要断开的服务器ID。如果为空，则断开所有服务器。

        # 步骤说明：
        # 1. 调用 mcp_clients 的 disconnect 方法，断开与指定服务器的连接。
        # 2. 从 connected_servers 字典中移除该服务器（如果 server_id 为空则清空所有）。
        # 3. 重新构建可用工具集合，只保留本地工具和还在线的 MCP

        # 断开与 MCP 服务器的连接S
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            # 移除指定服务器
            self.connected_servers.pop(server_id, None)
        else:
            # 清空所有已连接服务器
            self.connected_servers.clear()

        # 只保留本地工具，移除所有 MCP 工具
        # Rebuild available tools without the disconnected server's tools
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]

        self.available_tools = ToolCollection(*base_tools)
        # 重新添加还在线的 MCP 工具
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self):
        """Clean up Manus agent resources."""
        # 清理 Manus agent 占用的资源。
        # 包括关闭浏览器和断开所有 MCP 服务器连接。

        # 如果有浏览器上下文辅助对象，先关闭浏览器，释放资源
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        # Disconnect from all MCP servers only if we were initialized
        # 只有在初始化过后才断开 MCP 服务器，避免重复操作
        if self._initialized:
            await self.disconnect_mcp_server() # 断开所有 MCP 服务器，并移除相关工具
            self._initialized = False # 标记为未初始化，防止重复清理

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""

        # 处理当前状态，并结合上下文决定下一步行动。
        # 主要功能：
        # 1. 如果还没初始化远程工具（MCP 服务器），先初始化。
        # 2. 检查最近几条消息是否用到了浏览器工具，如果用到了，就动态生成更适合浏览器场景的下一步提示词。
        # 3. 调用父类的 think 方法，继续决策。
        # 4. 恢复原始的提示词，保证下次循环不受影响。

        # 1. 初始化 MCP 服务器（只初始化一次）
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        # 2. 记录原始的 next_step_prompt，后面要恢复
        original_prompt = self.next_step_prompt

        # 3. 检查最近3条消息是否用到了浏览器工具
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        # 4. 如果最近用到了浏览器工具，就动态生成更适合的下一步提示词，类似于：
        # 当前浏览器状态：
        #     URL: https://www.baidu.com
        #     Title: 百度一下，你就知道
        #     2 tab(s) available
        # 页面上方内容： (200 pixels)
        # 页面下方内容： (500 pixels)
        # 结果信息：
        # 请根据当前网页内容，决定下一步操作。
        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        # 5. 调用父类的 think 方法，继续智能体的推理和决策
        result = await super().think()

        # Restore original prompt
        # 6. 恢复原始的 next_step_prompt，避免影响后续循环
        self.next_step_prompt = original_prompt

        return result
