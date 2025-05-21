from contextlib import AsyncExitStack
from typing import Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import ListToolsResult, TextContent

from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.tool_collection import ToolCollection


#是一个MCP 远程工具代理，用于通过 MCP 协议在客户端调用远程服务器上的工具
class MCPClientTool(BaseTool):
    """Represents a tool proxy that can be called on the MCP server from the client side."""

    session: Optional[ClientSession] = None  # 当前工具绑定的 MCP 客户端会话对象，用于和 MCP 服务器通信。
    server_id: str = ""  # Add server identifier  服务器标识符，用于区分不同的 MCP 服务器。
    original_name: str = ""  # 工具在远程服务器上的原始名称

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool by making a remote call to the MCP server."""
        # 如果没有有效的 MCP 会话，直接返回错误。
        if not self.session:
            return ToolResult(error="Not connected to MCP server")

        try:
            logger.info(f"Executing tool: {self.original_name}")
            # 将参数传递给远程 MCP 服务器的对应工具。
            result = await self.session.call_tool(self.original_name, kwargs)
            #遍历返回内容，将所有 TextContent 类型的文本拼接成字符串。
            content_str = ", ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            return ToolResult(output=content_str or "No output returned.")
        except Exception as e:
            return ToolResult(error=f"Error executing tool: {str(e)}")

# 多服务器 MCP 工具集合管理器 ,用于通过 Model Context Protocol (MCP) 管理和调用多个远程服务器上的工具
class MCPClients(ToolCollection):
    """
    A collection of tools that connects to multiple MCP servers and manages available tools through the Model Context Protocol.
    """

    sessions: Dict[str, ClientSession] = {} # 保存每个 server_id 对应的 MCP 客户端会话对象，负责与远程服务器通信。
    exit_stacks: Dict[str, AsyncExitStack] = {} # 保存每个 server_id 的异步退出栈，用于资源管理和自动清理。
    description: str = "MCP client tools for server interaction" # 工具集合的描述信息。

    def __init__(self):
        super().__init__()  # Initialize with empty tools list
        self.name = "mcp"  # Keep name for backward compatibility

###########################################################################################
    # def _normalize_server_id(self, server_id: str) -> str:
    #     """Normalize server ID to only contain allowed characters."""
    #     # Replace any non-alphanumeric characters (except _.-) with underscore
    #     import re
    #     return re.sub(r'[^a-zA-Z0-9_\.-]', '_', server_id)

###########################################################################################
    # 通过 SSE（Server-Sent Events）协议连接到一个 MCP 服务器，并自动完成资源管理和工具注册
    async def connect_sse(self, server_url: str, server_id: str = "") -> None:
        """Connect to an MCP server using SSE transport."""
        # 1.参数校验
        if not server_url:
            raise ValueError("Server URL is required.")
        # 2.确定 server_id
        server_id = server_id or server_url

        #server_id = self._normalize_server_id(server_id or server_url)

        # 3.断开已有连接
        # Always ensure clean disconnection before new connection
        if server_id in self.sessions:
            await self.disconnect(server_id)
        # 4.创建资源管理栈
        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack
        # 5.建立 SSE 连接并初始化会话
        streams_context = sse_client(url=server_url)
        streams = await exit_stack.enter_async_context(streams_context)
        session = await exit_stack.enter_async_context(ClientSession(*streams))
        self.sessions[server_id] = session
        # 6.初始化并注册远程工具
        await self._initialize_and_list_tools(server_id)
        # 典型用法：await mcp_clients.connect_sse("http://server:port", server_id="my_server")

    # 通过 stdio（标准输入输出）协议连接到一个 MCP 服务器，并自动完成资源管理和工具注册
    async def connect_stdio(
        self, command: str, args: List[str], server_id: str = ""
    ) -> None:
        """Connect to an MCP server using stdio transport."""

        # 1.参数校验
        if not command:
            raise ValueError("Server command is required.")
        # 2.确定 server_id
        server_id = self.server_id or command

        # Always ensure clean disconnection before new connection
        # 3.断开已有连接
        if server_id in self.sessions:
            await self.disconnect(server_id)

        # 4.创建资源管理栈
        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        # 5.建立 stdio 连接并初始化会话
        server_params = StdioServerParameters(command=command, args=args)
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await exit_stack.enter_async_context(ClientSession(read, write))
        self.sessions[server_id] = session

        # 初始化并注册远程工具
        await self._initialize_and_list_tools(server_id)
        # 典型用法：await mcp_clients.connect_stdio("python", ["-m", "my_server"], server_id="local_server")

    # 初始化 MCP 会话并注册远程服务器上的所有工具
    async def _initialize_and_list_tools(self, server_id: str) -> None:
        """Initialize session and populate tool map."""

        #1. 获取会话对象 ，检查并获取指定 server_id 的 MCP 会话对象。
        session = self.sessions.get(server_id)
        if not session:
            raise RuntimeError(f"Session not initialized for server {server_id}")

        # 2.初始化会话并获取工具列表 ，初始化会话（如握手、认证等）。调用远程服务器的 list_tools 方法，获取所有可用工具的描述。
        await session.initialize()
        response = await session.list_tools()

        # 3.为每个远程工具创建本地代理对象
        # Create proper tool objects for each server tool
        for tool in response.tools:
            original_name = tool.name
            # Always prefix with server_id to ensure uniqueness
            tool_name = f"mcp_{server_id}_{original_name}"

            server_tool = MCPClientTool(
                name=tool_name,
                description=tool.description,
                parameters=tool.inputSchema,
                session=session,
                server_id=server_id,
                original_name=original_name,
            )
            self.tool_map[tool_name] = server_tool

        # Update tools tuple
        # 4.更新本地工具集合
        self.tools = tuple(self.tool_map.values())
        logger.info(
            f"Connected to server {server_id} with tools: {[tool.name for tool in response.tools]}"
        )
        #总结：该方法让你自动同步远程 MCP 服务器上的所有工具到本地，并为每个工具创建唯一的本地代理对象，方便后续统一调用和管理。

    async def list_tools(self) -> ListToolsResult:
        """List all available tools."""
        tools_result = ListToolsResult(tools=[])
        for session in self.sessions.values():
            response = await session.list_tools()
            tools_result.tools += response.tools
        return tools_result

    # 用于断开与指定 MCP 服务器或所有 MCP 服务器的连接，并清理相关资源
    async def disconnect(self, server_id: str = "") -> None:
        """Disconnect from a specific MCP server or all servers if no server_id provided."""
        #1. 断开指定服务器
        if server_id:
            if server_id in self.sessions:
                try:
                    # 获取对应的 exit_stack（资源管理栈）
                    exit_stack = self.exit_stacks.get(server_id)

                    # Close the exit stack which will handle session cleanup
                    # 自动关闭所有异步资源（如会话、流等）。
                    if exit_stack:
                        try:
                            await exit_stack.aclose()
                        except RuntimeError as e:
                            if "cancel scope" in str(e).lower():
                                logger.warning(
                                    f"Cancel scope error during disconnect from {server_id}, continuing with cleanup: {e}"
                                )
                            else:
                                raise

                    # Clean up references
                    #从 sessions 和 exit_stacks 字典中移除该服务器的引用
                    self.sessions.pop(server_id, None)
                    self.exit_stacks.pop(server_id, None)

                    # Remove tools associated with this server
                    # 从 tool_map 中移除所有属于该服务器的工具，并刷新 self.tools。
                    self.tool_map = {
                        k: v
                        for k, v in self.tool_map.items()
                        if v.server_id != server_id
                    }
                    self.tools = tuple(self.tool_map.values())
                    logger.info(f"Disconnected from MCP server {server_id}")
                except Exception as e:
                    logger.error(f"Error disconnecting from server {server_id}: {e}")
        else:
            # Disconnect from all servers in a deterministic order
            # 如果没有传入 server_id，则遍历所有已连接服务器，依次断开
            for sid in sorted(list(self.sessions.keys())):
                await self.disconnect(sid)
            self.tool_map = {}
            self.tools = tuple()
            logger.info("Disconnected from all MCP servers")
