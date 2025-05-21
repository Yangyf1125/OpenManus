import asyncio
import json
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import (TOOL_CHOICE_TYPE, AgentState, Message, ToolCall,
                        ToolChoice)
from app.tool import CreateChatCompletion, Terminate, ToolCollection

#定义了一个常量字符串，用于提示“需要工具调用，但没有提供任何工具调用。
#它通常在 ToolCallAgent 的 act 方法中被用来在必须有工具调用时却没有时抛出异常：
TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""
       #用于处理工具/函数调用的智能体基类，提供更高级的抽象

    # 增加了工具相关的属性（如 available_tools, tool_calls, tool_choices 等）。
    # 支持多种工具调用模式和特殊工具处理。
    # 默认最大步数提升到 30 步，适合更复杂的多轮工具调用场景。

    # 代理名称和描述
    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    # 系统提示词和下一步提示词
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # 可用工具集合，包含了如 CreateChatCompletion（对话生成工具）、Terminate（终止/结束工具） 等工具
    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )

    # 工具调用模式（自动/必须/禁止等）
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore

    # 特殊工具名称列表（如 Terminate），用于特殊处理
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    # 当前要执行的工具调用列表
    tool_calls: List[ToolCall] = Field(default_factory=list)

    # 当前步骤的 base64 图片（如有）
    _current_base64_image: Optional[str] = None

    # 最大执行步数 最大观察步数（可选）
    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
            # 处理当前状态并决定下一步行动（是否需要工具调用）。
            # 返回值为 True 表示需要执行 act，False 表示只思考不行动。

        # 如果有下一步提示词，把它作为用户消息加入对话历史
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # 调用 LLM，获取带工具选项的回复
            # Get response with tool options
            #返回值：
                # response.content：模型生成的回复内容
                # response.tool_calls：模型建议要调用的工具及参数
            response = await self.llm.ask_tool(
                messages=self.messages, # 当前所有对话消息（历史+新消息）
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                #把 agent 当前可用的所有工具，转换成 LLM（大语言模型）可以识别和调用的参数格式，传递给 LLM。
                tools=self.available_tools.to_params(), # 可用工具的参数列表，
                tool_choice=self.tool_choices, # 工具调用模式（AUTO/REQUIRED/NONE）
            )
        except ValueError:
            raise
        except Exception as e:
            # 如果是 Token 限制导致的错误，记录日志并终止
            # Check if this is a RetryError containing TokenLimitExceeded
            # 检查异常的 __cause__ 属性是否是 TokenLimitExceeded 类型
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                # 把错误信息作为 assistant 消息加入 memory
                self.memory.add_message(
                    Message.assistant_message(
                        f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        # 解析 LLM 返回的工具调用和内容
        # 解析大语言模型（LLM）返回的结果，分别提取工具调用列表和回复内容，并赋值给 agent 的属性。
        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        # Log response info
        # 日志记录
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # 如果工具调用模式为 NONE，不允许用工具
            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # 创建并添加 assistant 消息
            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            # 如果工具调用模式为 REQUIRED，但没有工具调用，交给 act 处理
            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # 如果工具调用模式为 AUTO，没有工具调用但有内容，继续
            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:

                ################################################################
                #self.state = AgentState.FINISHED
                #######################################################################

                return bool(content)

            # 有工具调用则返回 True，否则 False
            return bool(self.tool_calls)

        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
            # 执行工具调用并处理结果。
            # Returns:
            #     str: 所有工具调用结果的拼接字符串。

        # 如果没有工具调用
        if not self.tool_calls:
            # 如果要求必须有工具调用，但实际没有，抛出异常
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            # 否则返回最后一条消息的内容，或默认提示
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        # 依次执行每一个工具调用
        # 格式例子：
        #self.tool_calls = [
        #            ToolCall(id="1", function=ToolFunction(name="CreateChatCompletion", arguments='{"text": "你好"}')),
        #            ToolCall(id="2", function=ToolFunction(name="Terminate", arguments='{}')),
        #        ]
        for command in self.tool_calls:
            # Reset base64_image for each tool call
            # 每次调用前重置 base64_image
            self._current_base64_image = None

            # 执行工具调用，获取结果
            result = await self.execute_tool(command)

            # 如果设置了最大观察长度，则截断结果
            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            # 把工具调用的结果作为 tool 消息加入 memory
            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        # 返回所有工具调用结果，使用换行分隔
        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling """
        #     执行单个工具调用，并进行健壮的异常处理。
        # Args:
        #     command: ToolCall 对象，包含工具名称和参数。

        # Returns:
        #     str: 工具调用的结果或错误信息。

        # command 例：
        # ToolCall(id="1", function=ToolFunction(name="CreateChatCompletion", arguments='{"text": "你好"}')),

        # 校验命令格式
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        # 检查工具名称是否在可用工具列表中
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            # 解析参数（JSON 格式）
            args = json.loads(command.function.arguments or "{}")

            # Execute the tool
            # 执行工具
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)
            # 这里会调用 CreateChatCompletion 工具，并传入 {"text": "你好"}

            # Handle special tools
             # 处理特殊工具（如 Terminate）
            await self._handle_special_tool(name=name, result=result)

            # Check if result is a ToolResult with base64_image
            # 如果结果有 base64_image，保存下来
            if hasattr(result, "base64_image") and result.base64_image:
                # Store the base64_image for later use in tool_message
                self._current_base64_image = result.base64_image

            # Format result for display (standard case)
            # 格式化结果
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        #用于处理特殊工具的执行和状态变更，比如遇到终止（Terminate）类工具时，自动将 agent 状态设置为 FINISHED

        # 判断当前工具是否属于特殊工具（如 Terminate）
        if not self._is_special_tool(name):
            return
        # 判断是否应该结束 agent 的执行
        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
            # 判断工具调用后是否应该结束 agent 的执行。
            # 默认总是返回 True，表示只要调用了特殊工具（如 Terminate），就会让 agent 结束。
            # 你可以在子类中重写这个方法，实现更复杂的判断逻辑。
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
            # 检查工具名称是否在特殊工具列表中。
            # Args:
            #     name (str): 工具名称
            # Returns:
            #     bool: 如果是特殊工具（如 Terminate），返回 True，否则返回 False
        # 忽略大小写进行比较
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """Clean up resources used by the agent's tools."""
        # 清理 agent 的所有工具实例占用的资源。
        # 如果工具实现了异步 cleanup 方法，则调用它进行资源释放
        # 保证 agent 运行结束后不会有资源泄漏，适合需要管理外部资源（如数据库、文件、网络连接等）的工具。
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            # 检查工具实例是否有 cleanup 方法且为异步函数
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cleanup when done."""
        try:
            return await super().run(request)
        finally:
            await self.cleanup()
