from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """

    # Core attributes

    name: str = Field(..., description="代理的唯一名称")
    description: Optional[str] = Field(None, description="可选的代理描述")

    # Prompts
    system_prompt: Optional[str] = Field(None, description="系统级指令提示")
    next_step_prompt: Optional[str] = Field(None, description="确定下一步操作的提示")

    # Dependencies
    llm: LLM = Field(default_factory=LLM, description="语言模型实例")
    memory: Memory = Field(default_factory=Memory, description="代理的内存存储")
    state: AgentState = Field(
        default=AgentState.IDLE, description="当前代理状态"
    )

    # Execution control
    max_steps: int = Field(default=10, description="终止前的最大步骤数")
    current_step: int = Field(default=0, description="执行中的当前步骤")

    duplicate_threshold: int = 2

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses
                         # 允许子类或实例包含未在模型中声明的额外字段
    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions."""
        #    上下文管理器，用于安全地切换 agent 状态。
        # Args:
        #     new_state: The state to transition to during the context.
        #                要切换到的新状态。
        # Yields:
        #     None: Allows execution within the new state.

        # Raises:
        #     ValueError: If the new_state is invalid.
        #                 如果 new_state 不是合法的 AgentState。

        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state  # 记录切换前的状态
        self.state = new_state # 切换到新状态
        try:
            yield  # 让 with 语句块中的代码运行
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure # 如果出错，状态切换为 ERROR
            raise e  # 继续抛出异常
        finally:
            self.state = previous_state  # Revert to previous state # 无论是否出错，最后都恢复原状态

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore 消息发送者的角色（如 user、system、assistant、tool）
        content: str, #消息内容
        base64_image: Optional[str] = None, #可选的图片内容（base64编码）
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            #定义了不同角色对应的消息构造方法。
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),#（支持额外参数）
        }

        #校验角色，如果 role 不在支持的角色列表中，抛出异常。
        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")


        # Create message with appropriate parameters based on role 构造消息并添加到 memory
        #对于 "tool" 角色，会把所有额外参数传进去。其他角色只传 base64_image。
        #最终调用 self.memory.add_message(...) 把消息加入记忆。
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.
        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """

        #    异步执行 agent 的主循环。
        # Args:
        #     可选的初始用户请求（字符串），会作为第一条消息加入 memory。

        # Returns:
        #     执行结果的字符串摘要。

        # Raises:
        #     如果 agent 当前不是 IDLE 空闲 状态，则抛出异常。
        # """

        # 函数简要说明：
        # 只允许空闲状态启动，防止重复运行。
        # 支持初始请求，自动加入对话历史。
        # 用上下文管理器安全切换状态，保证异常时也能恢复。
        # 主循环里每步都调用 step()，并检测死循环。
        # 达到最大步数自动终止，最后清理资源并返回结果。

        # 1. 只有空闲状态才能运行
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        # 2. 如果有初始请求，把它作为用户消息加入 memory
        if request:
            self.update_memory("user", request)

        results: List[str] = []# 用于收集每一步的执行结果

        # 3. 切换到 RUNNING 状态，保证执行期间状态正确
        async with self.state_context(AgentState.RUNNING):
            # 4. 主循环：未达到最大步数且未完成
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1 # 步数加一
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step() # 执行一步（子类实现）

                # Check for stuck state
                # 5. 检查是否陷入死循环
                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")

            # 6. 如果达到最大步数，重置步数并恢复状态
            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")

        # 7. 善后处理，清理资源
        await SANDBOX_CLIENT.cleanup()
        # 8. 返回所有步骤的结果
        return "\n".join(results) if results else "No steps executed"

    @abstractmethod #表示这是一个抽象方法，基类不实现具体内容，必须由子类实现。如果子类没有实现这个方法，实例化时会报错
    async def step(self) -> str:
        #每个 agent 的“单步操作”逻辑都不一样，所以需要子类根据实际需求来实现
        """Execute a single step in the agent's workflow.
        Must be implemented by subclasses to define specific behavior."""

        # 执行 agent 工作流中的单步操作。
        # 这个方法必须由子类实现，用于定义每一步的具体行为。


    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        #    处理 agent 卡死（重复回复）状态的方法。

        # 当检测到 agent 多次生成重复内容时，
        # 给 next_step_prompt 增加一条提示，建议尝试新策略，避免重复无效路径。
        # 同时输出警告日志，便于排查问题。

        # 提示内容，建议尝试新策略
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."

        # 把提示内容加到下一步提示前面
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"

        # 输出警告日志
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""

        # 检查 agent 是否陷入死循环（即多次生成重复内容）。

        # Returns:
        #     bool: 如果检测到 assistant 连续多次回复相同内容，则返回 True，否则返回 False。


        # 如果消息数量不足2条，不可能卡死，直接返回 False
        if len(self.memory.messages) < 2:
            return False

        # 取最后一条消息
        last_message = self.memory.messages[-1]
        # 如果最后一条消息没有内容，也不算卡死
        if not last_message.content:
            return False

        # Count identical content occurrences
        # 统计之前 assistant 角色中，与最后一条内容相同的消息数量
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])  # 倒序遍历除最后一条外的消息
            if msg.role == "assistant" and msg.content == last_message.content
        )

        # 如果重复次数达到或超过阈值（如2次），认为 agent 卡死
        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""

        # 获取 agent 的所有消息列表。

        # Returns:
        #     List[Message]: 当前 agent 记忆中的所有消息。

        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""

        # 设置 agent 的消息列表。

        # Args:
        #     value: 要设置的新消息列表（List[Message]）。

        self.memory.messages = value
