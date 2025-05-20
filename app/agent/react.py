from abc import ABC, abstractmethod
from typing import Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.llm import LLM
from app.schema import AgentState, Memory


class ReActAgent(BaseAgent, ABC):
    name: str # 代理名称，必须字段
    description: Optional[str] = None # 代理描述，可选

    system_prompt: Optional[str] = None # 系统提示词，可选
    next_step_prompt: Optional[str] = None # 下一步操作提示，可选

    llm: Optional[LLM] = Field(default_factory=LLM) # 语言模型实例，可选，默认自动创建
    memory: Memory = Field(default_factory=Memory) # 记忆存储，默认自动创建
    state: AgentState = AgentState.IDLE # 当前状态，默认空闲

    max_steps: int = 10 # 最大执行步数
    current_step: int = 0 # 当前执行步数

    #以下为相比于BaseAgent新增的内容

    @abstractmethod #表示这是一个抽象方法，基类不实现，子类必须实现。
    async def think(self) -> bool:
        #让 agent 先“思考”，决定是否需要执行下一步操作
        #如果返回 True，则会执行 act；如果返回 False，则本步只思考不行动。
        """Process current state and decide next action"""

            # 处理当前状态并决定下一步行动。
            # 必须由子类实现，返回值为 bool，表示是否需要执行 act。


    @abstractmethod
    #作用：让 agent 执行具体的操作，比如调用工具、回复消息等。
    async def act(self) -> str:
        """Execute decided actions"""
            # 执行 agent 决定要采取的具体行动。
            # 必须由子类实现，返回值为 str，表示行动的结果或描述。


    async def step(self) -> str:
        """Execute a single step: think and act."""
            # 执行单步操作：先调用 think() 进行思考，再根据结果决定是否行动。

            # Returns:
            #     str: 行动结果或思考结果说明。

        should_act = await self.think()  # 先让 agent 思考，决定是否需要行动
        if not should_act: # 如果不需要行动，直接返回说明
            return "Thinking complete - no action needed"
        # 如果需要行动，执行 act() 并返回结果
        return await self.act()
