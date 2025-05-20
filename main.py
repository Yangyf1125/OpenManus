import asyncio

from app.agent.manus import Manus
from app.logger import logger


async def main():
    # Create and initialize Manus agent
    # 1.始化 Manus 智能体，agent = await Manus.create()，创建并初始化一个 Manus 智能体实例。
    agent = await Manus.create()
    try:
        prompt = input("Enter your prompt: ") # 获取用户输入
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        await agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup() # 异常与资源清理


if __name__ == "__main__":
    asyncio.run(main()) # 异步主程序启动
