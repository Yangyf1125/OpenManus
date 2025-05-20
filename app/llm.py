import math
from typing import Dict, List, Optional, Union

import tiktoken
from openai import (APIError, AsyncAzureOpenAI, AsyncOpenAI,
                    AuthenticationError, OpenAIError, RateLimitError)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_random_exponential)

from app.bedrock import BedrockClient
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import (ROLE_VALUES, TOOL_CHOICE_TYPE, TOOL_CHOICE_VALUES,
                        Message, ToolChoice)

REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

# 估算 LLM 消息和图片的 token 数量
class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4                 # 每条消息的基础 token 数
    FORMAT_TOKENS = 2                       # 整体格式相关的 token 数
    LOW_DETAIL_IMAGE_TOKENS = 85            # 低细节图片固定 token 数
    HIGH_DETAIL_TILE_TOKENS = 170           # 高细节图片每个 tile 的 token 数

    # Image processing constants
    MAX_SIZE = 2048                         # 图片最大边长（像素），用于缩放
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768     # 高细节图片短边缩放目标（像素）
    TILE_SIZE = 512                         # 高细节图片 tile 的尺寸（像素）

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer # 通过传入不同的 tokenizer，TokenCounter 可以适配不同 LLM 的 token 计算方式。

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        # 这个方法用于计算一段文本在当前 LLM 分词器下的 token 数量。
        # 如果 text 为空，返回 0。
        # 如果有内容，就用 self.tokenizer.encode(text) 把文本编码成 token，然后返回 token 的数量。
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """

        # 根据图片细节等级和尺寸，估算图片的 token 数量。

        # - 低细节（"low"）：固定 85 token
        # - 高细节或中等细节（"high"/"medium"）：
        #     1. 如果有尺寸，按尺寸计算 tile 数量，每个 tile 170 token，最后加 85 token
        #     2. 如果没有尺寸，high 默认按 1024x1024 估算，中等细节直接返回 1024


        detail = image_item.get("detail", "medium")

        # 低细节图片，直接返回固定 token 数
        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # 高细节或中等细节，优先用尺寸计算
        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        # 没有尺寸时的默认估算
        return (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        # Step 1: 如果图片太大，先等比缩放到最大边长不超过 MAX_SIZE（如 2048）
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        # Step 2: 再缩放，使最短边等于 HIGH_DETAIL_TARGET_SHORT_SIDE（如 768）
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        # Step 3: 计算 512px tile 的数量（横向和纵向都向上取整）
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        # Step 4: 总 token = tile 数量 * 每 tile token 数 + 低细节图片 token
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS


    # 用于统计消息内容中的 token 数量，支持文本、多段内容和图片
    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        # 如果 content 是字符串，直接统计 token 数
        if isinstance(content, str):
            return self.count_text(content)

        # 如果 content 是列表，遍历每个元素
        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count


    # 用于统计工具调用（function call）相关的 token 数量，常用于 LLM function calling 场景
    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    # 用于统计一组消息（如对话历史）在 LLM 中的总 token 数，适用于 token 限制判断、计费等场景
    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    _instances: Dict[str, "LLM"] = {}

    #实现了按配置名（config_name）单例的模式，每个 config_name 只会创建一个 LLM 实例，后续用同样的 config_name 获取到的是同一个对象（单例）。
    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]
        # llm1 = LLM("default")
        # llm2 = LLM("default")
        # assert llm1 is llm2  # True，同一个实例
        # llm3 = LLM("other")
        # assert llm1 is not llm3  # 不同配置名，不同实例


    # 初始化 LLM 实例的各种配置和资源，用例 llm = LLM("default")
    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            # Add token counting related attributes
            self.total_input_tokens = 0
            self.total_completion_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            # Initialize tokenizer
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # If the model is not in tiktoken's presets, use cl100k_base as default
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            elif self.api_type == "aws":
                self.client = BedrockClient()
            else:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

            self.token_counter = TokenCounter(self.tokenizer)


    # 计算一段文本在当前 LLM 分词器下的 token 数量，是最基础的 token 统计工具
    # llm = LLM("default")
    # tokens = llm.count_tokens("你好，世界！")
    # print(tokens)  # 输出这段文本的 token 数
    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    # 统计一组消息（如对话历史）在 LLM 中的总 token 数
    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    # 更新 LLM 实例的 token 统计信息
    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # 累加输入和输出 token 数
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    # 判断本次请求的 token 数是否会超出 LLM 的最大输入 token 限制
    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # If max_input_tokens is not set, always return True
        return True

    # 生成 token 超限时的错误提示信息
    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"


    #用于将输入的消息（可以是 dict 或 Message 对象）格式化为 OpenAI API 兼容的消息格式，并根据模型是否支持图片自动处理图片内容
    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    # Just remove the base64_image field and keep the text content
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages


    # 用于自动重试被装饰的异步方法，常用于 LLM API 调用等易出错的场景
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask(# 向大模型发送对话消息，获取模型回复
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Check if the model supports images
            # 1.判断模型是否支持图片
            supports_images = self.model in MULTIMODAL_MODELS

            # Format system and user messages with image support check
            # 2.格式化消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images) # 将输入消息转为 OpenAI API 兼容格式，并自动处理图片内容。
                messages = system_msgs + self.format_messages(messages, supports_images) # 把系统消息和用户消息合并
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            # 3.统计 token 数
            input_tokens = self.count_message_tokens(messages)

            # Check if token limits are exceeded
            # 4.检查 token 限制
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # 5.组装 API 参数
            params = {
                "model": self.model,
                "messages": messages,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # 6.调用 LLM API
            if not stream:
                # Non-streaming request
                # 非流式输出，直接等待完整回复
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                # Update token counts
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                return response.choices[0].message.content

            # 流式输出，边生成边输出（适合长文本或实时场景）
            # Streaming request, For streaming, update estimated token count before making the request
            self.update_token_count(input_tokens) # 先记录本次输入的 token 数

            response = await self.client.chat.completions.create(**params, stream=True)

            collected_messages = []
            completion_text = ""
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message
                print(chunk_message, end="", flush=True) # 实时输出到控制台

            print()  # Newline after streaming # 输出换行，结束流式输出
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            # estimate completion tokens for streaming response
            # 估算流式回复的 completion token 数
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"Estimated completion tokens for streaming response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask")
            raise


    # 用于带图片输入的多模态对话请求的方法。它可以让你把文本消息和图片一起发送给支持多模态的 LLM
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            if self.model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS}"
                )

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # Set up API parameters
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # Add model-specific parameters
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Handle non-streaming request
            if not stream:
                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content

            # Handle streaming request
            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
            raise



    # 用于调用带有 function calling（工具调用）能力的大模型，让 LLM 能根据你的消息自动选择并调用工具（如函数、插件等），并返回结构化回复
    @retry(
        wait=wait_random_exponential(min=1, max=60), # 每次重试等待时间呈指数增长，最短1秒，最长60秒
        stop=stop_after_attempt(6), # 最多重试6次
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded # 只在遇到这些异常时重试（TokenLimitExceeded不会重试）
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]], #对话消息列表
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None, # 工具列表（如函数描述的 dict）
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore 工具选择策略（如 "auto"、"none"、"required"）
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            # 1.确保 tool_choice 参数合法，否则直接报错。
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Check if the model supports images
            # 2.判断模型是否支持图片
            supports_images = self.model in MULTIMODAL_MODELS

            # Format messages
            # 3. 格式化消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            # 4. 统计 token 数
            input_tokens = self.count_message_tokens(messages)

            # If there are tools, calculate token count for tool descriptions\
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # Check if token limits are exceeded
            # 5. 检查 token 限制
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            # 6. 校验工具参数
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")


            # Set up the completion request
            # 7. 组装 API 参数
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            params["stream"] = False  # Always use non-streaming for tool requests
            # 8. 调用 LLM API
            response: ChatCompletion = await self.client.chat.completions.create(
                **params
            )

            # Check if response is valid
            # 9. 检查回复有效性
            if not response.choices or not response.choices[0].message:
                print(response)
                # raise ValueError("Invalid or empty response from LLM")
                return None

            # Update token counts
            # 10. 更新 token 统计
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

            # 11. 返回结构化回复
            return response.choices[0].message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise
