import multiprocessing
import sys
from io import StringIO
from typing import Dict

from app.tool.base import BaseTool


# 用来安全地执行一段 Python 代码，并带有超时和输出捕获功能
class PythonExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""
    # 工具名称
    name: str = "python_execute"
    # 工具描述
    description: str = "Executes Python code string. Note: Only print outputs are visible, function return values are not captured. Use print statements to see results."
    # 工具参数定义（JSON Schema 格式）
    parameters: dict = {
        "type": "object", # 参数整体是一个对象（字典）
        "properties": {
            "code": { # 只需要一个参数：code
                "type": "string", # 类型是字符串
                "description": "The Python code to execute.", # 参数描述
            },
        },
        "required": ["code"], # code 是必填项
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:

    # 在受控环境下执行一段 Python 代码，并把 print 输出捕获到 result_dict 里。

    # 参数说明：
    # - code: 要执行的 Python 代码字符串
    # - result_dict: 用于存放执行结果的共享字典（进程间通信用）
    # - safe_globals: 限制可用全局变量的字典（用于安全隔离）

    # 执行流程：
    # 1. 临时重定向 sys.stdout，把 print 输出捕获到 StringIO。
    # 2. 用 exec 执行代码，限制其全局变量为 safe_globals。
    # 3. 把输出内容写入 result_dict["observation"]，并标记 success。
    # 4. 如果出错，把异常信息写入 result_dict["observation"]，并标记失败。
    # 5. 最后恢复 sys.stdout，避免影响后续代码。

        original_stdout = sys.stdout # 记录原始的标准输出
        try:
            output_buffer = StringIO() # 创建一个内存字符串缓冲区
            sys.stdout = output_buffer  # 捕获 print 输出 ，# 临时把标准输出重定向到缓冲区
            exec(code, safe_globals, safe_globals)  # 执行代码
            result_dict["observation"] = output_buffer.getvalue()  # 获取输出内容
            result_dict["success"] = True # 标记执行成功
        except Exception as e:
            result_dict["observation"] = str(e)  # 捕获异常信息
            result_dict["success"] = False # 标记执行失败
        finally:
            sys.stdout = original_stdout  # 恢复标准输出 ，避免影响后续代码

    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.

        Returns:
            Dict: Contains 'output' with execution output or error message and 'success' status.
        """
        # 执行传入的 Python 代码，带超时限制。
        # 参数：
        #     code (str): 要执行的 Python 代码字符串
        #     timeout (int): 最长允许执行的秒数，超时会强制终止
        # 返回：
        #     Dict: 包含 'observation'（输出或错误信息）和 'success'（是否成功）

        with multiprocessing.Manager() as manager:
            # 用于进程间通信的共享字典
            result = manager.dict({"observation": "", "success": False})

            # 构造安全的全局变量环境，限制代码能访问的内容
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}

            # 创建子进程执行代码，防止主进程被阻塞或崩溃
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            proc.start()
            proc.join(timeout) # 等待指定时间

            # 如果超时还没结束，强制终止进程
            # timeout process
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return {
                    "observation": f"Execution timeout after {timeout} seconds",
                    "success": False,
                }

            # 返回执行结果（输出或错误信息）
            return dict(result)

# 用法
# result = await tool.execute(code="print(1+1)")
# print(result)  # {'observation': '2\n', 'success': True}
