from pathlib import Path
import random
import magic
from string import hexdigits
import os
from myllm import chat_model as llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
"""目标：构建一个能查看文件、创建文件、写入文件的文件Agent
    功能：
"""


@tool
def write_tool(path:str | None,file_name:str| None,content:str) -> str:
    """一个用于创建文件并写入内容的工具

    Args:
        path (str): 文件的路径,该值为空时，默认为当前目录
        file_name (str): 创建的文件名，该值为空时，默认创建以随机字符串为名的txt文件
        content (str): 写入文件的内容
    """
    if not file_name:
        file_name = "".join([random.choice(hexdigits) for i in range(10)]) + "_agent.txt"
    if not path:
        path = Path.cwd()
    else:
        path = Path(path)
        
    write_path = path / file_name
    
    print(f"写入文件夹位置:{write_path}")
    if os.access(path,os.W_OK):
        try:
            with open(write_path,"w") as file:
                file.write(content)
            return f"✅写入成功！文件位置：{write_path}"
        except Exception as e:
            return f"❌写入失败: {e}"
    else:
        return f"❌{write_path} 没有写入权限"

@tool
def list_tool(path:str | None) -> str:
    """列出指定文件夹下所有文件的名称及其类型

    Args:
        path (str | None): 文件夹路径

    Returns:
        str: 文件夹下文件名称或者错误消息
    """
    if not path:
        path = Path.cwd()
    else:
        path = Path(path)
    if os.access(path,os.R_OK):
        try:
            # files = "\n".join([name for name in os.listdir(path)])
            files = [name for name in os.listdir(path)]
            return f"✅文件夹路径{path}中的文件有：{files}"
        except Exception as e:
            return f"❌列文件失败: {e}"
    else:
        return f"❌{path} 没有读权限"

@tool
def delete_tool(path:str | None) -> str:
    """用于删除单个文件的工具，删除前需要向用户确认，并给出完整文件删除路径

    Args:
        path (str | None): 需要删除文件的路径

    Returns:
        str: 删除的结果
    """
    path = Path(path)
    if Path.is_dir(path):
        return f"{path}是一个文件夹！请指定单个文件"
    try:
        os.remove(path)
        return f"✅文件{path}删除成功"
    except PermissionError:
        return f"❌文件{path}删除权限不足，请检查父目录权限或文件是否被占用"
    except FileNotFoundError:
        return f"❌文件{path}不存在"
    except Exception as e:
        return False, f"❌未知错误: {e}"

@tool
def file_type_tool(path:str | None) -> str:
    """用于判断文件类型的工具

    Args:
        path (str | None): 文件路径

    Returns:
        str: 文件类型
    """
    path : Path = Path(path)
    if path.is_dir():
        return f"✅文件夹"
    mime = magic.Magic(mime=True)
    try:
        mime_type = mime.from_file(str(path))
        return f"✅{mime_type}"
    except Exception as e:
        return f"❌文件{path}类型判断失败"
    
tools = [write_tool,list_tool,file_type_tool,delete_tool]


history_memory = {}

def get_history(session_id:str):
    if session_id not in history_memory:
        history_memory[session_id] = InMemoryChatMessageHistory()
    return history_memory[session_id]


def execute_tool_call(tool_call: dict) -> str:
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    for tool in tools:
        if getattr(tool, "name", None) == tool_name or getattr(tool, "__name__", None) == tool_name:
            try:
                return str(tool.run(tool_args))
            except Exception as e:
                return f"❌工具执行失败: {e}"
    return f"❌未找到工具: {tool_name}"

prompt = ChatPromptTemplate.from_messages([
    ("system","""你是一名智能文件管理助手，并遵循以下规则:
     1. 根据用户请求，自主选择调用哪些工具
     2. 只会处理文件操作相关的请求，若不是文件操作的请求，不做任何操作并礼貌拒绝用户
     """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{user_input}")
])




base_chain = prompt | llm.bind_tools(tools)
agent_with_memory = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_history,
    history_messages_key="chat_history",
    input_messages_key="user_input"
)

config={"configurable": {"session_id": "1"}}
while True:
    user_input = input("❓")
    result = agent_with_memory.invoke({"user_input" : user_input},config=config)
    history = get_history("1")
    # 大模型只负责“说话”，无法自己调用工具
    # 因此需要返回要调用的函数名称和函数参数的
    # 使用 ToolMessage 标记为一个工具调用消息，从而langchain才知道大模型要调用工具了
    
    if isinstance(result, AIMessage) and result.tool_calls:
        print("\n🔧【模型决定调用工具】")
        for call in result.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]

            print(f"➡️ 工具名：{tool_name}")
            print(f"➡️ 参数：{tool_args}")

            # 调用大模型指定的工具
            # Q:一次只调用一个工具？
            tool_func = next(t for t in tools if t.name == tool_name)
            observation = tool_func.invoke(tool_args)

            print("\n📦【工具执行结果】")
            print(observation)

            # OpenAI API 规定：每一个 tool_calls 必须有且仅有一个对应的 ToolMessage，且 ID 必须匹配
            # content转化为str
            history.add_message(
                ToolMessage(
                    tool_call_id=call["id"],
                    content=str(observation)
                )
            )

        print("\n✅【本轮结束：工具执行完成】\n")
        continue  # 回到 while True 等用户输入

    print(f"ℹ️{result.content}")
    
