from pathlib import Path
import random
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
        print(f"拥有权限，开始写入:{write_path}")
        try:
            with open(write_path,"w") as file:
                file.write(content)
            return f"✅写入成功！文件位置：{write_path}"
        except Exception as e:
            return f"❌写入失败: {e}"
    else:
        return f"❌{write_path} 没有写入权限"

tools = [write_tool]


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
    if isinstance(result, AIMessage) and result.tool_calls:
        print("\n🔧【模型决定调用工具】")
        for call in result.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]

            print(f"➡️ 工具名：{tool_name}")
            print(f"➡️ 参数：{tool_args}")

            tool_func = next(t for t in tools if t.name == tool_name)
            observation = tool_func.invoke(tool_args)

            print("\n📦【工具执行结果】")
            print(observation)

            history.add_message(
                ToolMessage(
                    tool_call_id=call["id"],
                    content=str(observation)
                )
            )

        print("\n✅【本轮结束：工具执行完成】\n")
        continue  # 回到 while True 等用户输入

    print(f"ℹ️{result.content}")
    
