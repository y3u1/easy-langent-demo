from myllm import chat_model as llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from typing import Dict

full_memory_prompt = ChatPromptTemplate.from_messages([
    ("system","你是一名专业的python ai程序员，擅长解决langchain,langgraph的问题，能细心地为小白提供指导.并且返回纯文本而不是markdown"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{user_input}")
])
praser = StrOutputParser()
base_chain = full_memory_prompt | llm | praser

full_memory : Dict[str,BaseChatMessageHistory] = {}


def get_full_memory(session_id:str) -> BaseChatMessageHistory:
    if session_id not in full_memory:
        full_memory[session_id] = InMemoryChatMessageHistory()
    return full_memory[session_id]
 
 

def get_window_memory(session_id:str) -> BaseChatMessageHistory:
    WINDOW_SIZE = 2
    if session_id not in full_memory:
        full_memory[session_id] = InMemoryChatMessageHistory()
    if len(full_memory[session_id].messages) > 2*WINDOW_SIZE:
        print("==============================记忆超出======================================")
        full_memory[session_id].messages = full_memory[session_id].messages[-2*WINDOW_SIZE:]
    return full_memory[session_id]
# 将原来的runnable对象包装成一个新的runnable对象
# 这个runnable对象能通过get_full_memory功能获取history
# 并将结果填入history_messages_key所在的占位符之上
# 同时将用户输入填充到user_input中
full_memory_chain = RunnableWithMessageHistory(
    runnable=base_chain,
    # get_session_history=get_full_memory,
    get_session_history=get_window_memory,
    history_messages_key="history",
    input_messages_key="user_input"
)

# 使用configurable配置session_id
# 注意，session_id是约定的名字
config = {"configurable": {"session_id": "user_001"}}

response1 = full_memory_chain.invoke({"user_input":"我是yue，我对runnable对象不太了解，你能回答我的问题吗"},config=config)
print(f"第一轮对话:\n{response1}")

response2 = full_memory_chain.invoke({"user_input":"我刚才问了什么问题？"},config=config)
print(f"第二轮对话:\n{response2}")


response3 = full_memory_chain.invoke({"user_input":"我是谁？"},config=config)
print(f"第三轮对话:\n{response3}")

print("\n窗口记忆的最终对话历史（最近2轮）：")
for msg in get_window_memory("user_001").messages:
    print(f"{msg.type}: {msg.content}")