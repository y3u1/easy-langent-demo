from myllm import chat_model as llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

intro = """
LangChain 是一个应用框架，旨在简化使用大型语言模型的应用程序。作为一个语言模型集成框架，LangChain 的用例与一般语言模型的用例有很大的重叠。 重叠范围包括文档分析和总结摘要, 代码分析和聊天机器人。 LangChain提供了一个标准接口，用于将不同的语言模型连接在一起，以及与其他工具和数据源的集成。
"""

praser = StrOutputParser()
def print_return(s):
    print(s)
    return s
runnable_print = RunnableLambda(
    lambda x : print_return(x)
)
extract_prompt = PromptTemplate(
    input_variables=["introduce"],
    template="根据以下产品介绍，提取至少3个核心卖点：{introduce}"
)

extract_chain = extract_prompt | llm | praser | runnable_print

skill_prompt = PromptTemplate(
    input_variables=["core"],
    template="根据以下产品的核心卖点，生成一段100字左右的营销话术：{core}"
)

skill_chain = skill_prompt | llm | praser | runnable_print

final_chain = ( extract_chain | skill_chain | praser)

result = final_chain.invoke({"introduce" : intro})
print(result)