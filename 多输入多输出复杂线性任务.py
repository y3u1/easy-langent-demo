from myllm import chat_model as llm


from langchain_core.runnables import RunnableMap,RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


sell_point_promt = PromptTemplate(
    input_variables=["introduction"],
    template="从以下产品介绍中提取三个核心卖点：{introduction}"
)

marking_prompt = PromptTemplate(
    input_variables=["target","points"],
    template="针对{target}群体，结合以下核心卖点，生成一套80字左右的朋友圈营销话术:{points}"
)
praser = StrOutputParser()
finnal_chain = (
    RunnableMap({
        "target" : RunnablePassthrough(),
        "points" : sell_point_promt | llm | praser
    })
    | marking_prompt
    | llm
    | praser
)

result = finnal_chain.invoke({
    "target" : "上班族群体",
    "introduction" : "这款无线耳机采用蓝牙5.3芯片，连接稳定无延迟..."
})

print(result)