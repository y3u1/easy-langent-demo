from myllm import chat_model as llm

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableSequence,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



def print_return(x):
    print(f"❗当前指令{x}")
    return x

pr_chain = RunnableLambda(lambda x : print_return(x))

"""一个能处理用户订单，退回款，保修问题的客服"""

order_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名处理用户订单问题的客服"),
    ("human", "用户问题:{user_question}")
])
order_chain = order_prompt | llm | StrOutputParser()

refund_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名能处理用户退款的客服"),
    ("human", "用户问题:{user_question}")
])
refund_chain = refund_prompt | llm | StrOutputParser()

warranty_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名能处理用户保修问题的客服"),
    ("human", "用户问题:{user_question}")
])
warranty_chain = warranty_prompt | llm | StrOutputParser()

# 默认提示词（统一变量名）
default_prompt = PromptTemplate.from_template(
    "抱歉，我无法解答你的问题'{user_question}'。请你重新描述问题，或者联系人工客服。"
)
default_chain = default_prompt | llm | StrOutputParser()

# ====================== 路由链 ======================
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是路由选择器，能结合用户问题和问题场景，输出以下标识：
- order：订单相关
- refund：退款相关
- warranty：维修相关
只返回标识，不要多余文字！"""),
    ("human", "用户问题：{user_question}")
])
router_chain = router_prompt | llm | StrOutputParser() | pr_chain

# runnable只会提取自己需要的值，对于多余的值进行忽略
final_chain = (
    RunnablePassthrough().assign(
        scene= lambda x : router_chain.invoke(x)# 新加参数
    ) 
    | RunnableBranch(
        (lambda x : x["scene"] == "order",order_chain),
        (lambda x: x["scene"] == "refund", refund_chain),
        (lambda x: x["scene"] == "warranty", warranty_chain),
        default_chain 
    )
).with_config(run_name="full_router_chain")


# 7. 测试不同场景输入
test_queries = [
    "我的订单什么时候发货？",
    "怎么申请退款呀？",
    "这个产品保修多久？",
    "你们家有什么新品？"  # 无法匹配，触发默认链
]

for query in test_queries:
    print(f"\n用户问题：{query}")
    print("客服回复：", final_chain.invoke({"user_question" : query}))
