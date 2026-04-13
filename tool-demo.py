from myllm import chat_model as llm
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.agent_toolkits import FileManagementToolkit


@tool
def get_weather(city:str):
    """查询指定城市天气"""
    weather_data = {
        "北京": "北京今日天气：晴，-2~8℃",
        "上海": "上海今日天气：多云，5~12℃",
        "广州": "广州今日天气：小雨，18~25℃",
    }
    return weather_data.get(city, f"暂无 {city} 数据")


tools =[get_weather]

agent = create_agent(
    model=llm,
    tools=tools,
    debug=True,  # 👈 打开过程打印
)

toolkit = FileManagementToolkit(root_dir=".")
tools = toolkit.get_tools()

# -------------------
# 3. 创建 Agent（最新版）
# -------------------
agent = create_agent(
    model=llm,
    tools=tools,
    debug=True,  # 打开调试，显示模型思考和工具调用过程
)

# -------------------
# 4. 执行任务
# -------------------
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "请创建一个名为 llm诗词.txt 的文件，并在文件中写入一首原创七言绝句，主题围绕科技与人文的融合。"}
    ]
})

print("\n任务执行完成！文件已写入。")
print("Agent最终输出：\n", response["messages"][-1].content)