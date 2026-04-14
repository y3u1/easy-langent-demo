from myllm import chat_model as llm



# api连接错误可以进行多次重试

# chain_with_fallback: RunnableWithFallbacks = core_chain.with_fallbacks(
#     fallbacks=[fallback_chain],
#     exceptions_to_handle=(ConnectionError, TimeoutError),# ✅ 官方推荐：只捕获临时错误或网络错误
# )