import json
from typing import Dict, List

from myllm import chat_model as llm
from langchain_core.prompts import FewShotPromptTemplate,PromptTemplate
from langchain_core.example_selectors import BaseExampleSelector , LengthBasedExampleSelector
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser,JsonOutputParser,BaseOutputParser

class Demo:
    def __init__(self,llm):
        self.llm = llm
        self.str_praser = StrOutputParser()
        self.json_praser = JsonOutputParser()
        
        
    def prompt_template_demo(self,user_role:str,subject:str):
        prompt = PromptTemplate(
            input_variables=["user_role","subject"],
            template="给{user_role}生成一段关于{subject}的建议"
        )
        prompt_format = prompt.format(
            user_role=user_role,
            subject=subject
        )
        print("生成的提示词模板:")
        print(prompt_format)
        chain = llm | self.str_praser
        result = chain.invoke([{"role":"user","content":prompt_format}])
        print("返回的结果：")
        print(result)
    
    def few_shot_prompt_template_demo(self,subject:str):
        examples = [
            {"subject" : "python" , "advices" : "1.学习建议\n - 学习python数据类型\n - 学习python控制结构\n - 了解python语法糖\n2.注意事项\n - 多敲代码实践\n - 不要钻牛角尖 "},
            {"subject" : "高等数学基础" , "advices" : "1.学习建议\n - 学习极限，连续等基本概念\n - 学习微分\n - 学习多元函数\n2.注意事项\n - 多算\n - 打草稿要公整"}
            
        ]
        examples_template = PromptTemplate(
            input_variables=["advices","subject"],
            template="学科:{subject}\n建议:{advices}"
        )
        few_shot_prompt_template = FewShotPromptTemplate(
            examples = examples,
            example_prompt=examples_template,
            # prefix="返回json格式而不是列表,键为： subject,advices",
            partial_variables={"format_instructions": self.json_praser.get_format_instructions()},
            suffix="根据上面的描述，给出建议，并只返回你补充的结果，{format_instructions}：\n学科:{subject}\n建议:",
            input_variables=["subject"]
        )
        few_shot_prompt_template_format = few_shot_prompt_template.format(subject=subject)
        print("生成的提示词模板:")
        print(few_shot_prompt_template_format)
        
        chain = llm | self.json_praser
        result = chain.invoke([{"role":"user","content":few_shot_prompt_template_format}])
        print("返回的结果：")
        print(result)
        
        print(f"subject:\n{result.get("学科")}")
        print(f"advices:\n{result.get("建议")}")
        
    def example_selector_demo(self,subject:str,difficult:str):
        with open("examples.json", "r", encoding="utf-8") as f:
            examples = json.load(f)  # 从JSON中直接提取示例数据列表
        # 示例文件格式参考（learning_method_examples.json）前面内容

        # 3. 方案A：ExampleSelector按长度筛选示例（控制提示词总长度）
        # example_selector = LengthBasedExampleSelector(
        #     examples=examples,
        #     example_prompt=PromptTemplate(
        #         input_variables=["subject", "difficulty", "method"],
        #         template="学科：{subject}\n难度：{difficulty}\n学习方法：{method}\n"
        #     ),
        #     max_length=150,  # 控制示例总长度，避免提示词过长
        #     get_text_length=lambda x: len(x)  # 长度计算函数
        # )

        # 4. 方案B（推荐）：自定义ExampleSelector按难度筛选示例
        # 当需要根据用户输入的特征（如难度）精准匹配示例时，可自定义ExampleSelector
        class DifficultyExampleSelector(BaseExampleSelector):
            """根据用户输入的 difficulty 字段筛选样本"""
            def __init__(self, examples: List[Dict[str, str]]):
                self.examples = examples

            def add_example(self, example: Dict[str, str]) -> None:
                self.examples.append(example)

            def select_examples(self, input_variables: Dict[str, str]) -> List[Dict]:
                # 获取用户输入的难度等级，如果没有提供则默认为 'easy'
                target_difficulty = input_variables.get("difficulty", "easy")
                # 过滤出匹配难度的所有示例
                return [ex for ex in self.examples if ex.get("difficulty") == target_difficulty]


        # 本案例使用方案B（按难度筛选），如需使用方案A（按长度筛选），取消方案A的注释并注释掉方案B即可
        example_selector = DifficultyExampleSelector(examples=examples)

        # 5. 构建工程化少样本模板
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,  # 替换固定examples为动态选择器
            example_prompt=PromptTemplate(
                input_variables=["subject", "difficulty", "method"],
                template="学科：{subject}\n难度：{difficulty}\n学习方法：{method}\n"
            ),
            example_separator="\n", # 控制examples示例之间的分隔方式
            prefix="少样本提示：",
            suffix="参考以上示例，回答：\n学科：{new_subject}\n难度：{difficulty}\n学习方法：",
            input_variables=["new_subject", "difficulty"]  # 新增难度参数
        )

        # 6. 动态生成不同难度的提示词
        # 场景1：生成入门级LangChain学习方法
        formatted_prompt_easy = few_shot_prompt.format(
            new_subject=subject,
            difficulty=difficult
        )
        print("少样本提示词：")
        print(formatted_prompt_easy)
        result_easy = llm.invoke([{"role": "user", "content": formatted_prompt_easy}])
        print("\n学习方法：")
        print(result_easy.content)
if __name__ == "__main__":
    demo = Demo(llm)
    # demo.prompt_template_demo("高校学生","langgraph学习")
    # demo.few_shot_prompt_template_demo("agent")
    demo.example_selector_demo("Langchain","hard")