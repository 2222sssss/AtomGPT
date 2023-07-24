from langchain import PromptTemplate
from langchain.chains import LLMChain
from duckduckgo_search import ddg
from atomgpt_for_langchain import AtomGPT
def answer_from_web(question,llm):
    web_content = ''
    try:
        results = ddg(question,region="cn-zh",max_results=5)
        if results:
            for result in results:
                web_content += result['body']+'\n'
                if len(web_content)>1526:
                    break
    except Exception as e:
        print(f"网络检索异常:{question}")
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""<s>Human:基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
已知网络检索内容:
{context}

问题:
{question}\n</s><s>Assistant:"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.predict(question=question, context=web_content)
    return response
llm = AtomGPT('AtomEchoAI/AtomGPT_14k_chat_4bit')
response = answer_from_web('北京在哪？',llm)
print(response)
