import gradio as gr
import random
import time

from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

GLOBAL_SCENE = "房产销售"

def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # template = """Use the following pieces of context to answer the question at the end.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # Use three sentences maximum and keep the answer as concise as possible.
    # {context}
    # Question: {question}
    # Helpful Answer:"""
    
    template = """请基于下面的示例结合场景回答问题
    示例: {context}
    场景: {scene}
    问题: {question}
    请注意:
    1. 如果你不知道答案，请不要尝试解答，回答 '这个问题我要问问领导' 即可.
    2. 请尽量结合
    """
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.8}
        ))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history, scene = GLOBAL_SCENE):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message, "scene": GLOBAL_SCENE})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    # if ans["source_documents"] or enable_chat:
    #     print(f"[result]{ans['result']}")
    #     print(f"[source_documents]{ans['source_documents']}")
    #     return ans["result"]
    # # 否则输出套路话术
    # else:
    #     return "这个问题我要问问领导"
    
    return ans["result"]


# def dropdown_hook(value):
#     GLOBAL_SCENE = value
    

def launch_gradio():
    
    dropdown = gr.Dropdown(
        ["房产销售", "家装", "电商"], label="选择场景", value="房产销售")  # 创建一个下拉框
    
    # style = gr.Dropdown(
    #     ["郭德纲", "李白", "爱因斯坦"], label="选择风格", value="房产销售")  # 创建一个下拉框
    
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="房产销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
        additional_inputs=[dropdown]
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
