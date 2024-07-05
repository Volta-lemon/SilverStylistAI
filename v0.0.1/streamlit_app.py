import streamlit as st
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())    # read local .env file
from langchain_community.chat_models import ChatZhipuAI
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# streamlit run streamlit_app.py
#export OPENAI_API_KEY=
#os.environ["OPENAI_API_BASE"] = 'https://api.chatgptid.net/v1'
# zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
os.environ["https_proxy"] = "http://127.0.0.1:7890"

def generate_response(input_text, api_key):
    llm = ChatZhipuAI(model="glm-4",temperature=0.7, api_key=api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    #st.info(output)
    return output

def get_vectordb(api_key:str):
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings(api_key=api_key)
    # 向量数据库持久化路径
    persist_directory = './data_base/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding,
    )
    return vectordb

#带有历史记录的问答链
def get_chat_qa_chain(question:str,api_key:str):
    vectordb = get_vectordb(api_key)

    llm = ChatZhipuAI(model="glm-4",temperature=0.7, api_key=api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    template = """你是一个杰出的优秀银饰客服，称呼其他人为姐姐或者小哥哥，根据，礼貌热情地进行推荐和回答。
    使用以下上下文作为你可能推荐的饰品，放在<context></context> XML标签内。
    <context>
    {context}
    </context>
    当回答用户时：
    你应该根据给出的内容进行回答,但不能用原来的context中的内容进行回答，主要目的是推荐<context></context>中的银饰产品，给出关于该饰品有价值的回答，尽量促成用户购买。
    回答的时候要带上，姐姐或者小哥哥的尊称。
    问题: {question}
    """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    retriever=vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        memory=memory,
        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT}
    )
    result = qa({"query": question})
    return result['result']

#不带历史记录的问答链
def get_qa_chain(question:str,api_key:str):
    vectordb = get_vectordb(api_key)
    llm = ChatZhipuAI(model="glm-4",temperature=0.7, api_key=api_key)
    template = """你是一个杰出的优秀银饰客服，称呼其他人为姐姐或者小哥哥，根据，礼貌热情地进行推荐和回答。
    使用以下上下文作为你可能推荐的饰品，放在<context></context> XML标签内。
    <context>
    {context}
    </context>
    当回答用户时：
    你应该根据给出的内容进行回答,但不能用原来的context中的内容进行回答，主要目的是推荐<context></context>中的银饰产品，给出关于该饰品有价值的回答，尽量促成用户购买。
    回答的时候要带上，姐姐或者小哥哥的尊称。
    问题: {question}
    """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]


# Streamlit 应用程序界面
def main():
    st.title('🦜🔗 银饰推荐小助手')
    api_key = st.sidebar.text_input('Zhipu API Key', type='password')
    print(api_key)
    # 添加一个选择按钮来选择不同的模型
    #selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    # selected_method = st.radio(
    #     "你想选择哪种模式进行对话？",
    #     ["None", "qa_chain", "chat_qa_chain"],
    #     captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container()
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        # if selected_method == "None":
        #     # 调用 respond 函数获取回答
        #     answer = generate_response(prompt, api_key)
        # elif selected_method == "qa_chain":
        #     answer = get_qa_chain(prompt,api_key)
        # elif selected_method == "chat_qa_chain":
        answer = get_chat_qa_chain(prompt,api_key)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   


if __name__ == "__main__":
    main()
