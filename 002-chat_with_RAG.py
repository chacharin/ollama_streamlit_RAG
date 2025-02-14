import streamlit as st  # เครื่องมือสร้างเว็บแอปพลิเคชั่น

# นำเข้าโมดูลที่ใช้สำหรับโมเดลแชทบอทและฐานข้อมูลแบบเวกเตอร์ (Vector Store)
from langchain_ollama import ChatOllama  # เครื่องมืเชื่อมต่อ Ollama
from langchain_chroma import Chroma  # เครื่องมือจัดการข้อมูล vector

# นำเข้าโมดูลที่ใช้สำหรับจัดการเอกสารและสร้าง embeddings
from langchain_community.document_loaders import PyPDFLoader  # เครื่องมือโหลดไฟล์ PDF
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings  # เครื่องมือแปลงข้อความเป็นเวกเตอร์ (Embeddings)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # เครื่องมือแบ่งเอกสารเป็นส่วนเล็ก ๆ (Chunks)
from langchain.prompts import PromptTemplate  # เครื่องมือสร้างโครงสร้างของคำสั่ง (Prompt)
from langchain.chains.combine_documents import create_stuff_documents_chain  # เครื่องมือรวมข้อมูลจากเอกสารเพื่อให้โมเดลใช้ตอบคำถาม
from langchain.chains import create_retrieval_chain  # เครื่องมือสร้างระบบดึงข้อมูลอัตโนมัติ

# 🏷️ ตั้งชื่อแอปพลิเคชัน Streamlit
st.title("💬 Product inquirie")

# 🚀 โหลดโมเดลเพียงครั้งเดียว (Cache ไว้เพื่อเพิ่มประสิทธิภาพ)
@st.cache_resource
def load_model():
    """
    โหลดโมเดลแชทบอทและสร้างระบบดึงข้อมูลจากฐานความรู้
    """

    # 1️⃣ ตั้งค่าโมเดลแชทบอท (เลือกโมเดลที่ต้องการ)
    model = ChatOllama(model="llama3.1:8b")  

    # 2️⃣ ตั้งค่า Prompt เพื่อกำหนดแนวทางการตอบของแชทบอท
    prompt = PromptTemplate.from_template(
    """
    [System Instructions]
    You are an expert customer support assistant for a company specializing in [Product]. 
    Your job is to provide clear, accurate, and concise answers based only on the provided context. 
    If the question is not relevant to the product knowledge, respond with: 
    "I'm sorry, but I don't have information about that. Please ask a product-related question."

    [User Query]
    - Question: {input}
    - Retrieved Knowledge: {context}

    [Response Guidelines]
    - Directly provide the **most relevant answer** based on the context.
    - If a process is involved, summarize it in **clear, short steps**.
    - **DO NOT** show any reasoning, explanations, or thought processes—**only the final answer**.
    - If the context does not contain the answer, reply with: "I'm sorry, I don't have that information."

    [Example Responses]
    **Scenario 1: When context is available**
    - Q: "How do I reset my device?"
    - A: "Press and hold the power button for 10 seconds until the LED blinks."

    **Scenario 2: When no context is available**
    - Q: "How do I return a product?"
    - A: "I'm sorry, I don't have that information."
    """
    )

    # 3️⃣ โหลดฐานข้อมูลเวกเตอร์ (Vector Store) ที่ใช้จัดเก็บความรู้จากเอกสาร
    embedding = FastEmbedEmbeddings()  # ใช้ FastEmbedEmbeddings เป็นตัวสร้าง Vector
    vector_store = Chroma(persist_directory="./knowledge", embedding_function=embedding)  # โหลดฐานความรู้

    # 4️⃣ กำหนดเงื่อนไขการค้นหาข้อมูล (Retriever)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",  # ใช้การค้นหาด้วยคะแนนความคล้ายกัน
        search_kwargs={"k": 3, "score_threshold": 0.5},  # ดึงข้อมูลที่คล้ายกัน 3 รายการ (k=3) และต้องมีคะแนนมากกว่า 0.5
    )

    # 5️⃣ รวมข้อมูลจากเอกสารแล้วให้โมเดลใช้ตอบคำถาม
    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    return chain  # ส่งคืนโมเดลแชทที่ตั้งค่าเสร็จแล้ว

# โหลดโมเดลแชทบอทเพียงครั้งเดียว
chain = load_model()

def ask(query: str):
    """
    ฟังก์ชันนี้ใช้สำหรับรับคำถามจากผู้ใช้และค้นหาคำตอบจากฐานความรู้
    """
    try:
        with st.spinner("🤖 Processing..."):  # แสดงแถบโหลดระหว่างประมวลผล
            result = chain.invoke({"input": query})  # ส่งคำถามไปที่แชทบอท
            answer = result["answer"]  # ดึงคำตอบที่ได้
            context_docs = result.get("context", [])  # ดึงเอกสารที่ใช้เป็นแหล่งข้อมูล

            sources = []
            for doc in context_docs:
                if "source" in doc.metadata:
                    sources.append(doc.metadata["source"])  # ดึงชื่อไฟล์ที่เป็นแหล่งข้อมูล

            return answer, sources  # ส่งคำตอบและแหล่งข้อมูลกลับไป

    except Exception as e:
        return f"❌ Error: {str(e)}", []  # แสดงข้อผิดพลาดหากเกิดปัญหา

# 📌 ส่วนของอินพุตจากผู้ใช้ (User Interface)
user_input = st.text_input("📩 Ask your questions here:", "")  # กล่องให้ผู้ใช้ป้อนคำถาม

# 📤 กดปุ่มเพื่อส่งคำถาม
if st.button("🔍 Submit"):
    if user_input.strip():  # ตรวจสอบว่าผู้ใช้พิมพ์คำถามจริงหรือไม่
        answer, sources = ask(user_input)  # ค้นหาคำตอบจากฐานความรู้
        st.write("📝 **Asnwer:**", answer)  # แสดงคำตอบที่ได้

        # 📚 แสดงแหล่งข้อมูลที่ใช้ตอบคำถาม (ถ้ามี)
        if sources:
            st.write("📌 **ที่มา:**")
            for source in set(sources):  # ลบรายการซ้ำ
                st.write(f"📄 {source}")

    else:
        st.warning("⚠️ Please fill the question before sumit!")  # แจ้งเตือนหากผู้ใช้ยังไม่ป้อนคำถาม
