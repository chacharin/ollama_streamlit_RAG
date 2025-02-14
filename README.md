## 📌 RAG Pipeline 

Retrieval-Augmented Generation (RAG) Pipeline แบ่งออกเป็น 2 ส่วนหลัก:
1️⃣ **Document Ingestion (การนำเข้าและจัดเก็บเอกสาร)**
2️⃣ **Retrieval & Generation (การค้นหาและสร้างคำตอบ)**

![RAG Pipeline](https://raw.githubusercontent.com/chacharin/ollama_streamlit_RAG/main/RAG-Step.png)
---

## 🔹 ส่วนที่ 1: Document Ingestion
### 📌 เครื่องมือที่ใช้
✅ PyPDFLoader → โหลดเอกสาร PDF และแปลงเป็นข้อความ  
✅ RecursiveCharacterTextSplitter → แบ่งเอกสารออกเป็น Chunks  
✅ FastEmbedEmbeddings → แปลงข้อความเป็น เวกเตอร์ฝังตัว (Embeddings)  
✅ Chroma Vector Store → บันทึกข้อมูลลงฐานควรในรูปแบบเวกเตอร์  

### 📌 กระบวนการทำงาน
- **Enterprise Knowledge Base** → เป็นแหล่งข้อมูลที่ใช้ เช่น ไฟล์ **PDFs, Docs**  
- **Preprocess Documents** → นำเอกสารมา **ประมวลผลและแยกข้อความ**  
- **Embedding Model** → แปลงข้อความเป็น Vector
- **Vector DB** → จัดเก็บเอกสารในรูป Vector เพื่อให้ AI ค้นหา 

---

## 🔹 ส่วนที่ 2: Retrieval และ Generation
### 📌 เครื่องมือที่ใช้
✅ Chroma.as_retriever() → ค้นหาข้อมูลที่เกี่ยวข้องจาก Vector Store
✅ ChatOllama → ใช้ LLM (Llama3.1) สร้างคำตอบจากข้อมูลที่ค้นพบ
✅ Streamlit → สร้าง Web App ให้ผู้ใช้ถามคำถามและรับคำตอบ
### 📌 กระบวนการทำงาน
- **User Query** → ผู้ใช้พิมพ์คำถามใน **Chat Bot Web App**  
- **Embedding Model** → แปลงคำถามเป็นเวกเตอร์  
- **Vector DB (Chroma, FAISS, Milvus)** → ค้นหาข้อมูลที่เกี่ยวข้องจากฐานข้อมูล  
- **LLM (Large Language Model)** → นำข้อมูล `{context}` ที่ดึงมาใช้สร้างคำตอบ  
- **แสดงคำตอบให้ผู้ใช้**  



