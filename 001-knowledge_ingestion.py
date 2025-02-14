import sys
from langchain_chroma import Chroma  # เครื่องมือจัดการข้อมูล vector
from langchain_community.document_loaders import PyPDFLoader  #เครื่องมือโหลดไฟล์ PDF เพื่อดึงข้อมูลมาใช้
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings  # เครื่องมือสร้าง vector
from langchain.text_splitter import RecursiveCharacterTextSplitter  # เครื่องมือแบ่งเอกสารออกเป็นส่วนเล็ก ๆ (Chunks)

def ingest():
    """
    ฟังก์ชันนี้ใช้สำหรับโหลดไฟล์ PDF, แยกข้อมูล, และสร้างฐานความรู้แบบเวกเตอร์ (Vector Store)
    เพื่อให้แชทบอทสามารถใช้ข้อมูลนี้ตอบคำถามได้
    """

    # โหลดไฟล์ PDF ที่ต้องการใช้เป็นฐานความรู้
    loader = PyPDFLoader(r"data\\toshiba-rice-cooker.pdf")  
    pages = loader.load_and_split()  # โหลดเอกสารและแบ่งเป็นหน้าต่าง ๆ

    # แบ่งเอกสารเป็นส่วนเล็ก ๆ เพื่อลดขนาดของข้อมูลและช่วยให้ระบบค้นหาข้อมูลได้แม่นยำขึ้น
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  # กำหนดขนาดของแต่ละส่วน (1024 ตัวอักษรต่อชิ้น)
        chunk_overlap=100,  # กำหนดให้แต่ละส่วนมีการซ้อนกัน 100 ตัวอักษร (ช่วยให้ข้อมูลไม่ขาด)
        length_function=len,  # ใช้ฟังก์ชัน len() ในการคำนวณขนาดของแต่ละส่วน
        add_start_index=True,  # เพิ่มหมายเลขเริ่มต้นของแต่ละส่วน เพื่อให้สามารถอ้างอิงได้ง่ายขึ้น
    )
    
    # แปลงเอกสารเป็นส่วนเล็ก ๆ (Chunks) ที่สามารถใช้ในการฝังตัวข้อมูล
    chunks = text_splitter.split_documents(pages)
    print(f"เอกสารทั้งหมดถูกแบ่งออกเป็น {len(chunks)} ส่วนจาก {len(pages)} หน้า")

    # สร้างเวกเตอร์ฝังตัว (Embeddings) เพื่อใช้แปลงข้อความเป็นตัวเลขที่ระบบสามารถประมวลผลได้
    embedding = FastEmbedEmbeddings()

    # สร้างฐานความรู้แบบเวกเตอร์ (Vector Store) โดยใช้ Chroma
    # - documents = chunks: ใช้ข้อมูลที่แบ่งแล้วมาเก็บในฐานความรู้
    # - embedding = embedding: ใช้ FastEmbedEmbeddings เป็นตัวแปลงข้อความเป็นเวกเตอร์
    # - persist_directory = "./knowledge": กำหนดให้ข้อมูลถูกบันทึกในโฟลเดอร์ "knowledge"
    Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./knowledge")

# เรียกใช้งานฟังก์ชัน ingest() เพื่อสร้างฐานข้อมูลของแชทบอท
ingest()
