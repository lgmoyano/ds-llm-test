"""
RAG-based chatbot for answering questions about mental health documents

- Runs form a terminal as: chainlit run schizobot.py
- Prompted against answering questions outside the scope of the documents
- Uses OpenAI API for Q&A, so be sure to set the OPENAI_API_KEY environment variable
- Saves the conversation to a SQLite database conversation.db
- Assumes ./archivos/dsm_v1.pdf and ./archivos/trastornos.pdf are available
"""

import os

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import fitz
import pytesseract
import chainlit as cl
from PIL import Image
import io
import uuid
import logging
import sqlite3
from datetime import datetime

logger = logging.getLogger('schizobot_logger')
logger.setLevel(logging.DEBUG)
logging.basicConfig(filename='./schizobot.log', level=logging.DEBUG)

qa_chain = None

def initialize_database():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
    try:
        current_dir = os.getcwd()
        logging.debug(f"Current working directory: {current_dir}")

        conn = sqlite3.connect('./conversation.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS conversation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                sender TEXT,
                message TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logging.debug("Database initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}")

def save_conversation(session_id, conversation):
    """Saves the entire conversation to the database."""
    try:
        print("on save", session_id, conversation)
        conn = sqlite3.connect('./conversation.db')
        c = conn.cursor()
        for msg in conversation:
            timestamp = msg['timestamp']
            sender = msg['sender']
            message = msg['message']
            logger.debug(f"message to be saved: {message}")
            c.execute('INSERT INTO conversation (session_id, timestamp, sender, message) VALUES (?, ?, ?, ?)',
                      (session_id, timestamp, sender, message))
        conn.commit()
        conn.close()
        print("closed")
        logging.debug("Conversation saved successfully.")
    except Exception as e:
        logging.error(f"Error saving conversation: {e}")

def extract_text_from_pdfs(pdf_files):
    """Extracts text from PDF files and returns a list of Document objects."""
    documents = []
    
    for pdf_file in pdf_files:
        # Assume pdf
        with fitz.open(pdf_file) as pdf:
            full_text = ""
            for page in pdf:
                full_text += page.get_text("text")
            full_text = full_text.replace("Created in Master PDF Editor\n", "") # filter watermark
        
        # If extraction went well, create a Document object
        if(len(full_text) > 0):
            print("case >0", pdf_file, ":", len(full_text))
            doc = Document(page_content=full_text, metadata={"source": pdf_file})
            documents.append(doc)
        # If extraction comes out empty, we assume it's an image and process it with OCR
        else: 
            doc = fitz.open(pdf_file)
        
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                
                full_text = pytesseract.image_to_string(img)

                # OCR usually comes muddeled with typos and stuff, so we throw it to the LLM to clean it up
                full_text = clean_text_with_llm(full_text)
                print("case 0", pdf_file, len(full_text))
                
                doc = Document(
                    page_content=full_text,
                    metadata={"source": pdf_file, "page_number": page_num + 1}
                )
            documents.append(doc)
        
    return documents

def clean_text_with_llm(text):
    """Basic text cleaning using OpenAI's LLM. Prompt in Spanish as in documents to avoid mix-ups"""

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Limpiar este texto que viene con fallas de un proceso de OCR para que sea lo más legible posible. Quitar saltos de línea, guiones, etc. Tener mucho cuidado de no quitar palabras como Alejo, etc. Puedes formatear títulos como se hace en markdown."},
            {"role": "user", "content": text}
        ]
    )
    cleaned_text = response.choices[0].message.content
    return cleaned_text

def create_vector_store(documents):
    """Creates a vector store from the given list of Document objects."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    split_docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(split_docs, embedding=embeddings)
    return vector_store

def create_qa_chain(vector_store):
    """Creates a QA chain using the given vector store."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Error de API key. Asegurarse que se encuentre en la variable de entorno 'OPENAI_API_KEY'.")

    model = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
    system_prompt = (
        "Responde de manera breve a las preguntas utilizando únicamente la información proporcionada en los documentos. No agregues información adicional que no esté en estos documentos. Si no encuentras la respuesta en los documentos, responde 'No estoy entrenado para responder esta pregunta, solo estoy entrenado para responder preguntas de neurociencia'. Como los documentos se refieren a Alejo, tienes que estar preparado responder sobre Alejo."
        "\n\n"
        "{context}"
    )
    # final prompt includes the system´s (with the extracted documents as context), plus the user's question
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

@cl.on_chat_start
async def on_chat_start():
    """
    Stuff to do as chainlit's workflow starts
    - db initialization
    - pdf extraction
    - vector store and chain creation
    - initial message, including it in history
    """

    global qa_chain
    logging.debug("on_chat_start() called.")
    initialize_database()
    
    cl.user_session.session_id = str(uuid.uuid4())
    cl.user_session.message_history = []

    pdf_text = extract_text_from_pdfs(["./archivos/dsm_v1.pdf","./archivos/trastornos.pdf"])
    logging.debug("PDF documents loaded and processed.")

    vector_store = create_vector_store(pdf_text)
    qa_chain = create_qa_chain(vector_store)
    initial_message = "Archivos cargados, ya puedes hacer tus preguntas. Escribe '!fin' para finalizar la sesión."
    await cl.Message(content="Archivos cargados, ya se puede preguntar.").send()
    
    cl.user_session.message_history.append({
        'timestamp': datetime.now().isoformat(),
        'sender': 'assistant',
        'message': initial_message
    })

@cl.on_message
async def handle_message(message):
    """
    Handles incoming messages from the user
    - checks end of chat or inexsitent chain
    - if all ok sends the user's message to the chain and stores the response
    - some basic error handling
    """
    global qa_chain
    logging.debug("handle_message() called.")
    user_message = message.content.strip()
    logging.debug(f"User message: {user_message}")

    if user_message.lower() == "!fin":
        farewell_message = "chau!"
        await cl.Message(content=farewell_message).send()
        
        cl.user_session.message_history.append({
            'timestamp': datetime.now().isoformat(),
            'sender': 'assistant',
            'message': farewell_message
        })
        cl.user_session.end_chat = True
        return

    if qa_chain is None:
        error_message = "El chatbot no se ha inicializado. Por favor, intenta más tarde."
        await cl.Message(content=error_message).send()
        
        cl.user_session.message_history.append({
            'timestamp': datetime.now().isoformat(),
            'sender': 'assistant',
            'message': error_message
        })
    else:        
        cl.user_session.message_history.append({
            'timestamp': datetime.now().isoformat(),
            'sender': 'user',
            'message': user_message
        })

        try:
            result = qa_chain.invoke({"input": user_message})
            answer = result.get('answer', "Lo siento, pero no puedo responder lo que no se encuentra en mis archivos.")
            logging.debug(f"Assistant's response: {answer}")
            await cl.Message(content=answer).send()

            cl.user_session.message_history.append({
                'timestamp': datetime.now().isoformat(),
                'sender': 'assistant',
                'message': answer
            })
        except Exception as e:
            error_message = f"Ocurrió un error: {str(e)}"
            logging.error(error_message)
            await cl.Message(content=error_message).send()

            cl.user_session.message_history.append({
                'timestamp': datetime.now().isoformat(),
                'sender': 'assistant',
                'message': error_message
            })

@cl.on_chat_end
async def on_chat_end():
    """" Handles the end of the chat session, saving the conversation to the database. """
    logging.debug("on_chat_end() called.")
    session_id = cl.user_session.get('session_id', str(uuid.uuid4()))
    conversation = cl.user_session.message_history
    logging.debug(f"Conversation data to be saved: {conversation}")
    save_conversation(session_id, conversation)
    logging.debug("Conversation saved to the database.")