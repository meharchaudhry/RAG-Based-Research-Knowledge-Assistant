import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
import glob

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(page_title="DIP Research Chatbot", layout="wide", page_icon="🤖")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 15%;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        margin-right: 15%;
        color: #333333;
    }
    .source-citation {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #1565c0;
        font-weight: 500;
        border: 1px solid #bbdefb;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

# Header
st.markdown('<h1 class="main-header">Discover India Program - Research Chatbot</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Ask questions about the DIP research papers and get AI-powered answers with citations
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Setup")
    
    st.markdown("""
    ### Required Environment Variables:
    - MISTRAL_API_KEY
    - PINECONE_API_KEY
    
    Make sure these are set before clicking initialize!
    """)
    
    directory = st.text_input(
        "Documents Directory", 
        value="/Users/samichirungta/Desktop/4_year/NLP/dip_project"
    )
    
    if st.button("Initialize Chatbot", type="primary", use_container_width=True):
        if not os.path.exists(directory):
            st.error("Directory not found!")
        elif not os.environ.get("MISTRAL_API_KEY"):
            st.error("MISTRAL_API_KEY not set!")
        elif not os.environ.get("PINECONE_API_KEY"):
            st.error("PINECONE_API_KEY not set!")
        else:
            try:
                with st.spinner("Loading documents..."):
                    # Load documents
                    documents = []
                    
                    # Get all PDF files
                    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
                    for pdf_file in pdf_files:
                        try:
                            loader = PyPDFLoader(pdf_file)
                            pdf_docs = loader.load()
                            documents.extend(pdf_docs)
                            st.info(f"Loaded PDF: {os.path.basename(pdf_file)}")
                        except Exception as e:
                            st.warning(f"Could not load {os.path.basename(pdf_file)}: {str(e)}")
                    
                    # Get all text files
                    txt_files = glob.glob(os.path.join(directory, "*.txt"))
                    for txt_file in txt_files:
                        try:
                            loader = TextLoader(txt_file)
                            txt_docs = loader.load()
                            documents.extend(txt_docs)
                            st.info(f"Loaded TXT: {os.path.basename(txt_file)}")
                        except Exception as e:
                            st.warning(f"Could not load {os.path.basename(txt_file)}: {str(e)}")
                    
                    if not documents:
                        st.error("No documents found! Please check your directory path and ensure it contains PDF or TXT files.")
                        st.stop()
                    
                    # Split documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=20
                    )
                    docs = text_splitter.split_documents(documents)
                    
                    st.success(f"Loaded {len(documents)} papers ({len(docs)} chunks)")
                
                with st.spinner("Creating embeddings..."):
                    # Initialize embeddings
                    embeddings = MistralAIEmbeddings(
                        model="mistral-embed",
                        api_key=os.environ["MISTRAL_API_KEY"]
                    )
                
                with st.spinner("Setting up Pinecone..."):
                    # Initialize Pinecone with new API
                    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
                    index_name = "langchain-mistral-demo"
                    
                    # Check if index exists
                    existing_indexes = [idx.name for idx in pc.list_indexes()]
                    
                    if index_name not in existing_indexes:
                        st.info("Creating new Pinecone index...")
                        pc.create_index(
                            name=index_name,
                            dimension=1024,
                            metric='cosine',
                            spec=ServerlessSpec(cloud='aws', region='us-east-1')
                        )
                        import time
                        time.sleep(15)
                    
                    # Create vector store
                    st.session_state.vector_index = LangchainPinecone.from_documents(
                        docs, 
                        embeddings, 
                        index_name=index_name
                    )
                
                with st.spinner("Initializing AI..."):
                    # Initialize QA chain
                    llm = ChatMistralAI(
                        model="mistral-large",
                        api_key=os.environ["MISTRAL_API_KEY"]
                    )
                    from langchain.chains import StuffDocumentsChain
                    from langchain.chains.llm import LLMChain
                    from langchain.prompts import PromptTemplate
                    
                    # Define prompt template
                    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
                    
                    prompt = PromptTemplate(
                        template=prompt_template, input_variables=["context", "question"]
                    )
                    
                    # Create the chain
                    llm_chain = LLMChain(llm=llm, prompt=prompt)
                    st.session_state.qa_chain = StuffDocumentsChain(
                        llm_chain=llm_chain,
                        document_variable_name="context"
                    )
                
                st.session_state.system_ready = True
                st.success("Chatbot ready! Start asking questions!")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # System status
    st.markdown("### System Status")
    if st.session_state.system_ready:
        st.success("System Active")
        st.info(f"Messages: {len(st.session_state.chat_history)}")
    else:
        st.warning("Awaiting initialization")
    
    if st.session_state.system_ready and st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
if not st.session_state.system_ready:
    st.info("Please initialize the chatbot using the sidebar to begin")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Example Questions
        
        - What are the main findings about traditional dance forms?
        - Which states were covered in the research?
        - Tell me about religious festivals mentioned
        - What architectural styles are discussed?
        - Summarize the food and cuisine research
        - What methodologies were used in fieldwork?
        """)
    
    with col2:
        st.markdown("""
        ### Features
        
        - Smart Search: Finds relevant information across all papers
        - Source Citations: Shows which papers were used
        - Natural Language: Ask questions naturally
        - Context Aware: Understands follow-up questions
        - Comprehensive: Searches through 20+ research papers
        """)
    
    st.markdown("""
    ---
    ### About DIP
    
    The Discover India Program at FLAME University is an experiential learning initiative where students:
    - Conduct 10-day field research across India
    - Work in teams of 10-12 members
    - Study culture, heritage, arts, and traditions
    - Produce 60-80 page comprehensive reports
    """)

else:
    # Chat display area
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message">'
                    f'<strong>You:</strong><br>{message["content"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                # Split answer and sources
                content_parts = message["content"].split("**Sources:**")
                answer = content_parts[0].strip()
                sources = content_parts[1].strip() if len(content_parts) > 1 else ""
                
                st.markdown(
                    f'<div class="chat-message assistant-message">'
                    f'<strong>Assistant:</strong><br>{answer}',
                    unsafe_allow_html=True
                )
                
                if sources:
                    st.markdown(
                        f'<div class="source-citation">'
                        f'<strong>Sources:</strong> {sources}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area at bottom
    st.markdown("---")
    
    with st.form(key="question_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_question = st.text_input(
                "Ask a question:",
                placeholder="Type your question here...",
                label_visibility="collapsed"
            )
        
        with col2:
            ask_button = st.form_submit_button("Send", use_container_width=True, type="primary")
    
    # Process question
    if ask_button and user_question and user_question.strip():
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        with st.spinner("Searching through research papers..."):
            try:
                # Search for relevant documents
                docs = st.session_state.vector_index.similarity_search(user_question, k=4)
                
                # Generate answer
                response = st.session_state.qa_chain.invoke({
                    "input_documents": docs,
                    "question": user_question
                })
                
                # Get sources
                sources = list(set([
                    os.path.basename(doc.metadata.get('source', 'Unknown'))
                    for doc in docs
                ]))
                
                # Format response with sources
                full_response = f"{response['output_text']}\n\n**Sources:** {', '.join(sources)}"
                
                # Add assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666;">'
    '🎓 <strong>DIP Research Chatbot</strong> | FLAME University | '
    'Powered by Mistral AI & Pinecone'
    '</div>',
    unsafe_allow_html=True
)