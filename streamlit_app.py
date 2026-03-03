import streamlit as st
import ollama
import os
import time
import threading
import queue
from rag_engine import RAGEngine
from assistant import OfflineAssistant

# Page Config
st.set_page_config(page_title="Offline AI Assistant", page_icon="🤖", layout="wide")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your offline AI assistant. How can I help you today?"}
    ]

if "rag" not in st.session_state:
    with st.spinner("Initializing RAG Engine..."):
        st.session_state.rag = RAGEngine()

if "voice_active" not in st.session_state:
    st.session_state.voice_active = False

if "msg_queue" not in st.session_state:
    st.session_state.msg_queue = queue.Queue()

# Callback for Voice Assistant
def voice_callback(role, content):
    st.session_state.msg_queue.put({"role": role, "content": content})

# Initialize Voice Assistant in session state if not already there
if "assistant_obj" not in st.session_state:
    st.session_state.assistant_obj = OfflineAssistant(callback=voice_callback)

# Sidebar for Configuration and Indexing
with st.sidebar:
    st.title("Settings")
    model_name = st.selectbox("LLM Model", ["phi3:latest", "gemma3:4b", "llama3:8b"], index=0)
    
    st.divider()
    
    st.subheader("Voice Assistant")
    voice_btn = st.toggle("Enable Voice Mode", value=st.session_state.voice_active)
    
    if voice_btn != st.session_state.voice_active:
        st.session_state.voice_active = voice_btn
        if voice_btn:
            # Start Voice Assistant Thread
            st.session_state.voice_thread = threading.Thread(target=st.session_state.assistant_obj.run, daemon=True)
            st.session_state.voice_thread.start()
            st.success("Voice Assistant Started")
        else:
            # Stop Voice Assistant
            st.session_state.assistant_obj.stop()
            st.info("Voice Assistant Stopped")
            # Re-init assistant object because stop() terminates pyaudio
            st.session_state.assistant_obj = OfflineAssistant(callback=voice_callback)
    
    st.divider()
    
    st.subheader("Data Management")
    if st.button("Reindex Personal Data"):
        with st.spinner("Indexing files in data/ folder..."):
            status = st.session_state.rag.index_data()
            st.success(status)
    
    st.info("Place your PDF, TXT, and DOCX files in the `offline_assistant/data/` folder.")

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:
            with st.expander("Retrieved Sources"):
                st.info(message["context"])

# Check for new messages from voice thread
new_msg = False
while not st.session_state.msg_queue.empty():
    msg = st.session_state.msg_queue.get()
    st.session_state.messages.append(msg)
    new_msg = True

if new_msg:
    st.rerun()

# Chat Input
if prompt := st.chat_input("Ask me something..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 1. Retrieve Context
        with st.spinner("Searching personal documents..."):
            context = st.session_state.rag.query(prompt)
        
        # 2. Build Prompt
        system_prompt = (
            "You are a helpful offline personal AI assistant. "
            "Use the following pieces of retrieved context from the user's personal documents "
            "to answer the question. If you don't know the answer or it's not in the context, "
            "you can still answer based on your general knowledge but mention if you are "
            "using general knowledge vs personal context.\n\n"
            f"PERSONAL CONTEXT:\n{context}\n\n"
            "Be concise and friendly."
        )

        try:
            # 3. Call Ollama (Streamed)
            response = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                stream=True,
            )
            
            for chunk in response:
                content = chunk['message']['content']
                full_response += content
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Save Assistant Message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "context": context if context else "No specific personal context found."
            })
                    
        except Exception as e:
            error_msg = f"Error: {str(e)}. Make sure Ollama is running (`ollama serve`)."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Auto-refresh if voice is active to check for new messages
if st.session_state.voice_active:
    time.sleep(0.5)
    st.rerun()
