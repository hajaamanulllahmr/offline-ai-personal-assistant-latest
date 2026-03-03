import streamlit as st
import threading
import time
from assistant_code import OfflineAssistant # Assuming your class is in assistant_code.py

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Offline AI Assistant", page_icon="🤖")
st.title("🤖 Offline RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "assistant_running" not in st.session_state:
    st.session_state.assistant_running = False

# --- Callback Function ---
# This updates the Streamlit UI from the assistant's thread
def ui_callback(role, text):
    st.session_state.messages.append({"role": role, "content": text})
    # Force a rerun to show new messages
    st.rerun()

# --- Assistant Logic ---
def start_assistant():
    if not st.session_state.assistant_running:
        # We initialize the assistant with our UI callback
        assistant = OfflineAssistant(callback=ui_callback)
        st.session_state.assistant_running = True
        
        # Run the assistant in a separate thread so it doesn't block the UI
        thread = threading.Thread(target=assistant.run, daemon=True)
        thread.start()
        st.session_state.assistant_thread = thread

# --- UI Layout ---
sidebar = st.sidebar
sidebar.header("Controls")

if sidebar.button("🚀 Start Voice Assistant"):
    start_assistant()
    sidebar.success("Assistant Listening...")

if sidebar.button("🛑 Stop Assistant"):
    st.session_state.assistant_running = False
    st.rerun()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Text input as an alternative to voice
if prompt := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Note: To fully integrate text-input with your class, 
    # you'd expose get_ollama_response as a public method.
    with st.chat_message("user"):
        st.markdown(prompt)
        def speak(self, text):
    if not text: return
    print(f"Assistant: {text}")
    
    # Update Streamlit UI via callback
    if self.callback:
        try:
            self.callback("assistant", text)
        except:
            pass 

    self.is_speaking = True
    # ... rest of your code