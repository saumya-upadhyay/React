import streamlit as st
from chatbot import get_response

st.set_page_config(page_title="Customer Service Chatbot", layout="centered")
st.title("🤖 Customer Support Chatbot ")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    bot_reply = get_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
