import uuid
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from agent.stock_agent import get_agent_runner, call_agent_streamlit

if "USER_ID" not in st.session_state:
    st.session_state["USER_ID"] = str(uuid.uuid4())
user_id = st.session_state["USER_ID"]
session_id = "streamlit_session"

st.set_page_config(page_title="Stock Market Agent", page_icon=":money_with_wings:")
st.title("Hi, I'm Simona!")
st.subheader("Your Stock Insight and Market Outlook News Agent.")

if "history" not in st.session_state:
    st.session_state["history"] = []

# Get agent runner and session info (singleton)
runner, session_service = get_agent_runner(user_id, session_id)

# You should also get user_id the same way it's set in your runner
import uuid
if "USER_ID" not in st.session_state:
    st.session_state["USER_ID"] = str(uuid.uuid4())
user_id = st.session_state["USER_ID"]

# Show previous chat history
for message in st.session_state["history"]:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "agent":
        if "steps" in message and message["steps"]:
            with st.expander("🔍 Agent Reasoning Steps"):
                for step in message["steps"]:
                    if step["type"] == "thought":
                        st.markdown(f"**🧠 Agent Thought:**\n\n{step['content']}")
                    elif step["type"] == "tool_call":
                        st.markdown(f"**🔧 Tool Call:** `{step['content']}`")
                    elif step["type"] == "args":
                        st.markdown("**Arguments:**")
                        st.code(step["content"], language=step.get("language", ""))
                    elif step["type"] == "tool_output":
                        st.markdown("**📦 Tool Output:**")
                        st.code(step["content"], language=step.get("language", ""))
        st.chat_message("assistant").write(message["content"])

# Chat input at the bottom
user_input = st.chat_input("Ask about a stock, index, or market trend...")

if user_input:
    st.session_state["history"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    steps, agent_response = call_agent_streamlit(user_input, runner, user_id, session_id)
    if agent_response:
        st.session_state["history"].append({
            "role": "agent",
            "content": agent_response,
            "steps": steps
        })
        st.chat_message("assistant").write(agent_response)
        st.rerun()
