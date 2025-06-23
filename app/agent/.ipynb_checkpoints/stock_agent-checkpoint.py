from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from agent.tools import fetch_news, retrieve_similar_labeled_example, classify_sentiment_with_few_shot
import streamlit as st
from google.genai import types
import asyncio


MODEL_ID="gemini-2.0-flash-lite"

AGENT_INSTRUCTIONS = """
You are a financial assistant that helps users stay updated on stock market trends by providing news highlights and summarizing the overall market sentiment.

**Workflow:**
- When the user asks about a stock, company, or market trend, use the 'fetch_news' tool to retrieve recent English news articles relevant to their query.
- For each fetched news article, use the 'retrieve_similar_labeled_example' tool to find the most semantically similar labeled finance news from the historical dataset.
- Use the 'classify_sentiment_with_few_shot' tool, providing both the new article and the matched labeled example, to classify the sentiment of the news as positive, negative, or neutral.
- Present a brief summary of the main news points to the user, including the sentiment for each article.
- At the end, provide a high-level summary of the overall sentiment trend for the user's query (e.g., "Most news about [topic] this week is positive/negative/neutral").
- If you are unable to find news or classify sentiment, let the user know and suggest they try a different query.

**General Guidance:**
- Always use the tools in the following order: 1) fetch_news, 2) retrieve_similar_labeled_example, 3) classify_sentiment_with_few_shot.
- If any tool fails or returns no results, inform the user politely and suggest next steps.
- Summarize results clearly, using easy-to-understand language and bullet points when sharing multiple news items or sentiments.
"""


def get_agent_runner(user_id, session_id):
    global runner, session_service
    try:
        return runner, session_service
    except NameError:
        APP_NAME = "stock_market_agent"
        fetch_news_tool = FunctionTool(func=fetch_news)
        retrieve_similar_labeles_tool = FunctionTool(func=retrieve_similar_labeled_example)
        sentiment_classification_tool = FunctionTool(func=classify_sentiment_with_few_shot)
        agent = Agent(
            model=MODEL_ID,
            name='stock_market_agent',
            instruction=AGENT_INSTRUCTIONS,
            tools=[fetch_news_tool, retrieve_similar_labeles_tool, sentiment_classification_tool]
        )
        session_service = InMemorySessionService()
        asyncio.run(session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        ))
        runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
        return runner, session_service


def call_agent_streamlit(query, runner, user_id, session_id):

    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=user_id, session_id=session_id, new_message=content)
    steps_list = []
    final_response = None

    with st.status("🤖 Agent reasoning...", expanded=True) as status:
        for event in events:
            role = getattr(event.content, 'role', None) if hasattr(event, 'content') and event.content else None

            for part in event.content.parts:
                # --- AGENT THOUGHT (capture all text, except explicit tool output) ---
                if hasattr(part, "text") and part.text and part.text.strip():
                    st.markdown(f"**🧠 Agent Thought:**\n\n{part.text.strip()}")
                    steps_list.append({
                        "type": "thought",
                        "content": part.text.strip()
                    })
                # Tool Call
                if hasattr(part, "function_call") and part.function_call is not None:
                    st.markdown(f"**🔧 Tool Call:** `{part.function_call.name}`")
                    steps_list.append({
                        "type": "tool_call",
                        "content": part.function_call.name
                    })
                    st.markdown("**Arguments:**")
                    args_code = "\n".join(f"{k}: {v}" for k, v in part.function_call.args.items())
                    st.code(args_code, language="yaml")
                    steps_list.append({
                        "type": "args",
                        "content": args_code,
                        "language": "yaml"
                    })
                # Tool Output (structured)
                if hasattr(part, "function_response") and part.function_response is not None:
                    st.markdown("**📦 Tool Output:**")
                    st.code(str(part.function_response.response), language="json")
                    steps_list.append({
                        "type": "tool_output",
                        "content": str(part.function_response.response),
                        "language": "json"
                    })
            # Tool Output (plain text, role == "tool", if no structured output)
            if role == "tool":
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text and part.text.strip():
                        st.markdown("**📦 Tool Output:**")
                        st.code(part.text.strip(), language="markdown")
                        steps_list.append({
                            "type": "tool_output",
                            "content": part.text.strip(),
                            "language": "markdown"
                        })
            # Final response
            if event.is_final_response():
                status.update(label="✅ Agent response complete!", state="complete")
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text and part.text.strip():
                        st.markdown(f"**💡 Final Agent Response:**\n\n{part.text.strip()}")
                        final_response = part.text.strip()

    return steps_list, final_response


