"""
=============================================================
LEVEL 2 ? YOUR FIRST REAL AI CHATBOT WITH LANGGRAPH
=============================================================

WHAT YOU WILL LEARN:
  - How to add a real LLM (Claude) as a node
  - What MessagesState is (special state for chat)
  - How HumanMessage and AIMessage work
  - How to run a simple back-and-forth chat

WHAT YOU NEED:
  - Anthropic API key from https://console.anthropic.com
  - Set it in the .env file (instructions below)

THE STORY:
  The simplest possible chatbot.
  One node. One LLM call. That's it.

  User sends message
      ?
  [LLM Node] ? Claude reads the messages, replies
      ?
  Response returned

  In Level 3 we add memory so it REMEMBERS.
  For now, every conversation starts fresh.
=============================================================
"""

# -- SETUP: CREATE YOUR .env FILE FIRST -------------------
#
# Create a file called .env in:
# c:\Users\91709\OneDrive\Documents\AI_langgraph_developer\
#
# Put this inside it:
# ANTHROPIC_API_KEY=your_key_here
#
# Get your key from: https://console.anthropic.com
#

# -- IMPORTS ----------------------------------------------
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages   # special reducer for chat
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Annotated
from typing_extensions import TypedDict

# Load API key from .env file
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env")
)


# -- STEP 1: STATE FOR CHAT --------------------------------
#
# For chatbots we use a special state with 'messages'.
# The key thing: Annotated[list, add_messages]
#
# This means: DON'T overwrite messages, APPEND new ones.
# So the conversation history builds up automatically.
#
# Think of it like: n8n's $input.all() ? all previous items
#
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    # Annotated[list, add_messages] means:
    #   "this is a list, and when a node adds to it,
    #    APPEND don't overwrite"


# -- STEP 2: THE LLM NODE ---------------------------------
#
# This node is the brain ? it calls Claude.
# It reads ALL messages from state (the full conversation)
# and returns Claude's reply.
#
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

def chat_node(state: ChatState) -> dict:
    """
    The only node in this graph.
    Reads messages, calls Claude, returns reply.
    """
    print("\n[AI] Claude is thinking...")

    # Add a system message to give Claude its personality
    system = SystemMessage(content=
        "You are a friendly assistant. "
        "Keep answers short and clear. "
        "Use simple language."
    )

    # Call Claude with the system message + all conversation messages
    response = llm.invoke([system] + state["messages"])

    print(f"[AI] Claude replied: {response.content[:80]}...")

    # Return the AI's reply ? add_messages will APPEND it to the list
    return {"messages": [response]}


# -- STEP 3: BUILD THE GRAPH -------------------------------
def build_chatbot():
    builder = StateGraph(ChatState)

    # Just ONE node ? the LLM
    builder.add_node("chat", chat_node)

    # Entry point -> chat node -> END
    builder.set_entry_point("chat")
    builder.add_edge("chat", END)

    return builder.compile()


# -- STEP 4: RUN IT ----------------------------------------
if __name__ == "__main__":

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("""
[ERROR] No API key found!

Please create a .env file at:
c:\\Users\\91709\\OneDrive\\Documents\\AI_langgraph_developer\\.env

Put this inside it:
ANTHROPIC_API_KEY=your_key_here

Get your key from: https://console.anthropic.com
        """)
        exit(1)

    chatbot = build_chatbot()

    print("=" * 50)
    print("LEVEL 2: Your First LangGraph Chatbot")
    print("Type 'quit' to exit")
    print("=" * 50)

    # Keep a message history ourselves for now (no memory yet ? Level 3 does that)
    conversation_history = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message to history
        conversation_history.append(HumanMessage(content=user_input))

        # Run the graph with the full conversation history
        result = chatbot.invoke({"messages": conversation_history})

        # Get the latest AI message (last item in messages list)
        ai_reply = result["messages"][-1].content

        # Add AI reply to history for next round
        conversation_history.append(result["messages"][-1])

        print(f"\nClaude: {ai_reply}")

    print("\n" + "=" * 50)
    print("WHAT YOU JUST BUILT:")
    print("=" * 50)
    print("""
  [OK] A real AI chatbot using LangGraph + Claude
  [OK] State (ChatState) held the conversation messages
  [OK] One Node (chat_node) called Claude and returned the reply
  [OK] add_messages reducer APPENDED each message to the history

  LIMITATION: If you restart this script, memory is gone.
  That is fixed in Level 3 with the Checkpointer.
    """)
