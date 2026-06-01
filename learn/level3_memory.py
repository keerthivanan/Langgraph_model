"""
=============================================================
LEVEL 3 ? MEMORY: AI THAT REMEMBERS YOU FOREVER
=============================================================

WHAT YOU WILL LEARN:
  - What a Checkpointer is (the memory database)
  - What thread_id is (your conversation ID)
  - How to have multi-turn conversations with real memory
  - How to switch between different conversations

THE PROBLEM WITH LEVEL 2:
  Every time you restart the script -> memory is gone.
  If you close the chat and reopen -> AI forgets everything.

THE SOLUTION: CHECKPOINTER
  A checkpointer saves the ENTIRE STATE to a database
  after every single node runs.

  Next time you send a message with the same thread_id,
  LangGraph loads the saved state and continues
  from where you left off.

  thread_id = conversation ID
  Same thread_id = same conversation memory
  Different thread_id = fresh conversation

THE STORY:
  Sarah talks to the AI in the morning (thread: "sarah-morning")
  AI remembers her name and preferences.
  Sarah comes back in the afternoon (same thread: "sarah-morning")
  AI still remembers everything.
  Her colleague John starts his own chat (thread: "john-work")
  John has a completely separate, private conversation.

  This is EXACTLY like n8n's workflow execution history ?
  each execution has its own ID and its own data.
=============================================================
"""

# -- IMPORTS ----------------------------------------------
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # <- THE CHECKPOINTER
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import Annotated
from typing_extensions import TypedDict

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


# -- STATE -------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# -- LLM NODE ---------------------------------------------
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

def chat_node(state: ChatState) -> dict:
    system = SystemMessage(content=
        "You are a helpful assistant with memory. "
        "Remember everything the user tells you. "
        "Reference past conversation when relevant. "
        "Keep answers concise."
    )
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}


# -- BUILD GRAPH WITH CHECKPOINTER -------------------------
#
# The ONLY difference from Level 2:
# We create a MemorySaver and pass it to compile()
#
# MemorySaver = saves state in RAM (good for learning)
# SqliteSaver = saves to a file (survives restarts ? better)
# PostgresSaver = saves to database (production ? best)
#
def build_chatbot_with_memory():
    builder = StateGraph(ChatState)
    builder.add_node("chat", chat_node)
    builder.set_entry_point("chat")
    builder.add_edge("chat", END)

    # This one line gives the AI permanent memory
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# -- HOW TO USE THE CHECKPOINTER ---------------------------
#
# You pass a "config" to every invoke() call.
# The config contains the thread_id.
# LangGraph uses thread_id to find/save the right memory.
#
# config = {"configurable": {"thread_id": "any-unique-string"}}
#


# -- RUN IT ------------------------------------------------
if __name__ == "__main__":

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[ERROR] No API key! Create .env file with ANTHROPIC_API_KEY=your_key")
        exit(1)

    chatbot = build_chatbot_with_memory()

    print("=" * 55)
    print("LEVEL 3: LangGraph Chatbot WITH Memory")
    print("=" * 55)
    print("""
HOW TO TEST MEMORY:
  1. Type: "My name is [your name] and I love pizza"
  2. Type: "What is my name?"
  3. Type: "What food do I love?"
  -> AI should remember both answers!

HOW TO TEST MULTIPLE CONVERSATIONS:
  When asked for thread_id, try:
  - "sarah" for one conversation
  - "john"  for a completely separate conversation

Type 'quit' to exit, 'switch' to change conversation
    """)

    # Ask which conversation to use
    thread_id = input("Enter your name/thread_id (e.g. 'sarah'): ").strip() or "default"
    print(f"\n[OK] Starting conversation: '{thread_id}'")
    print("(All your messages here are saved under this ID)\n")

    while True:
        user_input = input(f"\n[{thread_id}] You: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye! Your conversation is saved.")
            break

        if user_input.lower() == "switch":
            thread_id = input("Enter new thread_id: ").strip() or "default"
            print(f"\n[OK] Switched to conversation: '{thread_id}'")
            print("(Loading memory for this conversation...)\n")
            continue

        if user_input.lower() == "history":
            # Show the saved state for this thread
            state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
            print(f"\n[SCROLL] Message history for '{thread_id}':")
            for i, msg in enumerate(state.values["messages"]):
                role = "You" if isinstance(msg, HumanMessage) else "Claude"
                print(f"  {i+1}. {role}: {msg.content[:60]}...")
            continue

        if not user_input:
            continue

        # -- THE KEY PART --
        # Pass thread_id in config so LangGraph knows
        # which conversation to load/save
        config = {"configurable": {"thread_id": thread_id}}

        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config   # <- THIS is what enables memory
        )

        ai_reply = result["messages"][-1].content
        print(f"\nClaude: {ai_reply}")
        print(f"(Memory saved for thread: '{thread_id}')")

    print("\n" + "=" * 55)
    print("WHAT HAPPENED UNDER THE HOOD:")
    print("=" * 55)
    print("""
  Each time you sent a message:

  1. LangGraph loaded saved state for your thread_id
     (all previous messages)

  2. Added your new message to the state

  3. Ran the chat_node (Claude replied)

  4. Saved the updated state back
     (now includes Claude's reply too)

  5. Next message: repeat from step 1

  The MemorySaver stored this in RAM.
  If you restart the script, memory is gone.

  To fix that: use SqliteSaver (saves to a file)
  -> Even after restarting Python, memory survives!

  Key insight: thread_id is just a string.
  You can use user email, session ID, anything unique.
    """)
