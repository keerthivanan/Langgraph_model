"""
=============================================================
LEVEL 4 ? TOOLS: AI THAT TAKES ACTIONS
=============================================================

WHAT YOU WILL LEARN:
  - What a Tool is (a function the AI can CHOOSE to call)
  - How to define tools with @tool decorator
  - What ToolNode is (the node that RUNS the tools)
  - How the AI decides WHEN to use a tool
  - The ReAct loop: Think -> Act -> Observe -> Think -> ...

THE PROBLEM WITH LEVELS 2 & 3:
  The AI only knows what Claude was trained on.
  It can't search YOUR documents.
  It can't check live data.
  It can't take real actions.

THE SOLUTION: TOOLS
  You define Python functions as "tools".
  The AI DECIDES when to call them.
  The tool runs, returns a result.
  The AI reads the result and continues.

THE STORY:
  We give the AI 3 tools:
  1. get_weather(city)    ? checks the weather
  2. calculate(expression) ? does math
  3. get_company_info(topic) ? looks up company data

  User: "What's the weather in Chennai?"
  AI thinks: "I need to use get_weather tool"
  AI calls: get_weather("Chennai")
  Tool returns: "Chennai: 38?C, Sunny"
  AI responds: "The weather in Chennai is 38?C and sunny!"

  THE GRAPH:
  [chat_node] -> did AI want to use a tool?
       ? YES -> [tool_node] -> back to [chat_node]
       ? NO  -> END

  Notice the LOOP! The AI can call tools multiple times
  before it's satisfied. This is the ReAct pattern.
=============================================================
"""

# -- IMPORTS ----------------------------------------------
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode          # handles running tools
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool            # @tool decorator
from typing import Annotated
from typing_extensions import TypedDict

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


# -- STEP 1: DEFINE YOUR TOOLS -----------------------------
#
# A tool is just a Python function with @tool decorator.
# The docstring is IMPORTANT ? the AI reads it to decide
# when to use this tool.
#

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.
    Use this when the user asks about weather or temperature.
    """
    # In real life: call a weather API
    # For learning: fake data
    weather_data = {
        "chennai":  "Chennai: 38?C, Sunny and humid",
        "mumbai":   "Mumbai: 32?C, Partly cloudy",
        "delhi":    "Delhi: 42?C, Hot and hazy",
        "bangalore": "Bangalore: 28?C, Pleasant",
        "kolkata":  "Kolkata: 35?C, Cloudy",
    }
    city_lower = city.lower()
    return weather_data.get(city_lower, f"{city}: 30?C, Data unavailable - showing default")


@tool
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.
    Use this for any math problem: addition, multiplication, percentages, etc.
    Input should be a valid Python math expression like '25 * 4' or '100 / 5 + 3'.
    """
    try:
        # Safe eval for math only
        allowed = {"+", "-", "*", "/", "(", ")", ".", " "}
        if all(c.isdigit() or c in allowed for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        else:
            return f"Could not calculate: {expression}. Use simple math operators only."
    except Exception as e:
        return f"Math error: {str(e)}"


@tool
def get_company_info(topic: str) -> str:
    """
    Look up company information and policies.
    Use this when the user asks about leave policy, salary, HR rules,
    company benefits, work hours, or any internal company information.
    """
    # In real life: this would search a vector database (RAG ? Level 5)
    # For learning: fake company data
    company_data = {
        "leave":    "Annual Leave Policy: 20 days paid leave per year. Sick leave: 10 days. "
                    "Casual leave: 5 days. Apply 2 weeks in advance for planned leaves.",
        "salary":   "Salary is credited on the last working day of each month. "
                    "Salary slips available in the HR portal.",
        "work hours": "Work hours: 9am to 6pm, Monday to Friday. "
                      "Remote work allowed 2 days per week.",
        "benefits": "Benefits include: Health insurance (family), "
                    "2 annual flight tickets, gym membership, "
                    "learning budget of ?50,000 per year.",
        "holidays": "Public holidays: 12 days per year. "
                    "List available in HR portal calendar.",
    }

    # Find the most relevant data
    topic_lower = topic.lower()
    for key, value in company_data.items():
        if key in topic_lower or topic_lower in key:
            return value

    return (f"Information about '{topic}' not found in company database. "
            f"Available topics: {', '.join(company_data.keys())}")


# -- STEP 2: BIND TOOLS TO THE LLM ------------------------
#
# We tell Claude: "You have access to these 3 tools.
# When you need info, call the right one."
#
tools = [get_weather, calculate, get_company_info]
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
llm_with_tools = llm.bind_tools(tools)  # <- Key: bind tools to LLM


# -- STEP 3: STATE -----------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# -- STEP 4: NODES -----------------------------------------

def agent_node(state: AgentState) -> dict:
    """
    The AI node. Claude decides:
    - Should I answer directly? -> returns text response
    - Do I need a tool? -> returns a tool_call (request to run a tool)
    """
    system = SystemMessage(content=
        "You are a helpful assistant for a company. "
        "You have tools to check weather, do math, and look up company info. "
        "Use tools when you need real data. "
        "Always be concise and helpful."
    )
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


# ToolNode is a pre-built LangGraph node that:
# 1. Reads the tool_call from the last AI message
# 2. Runs the actual Python tool function
# 3. Returns a ToolMessage with the result
tool_node = ToolNode(tools)


# -- STEP 5: THE ROUTING FUNCTION -------------------------
#
# After the AI node, check: did Claude want to use a tool?
# If YES -> run tool_node
# If NO  -> we're done, go to END
#
def should_use_tool(state: AgentState) -> str:
    """
    Check if the last AI message contains a tool call.
    If yes -> route to tool_node
    If no  -> route to END
    """
    last_message = state["messages"][-1]

    # Check if the message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"\n[TOOL] AI wants to use tool: {last_message.tool_calls[0]['name']}")
        return "use_tool"
    else:
        print(f"\n[OK] AI answered directly (no tool needed)")
        return "done"


# -- STEP 6: BUILD THE GRAPH -------------------------------
#
# THE LOOP PATTERN:
#
#  START
#    ?
#  [agent_node]  <- Claude thinks
#    ?
#  did it call a tool?
#    YES -> [tool_node] -> back to [agent_node]  (loop!)
#    NO  -> END
#
# This loop lets the AI call multiple tools in sequence
# until it has all the information it needs.
#
def build_agent():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.set_entry_point("agent")

    builder.add_conditional_edges(
        "agent",
        should_use_tool,
        {
            "use_tool": "tools",  # has tool call -> run tools
            "done":     END       # no tool call  -> finished
        }
    )

    # After tools run -> back to agent (so Claude can read the result)
    builder.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# -- STEP 7: RUN IT ----------------------------------------
if __name__ == "__main__":

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[ERROR] No API key! Create .env with ANTHROPIC_API_KEY=your_key")
        exit(1)

    agent = build_agent()

    print("=" * 55)
    print("LEVEL 4: LangGraph Agent WITH Tools")
    print("=" * 55)
    print("""
TRY THESE QUESTIONS (the AI will use tools automatically):

  Weather:     "What's the weather in Chennai?"
  Math:        "What is 1250 * 12?"
  Company:     "What is the leave policy?"
  Combined:    "What's the weather in Mumbai and how many
                work days are in a year? (250 * 8 = ?)"
  No tool:     "Tell me a joke" (AI answers directly)

Notice: you never tell the AI WHICH tool to use.
It reads your question and DECIDES by itself.
    """)

    config = {"configurable": {"thread_id": "tools-demo"}}

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input:
            continue

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )

        ai_reply = result["messages"][-1].content
        print(f"\nClaude: {ai_reply}")

    print("\n" + "=" * 55)
    print("THE REACT LOOP YOU JUST SAW:")
    print("=" * 55)
    print("""
  For "What's the weather in Chennai?":

  Round 1:
    agent_node -> Claude thinks "I need weather tool"
               -> returns tool_call: get_weather("Chennai")
    -> route: "use_tool" -> tool_node runs

  Round 2:
    tool_node -> runs get_weather("Chennai")
              -> returns ToolMessage: "Chennai: 38?C, Sunny"
    -> back to agent_node

  Round 3:
    agent_node -> Claude reads tool result
               -> writes final answer: "Chennai is 38?C..."
               -> NO more tool calls
    -> route: "done" -> END

  This loop is called ReAct (Reason + Act).
  It's the foundation of every production AI agent.
    """)
