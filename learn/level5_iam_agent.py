"""
=============================================================
LEVEL 5 ? IAM AGENT: AI THAT KNOWS WHO YOU ARE
=============================================================

WHAT YOU WILL LEARN:
  - How to combine IAM (identity) with LangGraph
  - How SecurityContext flows through agent State
  - How tools check permissions before doing anything
  - How the AI's personality changes based on user role
  - The complete production pattern for this job role

THIS IS THE FULL PROJECT.
Everything from Levels 1-4 combined + IAM from your iam.py

THE STORY:
  3 employees log in to the same AI chatbot.
  Same chatbot. Completely different experience.

  Guest (guest_user):
    -> Can only ask about public company info
    -> Asks "What is the HR leave policy?" -> DENIED
    -> Asks "What are your working hours?" -> ANSWERED

  HR Employee (employee_hr / sarah):
    -> Can ask about HR docs, policies, employee data
    -> Asks "What is the HR leave policy?" -> ANSWERED
    -> Asks "Show me system configs" -> DENIED

  IT Admin (it_admin / alex):
    -> Can ask about everything + system configs
    -> All questions answered

HOW IT WORKS:
  1. User provides their username (simulating a login)
  2. We generate a real OAuth JWT for them (using iam.py)
  3. JWT gets verified -> SecurityContext created
  4. SecurityContext stored IN THE AGENT STATE
  5. Every tool checks state["security_context"].scopes
     before doing anything
  6. The LLM system prompt is PERSONALIZED per user
=============================================================
"""

# -- IMPORTS ----------------------------------------------
import os
import sys
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import Annotated, Optional
from typing_extensions import TypedDict

# Add backend to path so we can import iam.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.iam import (
    SecurityContext,
    generate_oauth_jwt,
    authenticate_request,
    PERSONAS
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


# -- STEP 1: STATE WITH SECURITY CONTEXT ------------------
#
# KEY ADDITION vs previous levels:
# We add 'security_context' to the state.
# This travels with EVERY message through the entire graph.
# Every node and tool can read it.
#
class SecureAgentState(TypedDict):
    messages:         Annotated[list, add_messages]
    security_context: Optional[SecurityContext]  # <- WHO IS THIS USER?
    access_log:       list                        # <- what they accessed


# -- STEP 2: IAM-AWARE TOOLS ------------------------------
#
# Each tool receives state, checks permissions,
# then either does the work or returns "access denied".
#
# This is RBAC (Role Based Access Control) in action.
#

@tool
def get_public_info(query: str, state: dict = None) -> str:
    """
    Get public company information like working hours, office location,
    general company details. Available to ALL users including guests.
    """
    public_info = {
        "working hours": "Monday to Friday, 9am to 6pm. Remote work 2 days/week.",
        "location":      "Chennai office: Tidel Park, OMR. Mumbai office: BKC.",
        "about":         "We are a 500-person tech company founded in 2018.",
        "contact":       "General: info@company.com | Support: help@company.com",
    }
    query_lower = query.lower()
    for key, value in public_info.items():
        if key in query_lower or query_lower in key:
            return f"[PUBLIC] {value}"
    return f"[PUBLIC] General info: We are a technology company. For specific details, please contact HR."


@tool
def get_hr_policy(policy_name: str, state: dict = None) -> str:
    """
    Get HR policies like leave policy, salary structure, benefits.
    Requires 'read:hr' scope. Only available to HR employees and admins.
    """
    # -- PERMISSION CHECK --
    # This is the key pattern: check scopes BEFORE doing any work
    if state and "security_context" in state:
        ctx = state["security_context"]
        if "read:hr" not in ctx.scopes:
            return (f"[ACCESS DENIED] You ({ctx.username}, role: {ctx.role}) "
                    f"do not have 'read:hr' permission to access HR policies. "
                    f"Your current scopes: {ctx.scopes}")

    hr_policies = {
        "leave":    "Annual leave: 20 days. Sick leave: 10 days. Casual: 5 days. "
                    "Apply via HR portal 2 weeks in advance.",
        "salary":   "Salary credited last working day of month. "
                    "Grade structure: L1 (?8-12L), L2 (?12-20L), L3 (?20-35L), L4+ (?35L+)",
        "benefits": "Health insurance: ?5L family floater. "
                    "Annual flights: 2 tickets. Learning budget: ?50K/year.",
        "appraisal":"Annual appraisal in March. Mid-year check-in in September. "
                    "Ratings: 1-5 scale. Increment range: 8-25%.",
    }

    policy_lower = policy_name.lower()
    for key, value in hr_policies.items():
        if key in policy_lower or policy_lower in key:
            return f"[HR RESTRICTED] {value}"

    return f"[HR RESTRICTED] Policy '{policy_name}' not found. Available: {', '.join(hr_policies.keys())}"


@tool
def get_system_config(config_name: str, state: dict = None) -> str:
    """
    Get system configuration, server details, and technical settings.
    Requires 'write:system' scope. Only available to IT admins.
    """
    # -- PERMISSION CHECK --
    if state and "security_context" in state:
        ctx = state["security_context"]
        if "write:system" not in ctx.scopes:
            return (f"[ACCESS DENIED] You ({ctx.username}, role: {ctx.role}) "
                    f"do not have 'write:system' permission. "
                    f"System configs are restricted to IT admins only.")

    system_configs = {
        "servers":   "Production: 12 EC2 instances (us-east-1). DB: RDS PostgreSQL 15. "
                     "Cache: ElastiCache Redis 7.",
        "vpn":       "VPN: Cisco AnyConnect. Server: vpn.company.com. "
                     "Credentials in 1Password vault 'IT-Systems'.",
        "api keys":  "Keys stored in AWS Secrets Manager. "
                     "Request access via IT ticket system.",
    }

    config_lower = config_name.lower()
    for key, value in system_configs.items():
        if key in config_lower or config_lower in key:
            return f"[SYSTEM RESTRICTED] {value}"

    return f"[SYSTEM] Config '{config_name}' not found."


# -- STEP 3: LLM WITH TOOLS --------------------------------
tools = [get_public_info, get_hr_policy, get_system_config]
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
llm_with_tools = llm.bind_tools(tools)


# -- STEP 4: NODES -----------------------------------------

def agent_node(state: SecureAgentState) -> dict:
    """
    The main AI node. Personalized per user role.
    The system prompt CHANGES based on who is logged in.
    """
    ctx = state.get("security_context")

    # Build personalized system prompt based on user's role
    if ctx:
        system_content = f"""You are a secure enterprise AI assistant.

CURRENT USER:
  Name:       {ctx.username}
  Role:       {ctx.role}
  Department: {ctx.department}
  Scopes:     {', '.join(ctx.scopes)}
  Auth:       {ctx.auth_method}

TOOL ACCESS RULES:
  get_public_info  -> available to EVERYONE
  get_hr_policy    -> only if 'read:hr' in scopes
  get_system_config -> only if 'write:system' in scopes

Always tell the user if they don't have permission for something.
Be helpful within their access level. Be concise."""
    else:
        system_content = "You are a helpful assistant. User is not authenticated."

    response = llm_with_tools.invoke(
        [SystemMessage(content=system_content)] + state["messages"]
    )
    return {"messages": [response]}


def tool_node_with_state(state: SecureAgentState) -> dict:
    """
    Custom tool runner that passes security_context into each tool.
    This is how tools know who is calling them.
    """
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Inject security context into tool arguments
        tool_args["state"] = {
            "security_context": state.get("security_context")
        }

        # Find and run the right tool
        tool_map = {t.name: t for t in tools}
        if tool_name in tool_map:
            result = tool_map[tool_name].invoke(tool_args)
        else:
            result = f"Tool '{tool_name}' not found"

        from langchain_core.messages import ToolMessage
        tool_results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))

        # Log the access
        print(f"\n[TOOL] Tool called: {tool_name}({list(tool_call['args'].keys())[0] if tool_call['args'] else ''})")
        print(f"   Result: {str(result)[:80]}...")

    return {"messages": tool_results}


def should_use_tool(state: SecureAgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "done"


# -- STEP 5: BUILD THE GRAPH -------------------------------
def build_secure_agent():
    builder = StateGraph(SecureAgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node_with_state)

    builder.set_entry_point("agent")

    builder.add_conditional_edges(
        "agent",
        should_use_tool,
        {"use_tool": "tools", "done": END}
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())


# -- STEP 6: RUN IT ----------------------------------------
if __name__ == "__main__":

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[ERROR] No API key! Create .env with ANTHROPIC_API_KEY=your_key")
        exit(1)

    agent = build_secure_agent()

    print("=" * 60)
    print("LEVEL 5: IAM-Aware LangGraph Agent")
    print("=" * 60)
    print("""
AVAILABLE USERS (simulating different employees):
  1. guest_user   -> guest role   -> read:public only
  2. employee_hr  -> employee     -> read:public + read:hr
  3. it_admin     -> admin        -> full access

TEST PLAN:
  Login as guest_user, ask: "What is the leave policy?"   -> DENIED
  Login as employee_hr, ask: "What is the leave policy?"  -> ANSWERED
  Login as it_admin, ask: "Show me server configs"        -> ANSWERED
  Login as employee_hr, ask: "Show me server configs"     -> DENIED
    """)

    # Simulate login: pick a user
    print("Who are you? Choose:")
    for i, (key, user) in enumerate(PERSONAS.items(), 1):
        print(f"  {i}. {key} (role: {user['role']}, scopes: {user['scopes']})")

    choice = input("\nEnter username (e.g. 'guest_user'): ").strip()

    if choice not in PERSONAS:
        print(f"Unknown user. Using 'guest_user'")
        choice = "guest_user"

    # Generate a JWT token (simulating OAuth login)
    token = generate_oauth_jwt(choice)
    print(f"\n[OK] OAuth token generated for {choice}")
    print(f"   Token (first 50 chars): {token[:50]}...")

    # Verify token -> get SecurityContext
    security_context = authenticate_request(f"Bearer {token}")
    print(f"\n[OK] Identity verified:")
    print(f"   User:       {security_context.username}")
    print(f"   Role:       {security_context.role}")
    print(f"   Department: {security_context.department}")
    print(f"   Scopes:     {security_context.scopes}")

    # The SecurityContext lives in the agent state for the whole conversation
    config = {"configurable": {"thread_id": f"session-{choice}"}}

    print(f"\n[START] Starting secure chat session for {security_context.username}")
    print("Type 'quit' to exit\n")

    # First message initializes the state with security_context
    first_run = True

    while True:
        user_input = input(f"\n[{security_context.username}] You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input:
            continue

        if first_run:
            # First message: inject security_context into state
            result = agent.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "security_context": security_context,
                    "access_log": []
                },
                config=config
            )
            first_run = False
        else:
            # Subsequent messages: state already has security_context
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

        ai_reply = result["messages"][-1].content
        print(f"\nAgent: {ai_reply}")

    print("\n" + "=" * 60)
    print("WHAT YOU BUILT ? THE FULL PICTURE:")
    print("=" * 60)
    print("""
  AUTHENTICATION (iam.py):
    OAuth JWT generated -> verified -> SecurityContext created

  STATE (SecureAgentState):
    messages + security_context flow through the whole graph

  NODES:
    agent_node -> personalized system prompt per user role
    tool_node  -> injects security_context into every tool call

  TOOLS (IAM-aware):
    get_public_info  -> no permission check (public)
    get_hr_policy    -> checks 'read:hr' scope
    get_system_config -> checks 'write:system' scope

  EDGES:
    agent -> tool? -> YES -> tools -> agent (loop)
                  -> NO  -> END

  MEMORY:
    MemorySaver with thread_id -> full conversation history

  THIS IS PRODUCTION-LEVEL LANGGRAPH + IAM.
  This is what the Chennai job interview expects.
    """)
