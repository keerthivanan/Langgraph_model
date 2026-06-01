# LangGraph Learning Path — Start Here

## Before Anything Else

1. Get your API key: https://console.anthropic.com
2. Create a file called `.env` at the ROOT of the project:
   Location: c:\Users\91709\OneDrive\Documents\AI_langgraph_developer\.env
   Contents: ANTHROPIC_API_KEY=your_actual_key_here

3. Activate your virtual environment (run this in terminal):
   cd c:\Users\91709\OneDrive\Documents\AI_langgraph_developer
   venv\Scripts\activate

---

## The 5 Levels — Run In Order

### Level 1 — State, Nodes, Edges (NO API KEY NEEDED)
What you learn: How data flows through a graph. Nodes. Conditional edges.
How to run:
  python learn/level1_state_and_nodes.py

What to observe:
  - The state (whiteboard) traveling through each worker
  - VIP customer takes a different path than Regular customer
  - This is the foundation of ALL LangGraph apps

---

### Level 2 — First Real Chatbot (API KEY NEEDED)
What you learn: Adding Claude as a node. Messages. Basic chat.
How to run:
  python learn/level2_first_chatbot.py

What to observe:
  - Claude responding to your questions
  - Conversation history building up in state["messages"]
  - LIMITATION: restart the script = memory lost (fixed in Level 3)

---

### Level 3 — Memory / Checkpointer (API KEY NEEDED)
What you learn: MemorySaver, thread_id, multi-turn memory.
How to run:
  python learn/level3_memory.py

What to observe:
  - Tell it "my name is X" → restart conversation → it still knows!
  - Use thread_id "sarah" → separate thread_id "john" → separate memories
  - Type "history" to see saved messages

---

### Level 4 — Tools (API KEY NEEDED)
What you learn: @tool decorator, ToolNode, ReAct loop.
How to run:
  python learn/level4_tools.py

What to observe:
  - Ask "What's the weather in Chennai?" → Claude calls get_weather tool
  - Ask "What is 1250 * 12?" → Claude calls calculate tool
  - Ask "What is the leave policy?" → Claude calls get_company_info tool
  - Ask "Tell me a joke" → Claude answers directly (no tool needed)

---

### Level 5 — IAM Agent (API KEY NEEDED) ← THE FULL PROJECT
What you learn: SecurityContext in state, permission-checked tools, role-based AI.
How to run:
  python learn/level5_iam_agent.py

What to observe:
  - Login as "guest_user" → ask "What is the HR leave policy?" → DENIED
  - Login as "employee_hr" → ask same → ANSWERED
  - Login as "it_admin" → ask "Show me server configs" → ANSWERED
  - Login as "employee_hr" → ask same → DENIED

---

## The Concepts Behind Each Level

| Level | Core Concept | n8n Equivalent |
|-------|-------------|----------------|
| 1 | State + Nodes + Edges | Data object + Nodes + Connections |
| 2 | LLM as a node | AI/OpenAI node in n8n |
| 3 | Checkpointer (memory) | n8n execution history |
| 4 | Tools (ReAct loop) | HTTP Request nodes |
| 5 | IAM + SecurityContext | API authentication headers |

---

## What Each File Does

| File | Purpose |
|------|---------|
| backend/app/iam.py | OAuth/OIDC/SAML authentication (already exists) |
| learn/level1_state_and_nodes.py | LangGraph fundamentals — no LLM |
| learn/level2_first_chatbot.py | Basic chatbot with Claude |
| learn/level3_memory.py | Multi-turn memory with checkpointer |
| learn/level4_tools.py | Agent with tools (ReAct pattern) |
| learn/level5_iam_agent.py | Full project: IAM + LangGraph |
