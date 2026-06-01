"""
=============================================================
LEVEL 1 ? STATE, NODES, EDGES
=============================================================

WHAT YOU WILL LEARN:
  - What State is (the whiteboard)
  - What Nodes are (the workers)
  - What Edges are (the roads)
  - How data flows through a graph

NO LLM NEEDED. No API key needed.
This is pure LangGraph logic.

THE STORY:
  We are building a factory that processes an order.

  Order comes in
      ?
  Worker 1: Reads the order, writes customer name to whiteboard
      ?
  Worker 2: Reads customer name, decides department (VIP or Regular)
      ?
  VIP?     -> Worker 3a: Adds "VIP Treatment" note
  Regular? -> Worker 3b: Adds "Standard Treatment" note
      ?
  Worker 4: Writes final summary

  This is EXACTLY how an n8n workflow works.
  Nodes = n8n nodes
  State = the data object flowing through n8n
  Conditional Edge = n8n Switch node
=============================================================
"""

# -- IMPORTS ----------------------------------------------
from typing_extensions import TypedDict   # for defining State shape
from langgraph.graph import StateGraph, END  # the main graph builder


# -- STEP 1: DEFINE THE STATE (THE WHITEBOARD) ------------
#
# This is the shared object that travels through ALL nodes.
# Every node reads from it. Every node writes back to it.
#
# Think of it like the "data" object in an n8n workflow.
#
class OrderState(TypedDict):
    order_id:       str   # the order number
    customer_name:  str   # who placed the order
    order_value:    float # how much they spent
    customer_type:  str   # "vip" or "regular" (decided by a node)
    treatment_note: str   # what treatment they get
    final_summary:  str   # the final output


# -- STEP 2: DEFINE THE NODES (THE WORKERS) ---------------
#
# Each node is just a Python function.
# Rule 1: it receives the full state
# Rule 2: it returns ONLY the parts it changed
#

def read_order_node(state: OrderState) -> dict:
    """
    Worker 1: Reads the incoming order.
    In real life this might call a database or API.
    Here we just print and pass through.
    """
    print(f"\n[BOX] [Worker 1] Reading order #{state['order_id']}")
    print(f"   Customer: {state['customer_name']}")
    print(f"   Value: ${state['order_value']}")

    # This node doesn't change anything ? just reads and logs
    # Returns empty dict = no state change
    return {}


def classify_customer_node(state: OrderState) -> dict:
    """
    Worker 2: Decides if customer is VIP or Regular.
    VIP = spent more than $500.

    This is the node BEFORE the conditional edge.
    It writes 'customer_type' to state so the edge can read it.
    """
    print(f"\n[SEARCH] [Worker 2] Classifying customer...")

    if state["order_value"] >= 500:
        customer_type = "vip"
        print(f"   -> VIP customer! (${state['order_value']} >= $500)")
    else:
        customer_type = "regular"
        print(f"   -> Regular customer (${state['order_value']} < $500)")

    # Returns the part of state that changed
    return {"customer_type": customer_type}


def vip_treatment_node(state: OrderState) -> dict:
    """
    Worker 3a: Handles VIP customers.
    Only reached if customer_type == "vip"
    """
    print(f"\n[VIP] [Worker 3a - VIP] Adding VIP treatment for {state['customer_name']}")

    return {
        "treatment_note": f"VIP Treatment: Free shipping + gift wrap + priority support"
    }


def regular_treatment_node(state: OrderState) -> dict:
    """
    Worker 3b: Handles Regular customers.
    Only reached if customer_type == "regular"
    """
    print(f"\n[LIST] [Worker 3b - Regular] Adding standard treatment for {state['customer_name']}")

    return {
        "treatment_note": f"Standard Treatment: Regular shipping + standard support"
    }


def summarize_node(state: OrderState) -> dict:
    """
    Worker 4: Creates the final summary.
    Reads everything from state and builds a final message.
    """
    print(f"\n[OK] [Worker 4] Creating final summary...")

    summary = (
        f"ORDER COMPLETE\n"
        f"Order ID: {state['order_id']}\n"
        f"Customer: {state['customer_name']}\n"
        f"Type: {state['customer_type'].upper()}\n"
        f"Note: {state['treatment_note']}"
    )

    print(f"\n{summary}")

    return {"final_summary": summary}


# -- STEP 3: THE CONDITIONAL EDGE FUNCTION ----------------
#
# This function decides WHICH road to take after Worker 2.
# It reads the state and returns the NAME of the next node.
#
# This is EXACTLY like an n8n Switch node.
#
def route_by_customer_type(state: OrderState) -> str:
    """
    Reads state, returns the name of the next node.
    LangGraph will route to whichever node name we return.
    """
    if state["customer_type"] == "vip":
        return "vip_treatment"      # go to vip_treatment_node
    else:
        return "regular_treatment"  # go to regular_treatment_node


# -- STEP 4: BUILD THE GRAPH -------------------------------
#
# This is where we assemble everything.
# Like building the n8n workflow canvas.
#
def build_graph():
    # Create the graph, tell it what State looks like
    builder = StateGraph(OrderState)

    # Add all worker nodes
    # Format: builder.add_node("name_of_node", the_function)
    builder.add_node("read_order",        read_order_node)
    builder.add_node("classify_customer", classify_customer_node)
    builder.add_node("vip_treatment",     vip_treatment_node)
    builder.add_node("regular_treatment", regular_treatment_node)
    builder.add_node("summarize",         summarize_node)

    # Set entry point ? which node runs FIRST
    # Like the Trigger node in n8n
    builder.set_entry_point("read_order")

    # Add simple edges (always go this way)
    # read_order -> classify_customer (always)
    builder.add_edge("read_order", "classify_customer")

    # Add conditional edge (choose based on state)
    # After classify_customer, call route_by_customer_type()
    # It returns "vip_treatment" or "regular_treatment"
    builder.add_conditional_edges(
        "classify_customer",       # from this node
        route_by_customer_type,    # call this function to decide
        {
            "vip_treatment":     "vip_treatment",     # if returns "vip_treatment" -> go here
            "regular_treatment": "regular_treatment"  # if returns "regular_treatment" -> go here
        }
    )

    # Both VIP and Regular paths end at summarize
    builder.add_edge("vip_treatment",     "summarize")
    builder.add_edge("regular_treatment", "summarize")

    # After summarize -> END (done)
    builder.add_edge("summarize", END)

    # Compile ? builds the runnable graph
    return builder.compile()


# -- STEP 5: RUN IT ----------------------------------------
if __name__ == "__main__":

    graph = build_graph()

    print("=" * 50)
    print("TEST 1: VIP Customer (order value >= $500)")
    print("=" * 50)

    # This is the initial state ? like the trigger data in n8n
    result = graph.invoke({
        "order_id":       "ORD-001",
        "customer_name":  "Sarah Johnson",
        "order_value":    750.00,
        "customer_type":  "",   # empty ? Worker 2 will fill this
        "treatment_note": "",   # empty ? Worker 3 will fill this
        "final_summary":  ""    # empty ? Worker 4 will fill this
    })

    print("\n" + "=" * 50)
    print("TEST 2: Regular Customer (order value < $500)")
    print("=" * 50)

    result2 = graph.invoke({
        "order_id":       "ORD-002",
        "customer_name":  "Mike Smith",
        "order_value":    150.00,
        "customer_type":  "",
        "treatment_note": "",
        "final_summary":  ""
    })

    print("\n" + "=" * 50)
    print("WHAT YOU JUST SAW:")
    print("=" * 50)
    print("""
  1. STATE flowed through every node like a baton in a relay race
  2. Each NODE read the state, did its job, wrote back only what changed
  3. The CONDITIONAL EDGE read 'customer_type' and chose the right road
  4. Both paths ended at the same summarize node

  This is LangGraph. That's it.

  Next: Level 2 ? add an LLM so the AI does the thinking
    """)
