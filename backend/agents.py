import asyncio
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from rag import ask as rag_ask
from booking import (
    check_availability,
    create_appointment,
    cancel_appointment,
    list_appointments,
)
from database import save_message, get_history
from config import GROQ_API_KEY, MODEL


# ─────────────────────────────────────────────
#  HOW THIS FILE WORKS — READ FIRST
#
#  This file is the traffic controller.
#  It sits between main.py and the two agents.
#
#  Before this file existed:
#    main.py → rag.py → answer
#
#  After this file:
#    main.py → agents.py (supervisor decides)
#                  ↓ MEDICAL_FAQ → rag_node → answer
#                  ↓ BOOKING     → booking_node → answer
#
#  The supervisor reads the patient's message,
#  asks Groq "is this a medical question or a
#  booking request?", then routes to the right node.
#
#  LangGraph is the framework that manages this
#  state machine. Think of it like a flowchart
#  that runs in Python.
#
#  KEY CONCEPT — AgentState:
#  All nodes in the graph share one state object.
#  Each node reads from it and writes back to it.
#  The final state contains the answer.
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  GROQ CLIENT
#  Used by the supervisor to classify intent.
#  Same model as rag.py but used only for the
#  short classification task.
# ─────────────────────────────────────────────

llm = ChatGroq(
    model       = MODEL,
    api_key     = GROQ_API_KEY,
    max_tokens  = 20,       # classification only needs a few words
    temperature = 0.0       # 0 = deterministic — always same answer
)                           # for the same input (good for routing)


# ─────────────────────────────────────────────
#  AGENT STATE
#
#  TypedDict defines the shape of the shared
#  state that flows through every node.
#
#  Every node receives the full state and returns
#  a partial update — LangGraph merges the update
#  back into the state automatically.
#
#  Fields:
#    message    → patient's raw text
#    session_id → conversation ID (from localStorage)
#    intent     → set by supervisor: MEDICAL_FAQ or BOOKING
#    response   → set by whichever node handles the request
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    message    : str
    session_id : str
    intent     : str        # filled by classify_intent node
    response   : str        # filled by rag_node or booking_node


# ─────────────────────────────────────────────
#  NODE 1 — classify_intent (the supervisor)
#
#  This node reads the patient's message and
#  decides which agent should handle it.
#
#  It asks Groq to classify into one of two
#  categories:
#    MEDICAL_FAQ → question about clinic, doctors,
#                  services, hours, symptoms (general)
#    BOOKING     → anything about booking, cancelling,
#                  rescheduling, or listing appointments
#
#  Returns: state with intent set to one of those two values
# ─────────────────────────────────────────────

async def classify_intent(state: AgentState) -> AgentState:
    """
    Reads the patient message and classifies it
    as MEDICAL_FAQ or BOOKING.

    Uses a strict prompt with a short list of
    examples so the model almost always gets it right.
    """

    classification_prompt = f"""You are a medical clinic chatbot router.
Classify the patient message into exactly one of these two categories:

MEDICAL_FAQ  — questions about clinic hours, doctors, services,
               symptoms (general), medications (general), location,
               contact info, or any general health information

BOOKING      — booking an appointment, checking availability,
               cancelling an appointment, rescheduling,
               or viewing existing appointments

Examples:
  "What time does the clinic open?"   → MEDICAL_FAQ
  "Who are your doctors?"             → MEDICAL_FAQ
  "I have a headache, what should I do?" → MEDICAL_FAQ
  "Book me an appointment"            → BOOKING
  "Is Dr. Mohan free on Monday?"      → BOOKING
  "Cancel my appointment"             → BOOKING
  "Show my upcoming appointments"     → BOOKING
  "I want to see a doctor tomorrow"   → BOOKING

Patient message: "{state['message']}"

Reply with ONLY one word — either MEDICAL_FAQ or BOOKING.
Nothing else. No explanation."""

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: llm.invoke([
            HumanMessage(content=classification_prompt)
        ])
    )

    # extract the intent from the response
    # strip whitespace and uppercase for safety
    raw    = result.content.strip().upper()
    intent = "BOOKING" if "BOOKING" in raw else "MEDICAL_FAQ"

    print(f"  [supervisor] intent: {intent}")

    return {**state, "intent": intent}


# ─────────────────────────────────────────────
#  ROUTER FUNCTION
#
#  LangGraph calls this after classify_intent
#  to decide which node to go to next.
#
#  It reads state["intent"] and returns a string
#  that matches one of the graph's edge names.
#
#  Return type Literal["rag", "booking"] tells
#  Python (and LangGraph) the only valid returns.
# ─────────────────────────────────────────────

def route_intent(state: AgentState) -> Literal["rag", "booking"]:
    """
    Routes to rag_node or booking_node based on intent.
    Called by LangGraph after classify_intent runs.
    """
    if state["intent"] == "BOOKING":
        return "booking"
    return "rag"


# ─────────────────────────────────────────────
#  NODE 2 — rag_node
#
#  Handles all MEDICAL_FAQ intents.
#  Simply calls rag.ask() which already handles:
#    - vector search on clinic documents
#    - building the prompt with context + history
#    - calling Groq for the answer
#    - saving to MongoDB chat_history
#
#  This node does not need to know how RAG works —
#  it just delegates to rag.py.
# ─────────────────────────────────────────────

async def rag_node(state: AgentState) -> AgentState:
    """
    Handles MEDICAL_FAQ intent using existing rag.py pipeline.
    Retrieves clinic context from MongoDB and generates answer.
    """
    print(f"  [rag_node] answering: {state['message'][:50]}...")

    response = await rag_ask(
        question   = state["message"],
        session_id = state["session_id"]
    )

    # rag.ask() already saves messages to MongoDB
    # so we just set the response in state
    return {**state, "response": response}


# ─────────────────────────────────────────────
#  NODE 3 — booking_node
#
#  Handles BOOKING intent.
#  This node has a multi-turn conversation with
#  the patient to collect all required info,
#  then calls the right booking tool.
#
#  The booking agent needs to:
#    1. Figure out WHAT the patient wants
#       (book / cancel / check slots / list bookings)
#    2. Collect any missing info
#       (which doctor? which date? what time?)
#    3. Call the right booking tool
#    4. Return the result
#
#  This is done by giving the LLM a system prompt
#  that describes all 4 tools + their parameters,
#  and asking it to either ask a follow-up question
#  or call a tool in a structured format.
# ─────────────────────────────────────────────

# booking LLM — higher token limit than the classifier
booking_llm = ChatGroq(
    model       = MODEL,
    api_key     = GROQ_API_KEY,
    max_tokens  = 512,
    temperature = 0.3
)

BOOKING_SYSTEM = """You are the appointment booking assistant for a medical clinic.

You have access to these 4 tools:

1. CHECK_AVAILABILITY  → check free slots
   Required: doctor name, date (YYYY-MM-DD)
   Doctors available: Dr. Mohan, Dr. Kumar

2. CREATE_APPOINTMENT  → book an appointment
   Required: patient_name, phone, doctor, date (YYYY-MM-DD), time_slot, reason

3. CANCEL_APPOINTMENT  → cancel a booking
   Required: appointment_id (Ref ID), phone

4. LIST_APPOINTMENTS   → show upcoming bookings
   Required: phone number

Rules:
- Collect all required info before calling a tool
- Ask ONE question at a time if info is missing
- Date format must be YYYY-MM-DD (e.g. 2024-03-15)
- When ready to call a tool, respond with a JSON block like this:

  ```json
  {
    "tool": "CHECK_AVAILABILITY",
    "args": {"doctor": "Dr. Mohan", "date": "2024-03-15"}
  }
  ```

- After the tool result, give a friendly confirmation or follow-up question
- Never make up appointment IDs or slot availability
- If the patient wants to cancel, ask for their phone and Ref ID
"""

async def booking_node(state: AgentState) -> AgentState:
    """
    Handles BOOKING intent using booking.py tools.
    Converses with the patient to collect info,
    then calls the right tool based on their request.
    """
    print(f"  [booking_node] handling: {state['message'][:50]}...")

    # get recent chat history for context
    history = await get_history(state["session_id"], limit=10)
    history_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else SystemMessage(content=m["content"])
        for m in history
    ]

    # ask the booking LLM what to do
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: booking_llm.invoke([
            SystemMessage(content=BOOKING_SYSTEM),
            *history_messages,
            HumanMessage(content=state["message"])
        ])
    )

    content = result.content.strip()

    # ── check if the LLM wants to call a tool ──
    # look for a JSON block with "tool" and "args"
    import json, re
    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)

    if json_match:
        # parse the tool call
        try:
            tool_call = json.loads(json_match.group(1))
            tool_name = tool_call.get("tool", "")
            args      = tool_call.get("args", {})

            print(f"  [booking_node] calling tool: {tool_name}")

            # ── call the right booking tool ──
            if tool_name == "CHECK_AVAILABILITY":
                tool_result = await check_availability(
                    doctor = args.get("doctor", ""),
                    date   = args.get("date", "")
                )

            elif tool_name == "CREATE_APPOINTMENT":
                tool_result = await create_appointment(
                    patient_name = args.get("patient_name", ""),
                    phone        = args.get("phone", ""),
                    doctor       = args.get("doctor", ""),
                    date         = args.get("date", ""),
                    time_slot    = args.get("time_slot", ""),
                    reason       = args.get("reason", "General consultation")
                )

            elif tool_name == "CANCEL_APPOINTMENT":
                tool_result = await cancel_appointment(
                    appointment_id = args.get("appointment_id", ""),
                    phone          = args.get("phone", "")
                )

            elif tool_name == "LIST_APPOINTMENTS":
                tool_result = await list_appointments(
                    phone = args.get("phone", "")
                )

            else:
                tool_result = {
                    "success": False,
                    "message": "Unknown tool requested."
                }

            # use the tool result message as the response
            response = tool_result.get("message", str(tool_result))

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [booking_node] tool parse error: {e}")
            response = content   # fall back to raw LLM response

    else:
        # no tool call — LLM is asking a follow-up question
        # e.g. "Which doctor would you like to see?"
        response = content

    # save to MongoDB chat_history
    await save_message(state["session_id"], "user",      state["message"])
    await save_message(state["session_id"], "assistant", response)

    return {**state, "response": response}


# ─────────────────────────────────────────────
#  BUILD THE LANGGRAPH STATE MACHINE
#
#  StateGraph takes AgentState as its type.
#  We add 3 nodes and wire them with edges.
#
#  Graph structure:
#    START
#      ↓
#    classify_intent          (always runs first)
#      ↓ conditional edge
#    rag_node  OR  booking_node
#      ↓
#    END
# ─────────────────────────────────────────────

def build_graph():
    """
    Builds and compiles the LangGraph state machine.
    Called once when the module loads.
    Returns a compiled graph ready to invoke.
    """

    # create the graph with AgentState as the state type
    graph = StateGraph(AgentState)

    # add the 3 nodes
    graph.add_node("classify",  classify_intent)
    graph.add_node("rag",       rag_node)
    graph.add_node("booking",   booking_node)

    # set entry point — always start at classify
    graph.set_entry_point("classify")

    # add conditional edge from classify → rag or booking
    # route_intent() reads state["intent"] and returns
    # "rag" or "booking" — LangGraph uses that to pick the edge
    graph.add_conditional_edges(
        "classify",         # from this node
        route_intent,       # call this function to decide
        {
            "rag":     "rag",       # if route_intent returns "rag" → go to rag node
            "booking": "booking"    # if route_intent returns "booking" → go to booking node
        }
    )

    # both nodes end the graph after running
    graph.add_edge("rag",     END)
    graph.add_edge("booking", END)

    # compile turns the graph definition into
    # a runnable object with invoke() method
    return graph.compile()


# compile once when module loads
# main.py imports and calls:  supervisor.ainvoke(state)
supervisor = build_graph()


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
#  Called by main.py instead of rag.ask()
#
#  Takes the same arguments as rag.ask() so
#  main.py only needs one line changed.
# ─────────────────────────────────────────────

async def run(message: str, session_id: str) -> str:
    """
    Main entry point called by main.py.
    Replaces the direct call to rag.ask().

    Args:
        message    : patient's message text
        session_id : conversation session ID

    Returns:
        response string to send back to the browser
    """

    initial_state: AgentState = {
        "message"    : message,
        "session_id" : session_id,
        "intent"     : "",       # filled by classify_intent
        "response"   : ""        # filled by rag_node or booking_node
    }

    # ainvoke runs the full graph asynchronously
    final_state = await supervisor.ainvoke(initial_state)

    return final_state["response"]


# ─────────────────────────────────────────────
#  LOCAL TEST
#  Run:  python backend/agents.py
#
#  Tests routing for both intents:
#    - a MEDICAL_FAQ question
#    - a BOOKING request
# ─────────────────────────────────────────────

if __name__ == "__main__":

    async def run_tests():
        print("\n─────────────────────────────────")
        print("  agents.py — supervisor routing test")
        print("─────────────────────────────────\n")

        TEST_SESSION = "agents-test-session-001"

        # TEST 1 — should route to rag_node
        print("TEST 1 — MEDICAL_FAQ routing")
        print("  Question: 'What are the clinic hours?'")
        response = await run("What are the clinic hours?", TEST_SESSION)
        print(f"  Response: {response[:100]}...")
        print()

        # TEST 2 — should route to booking_node
        print("TEST 2 — BOOKING routing")
        print("  Question: 'I want to book an appointment'")
        response = await run("I want to book an appointment", TEST_SESSION)
        print(f"  Response: {response[:100]}...")
        print()

        # TEST 3 — edge case: ambiguous message
        print("TEST 3 — edge case routing")
        print("  Question: 'Can I see Dr. Mohan tomorrow?'")
        response = await run("Can I see Dr. Mohan tomorrow?", TEST_SESSION)
        print(f"  Response: {response[:100]}...")
        print()

        print("─────────────────────────────────")
        print("  Routing test complete.")
        print("  Check that TEST 1 answered about clinic hours")
        print("  and TEST 2+3 asked about booking details.")
        print("─────────────────────────────────\n")

    asyncio.run(run_tests())