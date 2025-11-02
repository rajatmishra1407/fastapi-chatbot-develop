from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import json
import re
from datetime import datetime

load_dotenv()

# -------------------
# Mini Knowledge Base
# -------------------
KNOWLEDGE_BASE = {
    "capital of australia": "Canberra",
    "capital of france": "Paris",
    "capital of japan": "Tokyo",
    "capital of usa": "Washington, D.C.",
    "capital of india": "New Delhi",
    "capital of germany": "Berlin",
    "capital of canada": "Ottawa",
    "capital of brazil": "Brasília",
    "capital of china": "Beijing",
    "capital of russia": "Moscow",
    "largest ocean": "Pacific Ocean",
    "tallest mountain": "Mount Everest",
    "speed of light": "299,792,458 meters per second",
    "boiling point of water": "100°C or 212°F at sea level",
    "number of continents": "7 continents",
    "longest river": "Nile River",
    "largest planet": "Jupiter",
    "smallest planet": "Mercury",
    "year ww2 ended": "1945",
    "inventor of telephone": "Alexander Graham Bell",
}

PROFANITY_LIST = [
    "damn", "hell", "shit", "fuck", "bitch", "bastard", "ass", "asshole",
    "dick", "piss", "crap", "slut", "whore", "fag", "nigger", "retard",
    "idiot", "stupid", "dumb", "moron", "imbecile"
]

VOWELS = set('aeiouAEIOU')

AMBIGUOUS_PATTERNS = {
    "weekend": ["Saturday", "Sunday"],
    "this weekend": ["Saturday", "Sunday"],
    "morning": ["early morning", "late morning"],
    "afternoon": ["early afternoon", "late afternoon"],
    "evening": ["early evening", "late evening"],
    "next week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "soon": ["today", "tomorrow", "this week"],
}

DAYS_OF_WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# -------------------
# LLM & Tools
# -------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key="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"
)

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform arithmetic: add, sub, mul, div"""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch stock price for symbol (e.g. 'AAPL')"""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    booking_state: dict
    booking_history: list  # NEW: Store all confirmed bookings
    in_booking_flow: bool
    input_valid: bool
    awaiting_clarification: bool
    clarification_options: list
    awaiting_confirmation: bool

# -------------------
# Helper Functions
# -------------------
def is_gibberish(text: str) -> bool:
    """Detect gibberish or nonsense input."""
    if not text or len(text.strip()) < 3:
        return False
    
    text = text.strip().lower()
    words = text.split()
    
    if len(words) > 2:
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.5:
            return True
    
    for word in words:
        if len(word) < 4 or word.isdigit():
            continue
            
        if len(set(word)) < len(word) * 0.4:
            return True
        
        if len(word) > 5:
            vowel_count = sum(1 for char in word if char in VOWELS)
            vowel_ratio = vowel_count / len(word)
            
            if vowel_ratio < 0.15:
                return True
    
    return False

def contains_profanity(text: str) -> bool:
    """Check if text contains profanity."""
    text_lower = text.lower()
    
    words = text_lower.split()
    for word in words:
        clean_word = word.strip('.,!?;:\'"')
        if clean_word in PROFANITY_LIST:
            return True
    
    for profanity in PROFANITY_LIST:
        if profanity in text_lower:
            return True
    
    return False

def validate_input(text: str) -> tuple[bool, str | None]:
    """Validate input for gibberish and profanity."""
    if not text or not text.strip():
        return False, "I didn't receive any message. Could you please try again?"
    
    if is_gibberish(text):
        return False, "I'm sorry, I didn't catch that—could you rephrase?"
    
    if contains_profanity(text):
        return False, "Let's keep our conversation respectful, please."
    
    return True, None

def check_knowledge_base(query: str) -> str | None:
    """Check if query matches knowledge base."""
    query_lower = query.lower().strip()
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            return value
    return None

def detect_contradiction(message: str) -> tuple[bool, str | None]:
    """Detect contradictory statements."""
    message_lower = message.lower()
    
    if "freezing" in message_lower or "frozen" in message_lower:
        temp_match = re.search(r'(\d+)\s*°?\s*[cf]', message_lower)
        if temp_match:
            temp = int(temp_match.group(1))
            if '°f' in message_lower or 'fahrenheit' in message_lower:
                if temp > 40:
                    return True, f"No—{temp}°F is quite warm, not freezing. Water freezes at 32°F."
            else:
                if temp > 10:
                    return True, f"No—{temp}°C is quite warm, not freezing. Water freezes at 0°C."
    
    if "hot" in message_lower and ("cold" in message_lower or "freezing" in message_lower):
        return True, "Something cannot be both hot and cold at the same time."
    
    if "daytime" in message_lower and "night" in message_lower:
        if "is daytime" in message_lower and "night" in message_lower:
            return True, "Daytime and nighttime are opposite—it cannot be both simultaneously."
    
    if ("large" in message_lower or "big" in message_lower) and ("small" in message_lower or "tiny" in message_lower):
        if "is" in message_lower:
            return True, "Something cannot be both large and small at the same time."
    
    return False, None

def is_booking_intent(message: str) -> bool:
    """
    Detect EXPLICIT booking intent only.
    Conservative detection - requires clear booking language.
    """
    booking_keywords = ["book", "reserve", "reservation", "appointment", "schedule"]
    booking_nouns = ["flight", "hotel", "restaurant", "table", "room", "ticket"]
    
    message_lower = message.lower()
    
    has_verb = any(keyword in message_lower for keyword in booking_keywords)
    has_noun = any(noun in message_lower for noun in booking_nouns)
    
    explicit_phrases = [
        "book a", "make a reservation", "reserve a", "schedule an appointment",
        "book me", "i want to book", "can i book", "i'd like to book",
        "need a reservation", "want a reservation", "continue booking",
        "continue with", "back to booking", "resume booking", "finish booking"
    ]
    
    has_explicit = any(phrase in message_lower for phrase in explicit_phrases)
    
    return (has_verb and has_noun) or has_explicit

def is_booking_query(message: str) -> bool:
    """Detect if user is asking about their bookings"""
    message_lower = message.lower()
    
    query_patterns = [
        "booking details", "my booking", "my reservation", "show booking",
        "what did i book", "booking information", "reservation details",
        "how many", "last booking", "previous booking", "booking history",
        "show my reservation", "my bookings", "slot", "slots i", "booked slot",
        "can you provide booking", "provide booking"
    ]
    
    return any(pattern in message_lower for pattern in query_patterns)

def get_next_booking_question(booking_state: dict) -> str:
    """Determine next question in booking flow."""
    
    if not booking_state.get("party_size"):
        return "How many people will be in your party?"
    
    else:
        return None

def format_booking_summary(booking_state: dict, include_header: bool = True) -> str:
    """Format booking details."""
    lines = []
    if include_header:
        lines.append("Here's your reservation:")
    
    lines.extend([
        f"👥 Party Size: {booking_state.get('party_size', 'Not set')} people",
        f"📅 Date: {booking_state.get('date', 'Not specified')}"
    ])
    
    return "\n".join(lines)

def is_confirmation_response(message: str) -> tuple[bool, bool]:
    """Check if message is yes/no confirmation."""
    message_lower = message.lower().strip()
    
    # Not a confirmation if it's a question
    if "?" in message or any(q in message_lower for q in ["what", "where", "when", "who", "why", "how"]):
        return False, False
    
    positive_patterns = ["yes", "yeah", "yep", "sure", "ok", "okay", "correct", "confirm", "that's right", "looks good", "perfect"]
    negative_patterns = ["no", "nope", "not", "wrong", "incorrect", "cancel", "change"]
    
    for pattern in positive_patterns:
        if message_lower.startswith(pattern) or message_lower == pattern or f"{pattern}," in message_lower:
            return True, True
    
    if any(pattern in message_lower for pattern in negative_patterns):
        return True, False
    
    return False, False

def extract_booking_info(message: str, booking_state: dict) -> dict:
    """Extract booking info from message."""
    message_clean = message.strip()
    for word in [", please", "please", ", thanks", "thanks"]:
        message_clean = message_clean.replace(word, "").strip()
    
    
    if not booking_state.get("party_size"):
        numbers = re.findall(r'\d+', message_clean)
        if numbers:
            booking_state["party_size"] = numbers[0]
        else:
            number_words = {
                "one": "1", "two": "2", "three": "3", "four": "4", 
                "five": "5", "six": "6", "seven": "7", "eight": "8"
            }
            message_lower = message_clean.lower()
            for word, num in number_words.items():
                if word in message_lower:
                    booking_state["party_size"] = num
                    break
            
            if not booking_state.get("party_size"):
                booking_state["party_size"] = message_clean
    
    
    return booking_state

def detect_ambiguity(message: str) -> tuple[bool, str | None, list]:
    """Detect ambiguous time/date references."""
    message_lower = message.lower()
    detected_options = []
    
    if "maybe" in message_lower:
        maybe_idx = message_lower.find("maybe")
        after_maybe = message_lower[maybe_idx + 5:].strip()
        
        for day in DAYS_OF_WEEK:
            if day in after_maybe:
                if "morning" in after_maybe:
                    detected_options.append(f"{day.capitalize()} morning")
                elif "afternoon" in after_maybe:
                    detected_options.append(f"{day.capitalize()} afternoon")
                elif "evening" in after_maybe:
                    detected_options.append(f"{day.capitalize()} evening")
                else:
                    detected_options.append(day.capitalize())
                break
    
    if " or " in message_lower:
        parts = message_lower.split(" or ")
        first_part = parts[0].strip()
        
        if "weekend" in first_part:
            detected_options.insert(0, "Saturday")
            detected_options.insert(1, "Sunday")
        elif "weekday" in first_part:
            detected_options.extend(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        else:
            for day in DAYS_OF_WEEK:
                if day in first_part:
                    if "morning" in first_part:
                        detected_options.insert(0, f"{day.capitalize()} morning")
                    elif "afternoon" in first_part:
                        detected_options.insert(0, f"{day.capitalize()} afternoon")
                    elif "evening" in first_part:
                        detected_options.insert(0, f"{day.capitalize()} evening")
                    else:
                        detected_options.insert(0, day.capitalize())
                    break
        
        seen = set()
        unique_options = []
        for opt in detected_options:
            if opt not in seen:
                seen.add(opt)
                unique_options.append(opt)
        
        if unique_options:
            return True, "your preferred time", unique_options
    
    for pattern, options in AMBIGUOUS_PATTERNS.items():
        if pattern in message_lower:
            return True, pattern, options
    
    mentioned_days = [day for day in DAYS_OF_WEEK if day in message_lower]
    if len(mentioned_days) > 1:
        return True, "day", [day.capitalize() for day in mentioned_days]
    
    return False, None, []

def format_options_list(options: list) -> str:
    """Format options into natural language."""
    if len(options) == 0:
        return ""
    elif len(options) == 1:
        return options[0]
    elif len(options) == 2:
        return f"{options[0]} or {options[1]}"
    else:
        return ", ".join(options[:-1]) + f", or {options[-1]}"

# -------------------
# Nodes
# -------------------
def input_validator(state: ChatState):
    """Validate user input."""
    last_message = state["messages"][-1].content
    
    is_valid, error_message = validate_input(last_message)
    
    if not is_valid:
        return {
            "messages": [AIMessage(content=error_message)],
            "input_valid": False
        }
    
    return {"input_valid": True}

def route_decision(state: ChatState) -> Literal["chat_node", "booking_handler", "booking_query_handler", "END"]:
    """
    Route based on intent:
    - Invalid input → END
    - Booking query (asking about bookings) → booking_query_handler
    - Active booking flow + booking-related input → booking_handler
    - Explicit new booking intent → booking_handler
    - Everything else → chat_node
    """
    if state.get("input_valid") == False:
        return "END"
    
    last_message = state["messages"][-1].content
    
    # Check if user is asking about their bookings
    if is_booking_query(last_message):
        return "booking_query_handler"
    
    # If in active booking flow, check if input is booking-related
    if state.get("in_booking_flow"):
        # If it's a confirmation response, stay in booking
        is_conf, _ = is_confirmation_response(last_message)
        if is_conf:
            return "booking_handler"
        
        # If it's clearly NOT a booking intent (like a question), switch to chat
        # This allows interruptions
        if not is_booking_intent(last_message):
            # Check if it's a question or statement that's not booking-related
            message_lower = last_message.lower()
            question_indicators = ["what", "where", "when", "who", "why", "how", "is ", "are ", "does ", "do ", "can ", "tell me"]
            
            if any(indicator in message_lower for indicator in question_indicators):
                # It's a question - route to chat and PAUSE booking
                return "chat_node"
        
        # Otherwise, continue booking flow
        return "booking_handler"
    
    # Check for new booking intent
    if is_booking_intent(last_message):
        return "booking_handler"
    
    # Everything else goes to chat
    return "chat_node"

def chat_node(state: ChatState):
    """
    Handle ALL non-booking queries.
    CLEANLY - no nagging about booking.
    """
    messages = state["messages"]
    last_message = messages[-1].content
    
    # If there was an active booking, PAUSE it (don't clear state)
    updates = {}
    if state.get("in_booking_flow"):
        # Don't clear in_booking_flow or awaiting_confirmation
        # Just pause temporarily for this interruption
        pass
    
    # Check for contradictions
    has_contradiction, correction = detect_contradiction(last_message)
    if has_contradiction:
        updates["messages"] = [AIMessage(content=correction)]
        return updates
    
    # Check knowledge base
    kb_answer = check_knowledge_base(last_message)
    if kb_answer:
        updates["messages"] = [AIMessage(content=f"{kb_answer}.")]
        return updates
    
    # Use LLM for everything else
    response = llm_with_tools.invoke(messages)
    updates["messages"] = [response]
    
    return updates

def booking_query_handler(state: ChatState):
    """Handle queries about booking history and details"""
    booking_history = state.get("booking_history", [])
    
    if not booking_history:
        return {
            "messages": [AIMessage(content="You haven't made any bookings yet. Would you like to book a table?")]
        }
    
    count = len(booking_history)
    
    # Format the most recent booking
    latest = booking_history[-1]
    
    response = f"📋 **Your Booking History**\n\n"
    response += f"Total bookings: {count}\n\n"
    response += f"**Most Recent Booking:**\n"
    response += f"👥 Party Size: {latest.get('party_size', 'Not specified')} people\n"
    response += f"📅 Date: {latest.get('date', 'Not specified')}\n"
    response += f"🕐 Booked on: {latest.get('confirmed_at', 'N/A')}\n"
    
    if count > 1:
        response += f"\n💡 You have {count - 1} other booking(s) on record."
    
    return {
        "messages": [AIMessage(content=response)]
    }

def booking_handler(state: ChatState):
    """Handle booking flow."""
    booking_state = state.get("booking_state", {})
    last_message = state["messages"][-1].content
    
    # Handle confirmation
    if state.get("awaiting_confirmation"):
        is_conf, is_positive = is_confirmation_response(last_message)
        
        if is_conf:
            if is_positive:
                # Add timestamp and save to history
                booking_with_timestamp = {
                    **booking_state,
                    "confirmed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                existing_history = state.get("booking_history", [])
                
                summary = f"✅ Perfect! Your reservation has been confirmed!\n\n{format_booking_summary(booking_state, include_header=False)}\n\nYou'll receive a confirmation email shortly. Is there anything else I can help you with?"
                
                return {
                    "messages": [AIMessage(content=summary)],
                    "booking_state": {},
                    "booking_history": existing_history + [booking_with_timestamp],  # Save to history
                    "in_booking_flow": False,
                    "awaiting_confirmation": False,
                    "awaiting_clarification": False
                }
            else:
                change_msg = "No problem! What would you like to change? (party size or date)"
                return {
                    "messages": [AIMessage(content=change_msg)],
                    "awaiting_confirmation": False,
                    "in_booking_flow": True  # Keep in booking flow for changes
                }
        else:
            # Handle field changes
            message_lower = last_message.lower()
            
            if "cancel" in message_lower or "start over" in message_lower:
                return {
                    "messages": [AIMessage(content="Booking cancelled. Let me know if you'd like to start a new reservation!")],
                    "booking_state": {},
                    "in_booking_flow": False,
                    "awaiting_confirmation": False
                }
            
           
           
            
    
    # Initialize new booking
    if not state.get("in_booking_flow"):
        booking_state = {}
        
        is_ambiguous, ambiguous_term, options = detect_ambiguity(last_message)
        
        if is_ambiguous:
            options_text = format_options_list(options)
            clarification_msg = f"Sure! I'll help you book a table. Just to confirm, do you mean {options_text}?"
            
            return {
                "messages": [AIMessage(content=clarification_msg)],
                "booking_state": booking_state,
                "in_booking_flow": True,
                "awaiting_clarification": True,
                "clarification_options": options
            }
        
        
    
    # Handle clarification
    if state.get("awaiting_clarification"):
        clarification_options = state.get("clarification_options", [])
        message_lower = last_message.lower()
        matched_option = None
        
        for option in clarification_options:
            if option.lower() in message_lower:
                matched_option = option
                break
        
        if matched_option:
            booking_state["date"] = matched_option
        else:
            message_clean = last_message.strip()
            for word in [", please", "please", ", thanks", "thanks"]:
                message_clean = message_clean.replace(word, "").strip()
            booking_state["date"] = message_clean
        
        response = AIMessage(content="Great! How many people will be in your party?")
        return {
            "messages": [response],
            "booking_state": booking_state,
            "awaiting_clarification": False,
            "clarification_options": [],
            "in_booking_flow": True  # Maintain booking flow
        }
    
    # Check for ambiguity in date
    is_ambiguous, ambiguous_term, options = detect_ambiguity(last_message)
    
    if is_ambiguous and not booking_state.get("date"):
        options_text = format_options_list(options)
        clarification_msg = f"Just to confirm, do you mean {options_text}?"
        
        return {
            "messages": [AIMessage(content=clarification_msg)],
            "booking_state": booking_state,
            "awaiting_clarification": True,
            "clarification_options": options,
            "in_booking_flow": True  # Maintain booking flow
        }
    
    # Extract booking info
    booking_state = extract_booking_info(last_message, booking_state)
    
    # Get next question
    next_question = get_next_booking_question(booking_state)
    
    if next_question:
        response = AIMessage(content=next_question)
        return {
            "messages": [response],
            "booking_state": booking_state,
            "in_booking_flow": True  # Maintain booking flow
        }
    else:
        # All slots filled
        confirmation_msg = f"{format_booking_summary(booking_state)}\n\nDoes this look correct? (yes/no)"
        
        return {
            "messages": [AIMessage(content=confirmation_msg)],
            "booking_state": booking_state,
            "awaiting_confirmation": True,
            "in_booking_flow": True  # CRITICAL FIX: Keep booking flow active
        }

tool_node = ToolNode(tools)

# -------------------
# Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot_clean.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# Graph
# -------------------
graph = StateGraph(ChatState)

graph.add_node("input_validator", input_validator)
graph.add_node("chat_node", chat_node)
graph.add_node("booking_handler", booking_handler)
graph.add_node("booking_query_handler", booking_query_handler)
graph.add_node("tools", tool_node)

graph.add_edge(START, "input_validator")

graph.add_conditional_edges(
    "input_validator",
    route_decision,
    {
        "chat_node": "chat_node",
        "booking_handler": "booking_handler",
        "booking_query_handler": "booking_query_handler",
        "END": END
    }
)

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')
graph.add_edge('chat_node', END)
graph.add_edge('booking_handler', END)
graph.add_edge('booking_query_handler', END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)