from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from langgraph_tool_backend import chatbot, KNOWLEDGE_BASE, PROFANITY_LIST, llm
from fastapi.responses import StreamingResponse, JSONResponse,FileResponse
import json

app = FastAPI(title="LangGraph Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# **************************************** Health Check State *************************

class HealthStatus:
    def __init__(self):
        self.nlu_initialized = False
        self.knowledge_base_loaded = False
        self.ethical_filter_active = False
    
    def is_healthy(self) -> bool:
        return (
            self.nlu_initialized and 
            self.knowledge_base_loaded and 
            self.ethical_filter_active
        )
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "nlu_initialized": self.nlu_initialized,
            "knowledge_base_loaded": self.knowledge_base_loaded,
            "ethical_filter_active": self.ethical_filter_active,
            "overall_health": "healthy" if self.is_healthy() else "unhealthy"
        }

health_status = HealthStatus()

@app.get("/healthz")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers
    Returns:
        - 200 OK: All systems operational
        - 503 Service Unavailable: One or more systems failed
    """
    health_data = health_status.get_status()
    is_healthy = health_status.is_healthy()
    
    response = {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": "2025-11-02T00:00:00Z",  # Add actual timestamp if needed
        "components": {
            "nlu_module": {
                "status": "operational" if health_data["nlu_initialized"] else "failed",
                "initialized": health_data["nlu_initialized"]
            },
            "knowledge_base": {
                "status": "loaded" if health_data["knowledge_base_loaded"] else "not_loaded",
                "loaded": health_data["knowledge_base_loaded"]
            },
            "ethical_filter": {
                "status": "active" if health_data["ethical_filter_active"] else "inactive",
                "active": health_data["ethical_filter_active"]
            }
        }
    }
    
    return JSONResponse(
        status_code=200 if is_healthy else 503,
        content=response
    )

# **************************************** Startup Event *************************

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    try:
        # 1. Check NLU Module (LLM) initialization
        test_response = llm.invoke("test")
        if test_response:
            health_status.nlu_initialized = True
            print("✓ NLU module initialized")
        
        # 2. Check Knowledge Base loaded
        if KNOWLEDGE_BASE and len(KNOWLEDGE_BASE) > 0:
            health_status.knowledge_base_loaded = True
            print(f"✓ Knowledge base loaded ({len(KNOWLEDGE_BASE)} entries)")
        
        # 3. Check Ethical Filter (Profanity List) active
        if PROFANITY_LIST and len(PROFANITY_LIST) > 0:
            health_status.ethical_filter_active = True
            print(f"✓ Ethical filter active ({len(PROFANITY_LIST)} words)")
        
        # Verify chatbot graph is compiled
        if chatbot:
            print("✓ Chatbot graph compiled")
        
        if health_status.is_healthy():
            print("\n🎉 All systems operational!")
        else:
            print("\n⚠️ Warning: Some systems failed to initialize")
            print(health_status.get_status())
            
    except Exception as e:
        print(f"❌ Startup error: {e}")
        health_status.nlu_initialized = False

# **************************************** Models *************************

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    message: str
    thread_id: str

class ThreadResponse(BaseModel):
    thread_id: str

class ConversationHistory(BaseModel):
    thread_id: str
    messages: List[ChatMessage]

class StreamRequest(BaseModel):
    message: str
    thread_id: str

# **************************************** Utility Functions *************************

def generate_thread_id() -> str:
    """Generate a unique thread ID"""
    return str(uuid.uuid4())

def load_conversation(thread_id: str) -> List[Dict[str, str]]:
    """Load conversation history from a thread"""
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', [])
        
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            elif isinstance(msg, AIMessage):
                role = 'assistant'
            else:
                continue
            
            formatted_messages.append({
                'role': role,
                'content': msg.content
            })
        
        return formatted_messages
    except Exception as e:
        return []

# **************************************** API Endpoints *************************

@app.get("/")
async def root():
    """Health check endpoint"""
    return FileResponse("static/index.html")
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get a response
    """
    try:
        config = {'configurable': {'thread_id': request.thread_id}}
        
        # Invoke the chatbot
        response = chatbot.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )
        
        # Extract the last AI message
        ai_message = None
        for msg in reversed(response['messages']):
            if isinstance(msg, AIMessage):
                ai_message = msg.content
                break
        
        if ai_message is None:
            raise HTTPException(status_code=500, detail="No response generated")
        
        return ChatResponse(
            message=ai_message,
            thread_id=request.thread_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: StreamRequest):
    """
    Stream chat responses token by token
    """
    async def generate():
        try:
            config = {'configurable': {'thread_id': request.thread_id}}
            
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # Send only assistant tokens
                    chunk_data = {
                        "type": "token",
                        "content": message_chunk.content
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send end signal
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            
        except Exception as e:
            error_data = {
                "type": "error",
                "content": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/thread/new", response_model=ThreadResponse)
async def create_new_thread():
    """
    Create a new conversation thread
    """
    thread_id = generate_thread_id()
    return ThreadResponse(thread_id=thread_id)

@app.get("/thread/{thread_id}/history", response_model=ConversationHistory)
async def get_thread_history(thread_id: str):
    """
    Get conversation history for a specific thread
    """
    messages = load_conversation(thread_id)
    
    formatted_messages = [
        ChatMessage(role=msg['role'], content=msg['content']) 
        for msg in messages
    ]
    
    return ConversationHistory(
        thread_id=thread_id,
        messages=formatted_messages
    )

@app.get("/threads", response_model=List[str])
async def get_all_threads():
    """
    Get all available thread IDs
    """
    try:
        from backend_final import retrieve_all_threads
        threads = retrieve_all_threads()
        return threads
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/thread/{thread_id}")
async def delete_thread(thread_id: str):
    """
    Delete a conversation thread (Note: This is a placeholder as LangGraph 
    doesn't have a built-in delete method. In production, you'd need to 
    implement this in your checkpointer)
    """
    return {"message": f"Thread {thread_id} deletion requested (not implemented)"}

@app.get("/thread/{thread_id}/booking-history")
async def get_booking_history(thread_id: str):
    """
    Get booking history for a specific thread
    """
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        booking_history = state.values.get('booking_history', [])
        
        return {
            "thread_id": thread_id,
            "total_bookings": len(booking_history),
            "bookings": booking_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# **************************************** Run Instructions *************************
# To run this API:
# 1. Save this file as main.py
# 2. Install dependencies: pip install fastapi uvicorn
# 3. Run: uvicorn main:app --reload
# 4. Access API docs at: http://localhost:8000/docs
# 5. Access API at: http://localhost:8000


