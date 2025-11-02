**FastAPI + LangGraph Chatbot**

An AI-powered conversational chatbot built with FastAPI, LangGraph, and LangChain, designed for contextual, ethical, and multi-turn conversations — deployable via Docker.

**Objective**

This chatbot handles multi-turn dialogue, ambiguous inputs, and ethical filtering — all within a single cohesive backend.

It does the following things:
- Maintains conversational state
- Detects and clarifies ambiguous inputs
- Handles off-topic interruptions
- Filters profanity/gibberish
- Supports containerized deployment via Docker

**Core Features**
1. Conversational AI

- Uses LangGraph + LangChain for intelligent flow control

- LLM integration (OpenAI GPT-4o-mini or compatible)

- Memory-based context tracking

2. Context-Aware Dialogs

- Keeps track of conversation history

- Handles topic switching and resuming mid-conversation

3. Knowledge Base

- Answers factual queries from a built-in JSON knowledge base

- Detects contradictory user statements and responds intelligently

4. Ethical & Input Guardrails

- Blocks gibberish, spam, and profanity

- Politely corrects invalid or contradictory questions

5. Booking Assistant

- Identifies “BookReservation” intent

- Extracts fuzzy time expressions (e.g., “this weekend or maybe Monday”)

- Asks clarifying follow-up questions before confirmation

6. Health Monitoring

- /healthz endpoint checks:

- LLM initialization

- Knowledge base readiness

- Ethical filter activation

**Tech Stack** 
- Component	Technology
- Framework	FastAPI
- AI Flow Control	LangGraph
- LLM Integration	LangChain (OpenAI GPT-4o-mini)
- Search Tool	DuckDuckGo Search
- Database	SQLite (conversation checkpoints)
- Deployment	Docker
- Runtime	Uvicorn
- Language	Python 3.10+

📁 Project Structure
fastapi-chatbot-develop/
│
├── main.py                     # FastAPI app & endpoints
├── langgraph_tool_backend.py   # LangGraph logic, knowledge base, ethical filter
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container build configuration
├── chatbot_clean.db            # SQLite checkpoint store
├──  index.html                    # (Optional) Frontend or docs   
└── README.md                   # Project documentation

**Setup Instructions (Local)** 
1. Clone the Repository
**git clone https://github.com/rajatmishra1407/fastapi-chatbot-develop.git**
cd fastapi-chatbot-develop

2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate         # Windows

3. Install Dependencies
pip install --no-cache-dir -r requirements.txt

5. Run Locally
uvicorn main:app --host 0.0.0.0 --port 8000


6. Open http://localhost:8000/docs
 - for API Swagger docs.

http://localhost:8000 to send messages and make chat.

**Docker Deployment**
1. Build Docker Image
docker build -t chatbot-app .

2. Run Container
docker run -p 8000:8000 chatbot-app

INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
3. Verify Endpoints

Health Check → http://localhost:8000/healthz

API Docs → http://localhost:8000/docs

🧠 Key API Endpoints
Endpoint	Method	Description
/chat	POST	Send message to chatbot
/chat/stream	POST	Stream chatbot responses
/thread/new	POST	Create a new conversation thread
/thread/{thread_id}/history	GET	Retrieve chat history
/thread/{thread_id}/booking-history	GET	Retrieve booking details
/threads	GET	List all conversation threads
/healthz	GET	System health check
