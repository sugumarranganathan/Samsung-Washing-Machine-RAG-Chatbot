# 🧺 Samsung Washing Machine Technical Support AI

An AI-powered, RAG-based technical support chatbot designed to answer Samsung washing machine questions using information retrieved from the provided technical documentation.

The application combines **Retrieval-Augmented Generation (RAG)**, **Qdrant vector search**, **Groq LLM**, **FastAPI**, **Docker**, **AWS Lambda**, **Amazon API Gateway**, **Amazon ECR**, **GitHub Actions CI/CD**, and a modern **GitHub Pages frontend**.

---

## 🌐 Live Demo

**Samsung Washing Machine AI Support**

https://sugumarranganathan.github.io/Samsung-Washing-Machine-RAG-Chatbot/

Users can ask technical questions such as:

- How do I start a wash cycle?
- Why is my washing machine vibrating?
- Why is the washing machine door locked?
- What should I do if water remains in the drum?
- How do I clean the washing machine?
- What does error code 5C mean?

Questions outside the available technical documentation are handled safely by informing the user that the required information is not available in the provided manual.

---

# 🎯 Problem Statement

Washing machine users frequently depend on technical manuals to troubleshoot operational problems, error codes, vibration, drainage, door-lock, and maintenance issues.

Traditional manuals can be difficult to search and understand quickly.

This project provides an AI-powered technical support interface that allows customers to ask questions in natural language and receive concise answers based on the available washing machine documentation.

---

# 💡 Solution

The system uses **Retrieval-Augmented Generation (RAG)**.

Instead of allowing the LLM to answer entirely from general knowledge, the application retrieves relevant information from the washing machine documentation using vector search.

The retrieved information is then provided to the LLM to generate the final answer.

### RAG Flow

```text
Customer Question
        ↓
FastAPI API
        ↓
RAG Pipeline
        ↓
Qdrant Vector Search
        ↓
Relevant Manual Content
        ↓
Groq LLM
        ↓
Technical Answer
        ↓
Customer

----

✨ Key Features
🤖 AI-powered technical support
📚 Retrieval-Augmented Generation
🔎 Qdrant vector similarity search
🧠 Groq LLM integration
📖 Manual/document-based answers
🛡️ Prevents unsupported general answers
⚡ FastAPI REST API
☁️ AWS Lambda serverless backend
🌐 Amazon API Gateway
🐳 Docker containerization
📦 Amazon ECR container registry
🚀 GitHub Actions CI/CD
💻 GitHub Pages frontend
💬 Conversation history stored locally
📋 Copy answer functionality
🔄 Retry and error handling
✨ Modern animated customer interface
📱 Responsive web UI
🏗️ System Architecture
                    ┌──────────────────────┐
                    │      Customer        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   GitHub Pages UI    │
                    │      index.html      │
                    └──────────┬───────────┘
                               │
                               │ HTTPS POST /chat
                               ▼
                    ┌──────────────────────┐
                    │   Amazon API Gateway │
                    │      HTTP API        │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │          AWS Lambda            │
              │ Samsung-Washing-Machine-RAG-  │
              │           Chatbot              │
              └───────────────┬────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │       FastAPI        │
                    │      REST API        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      RAG Pipeline    │
                    └──────────┬───────────┘
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
              ┌─────────────┐     ┌─────────────┐
              │   Qdrant    │     │   Groq LLM  │
              │ Vector DB   │     │ Answer Gen.  │
              └─────────────┘     └─────────────┘
🔄 Application Workflow
User enters question
        ↓
Frontend sends POST request
        ↓
API Gateway receives request
        ↓
Lambda executes FastAPI application
        ↓
RAG retrieves relevant manual information
        ↓
Qdrant performs vector search
        ↓
Relevant context is provided to Groq LLM
        ↓
LLM generates technical response
        ↓
Lambda returns JSON response
        ↓
Frontend displays answer
🧠 Retrieval-Augmented Generation

The project uses RAG to improve answer reliability.

The basic process is:

Technical Documents
        ↓
Document Processing
        ↓
Text Chunks
        ↓
Embeddings
        ↓
Qdrant Vector Database
        ↓
User Question
        ↓
Similarity Search
        ↓
Relevant Context
        ↓
Groq LLM
        ↓
Final Answer

This helps the chatbot answer questions using the available washing machine documentation.

---
🚀 DevOps, MLOps & CI/CD Implementation
Technologies and Practices Used in the Project

This project integrates DevOps practices, CI/CD automation, MLOps components, cloud deployment, and serverless architecture to build and deploy the RAG-based AI technical support chatbot.

| Area                            | Technologies / Practices Used                               | Status     |
| ------------------------------- | ----------------------------------------------------------- | ---------- |
| **DevOps**                      | GitHub, GitHub Actions, Docker, AWS, OIDC                   | ✅ Used     |
| **Continuous Integration (CI)** | Dependency installation, Python `py_compile` checks         | ✅ Used     |
| **Continuous Deployment (CD)**  | Docker → Amazon ECR → AWS Lambda automatic deployment       | ✅ Used     |
| **Continuous Testing (CT)**     | Automated Python validation                                 | 🟡 Basic   |
| **MLOps**                       | RAG, embeddings, Qdrant, Groq, Docker, automated deployment | 🟡 Partial |
| **Cloud Deployment**            | AWS Lambda, API Gateway, Amazon ECR                         | ✅ Used     |
| **Serverless Architecture**     | AWS Lambda, API Gateway                                     | ✅ Used     |
| **Frontend Deployment**         | GitHub Pages                                                | ✅ Used     |

-----

🔄 CI/CD Pipeline
Developer
    ↓
GitHub Repository
    ↓
GitHub Actions
    ↓
Continuous Integration (CI)
    ↓
Python Validation
    ↓
Docker Build
    ↓
Amazon ECR
    ↓
Continuous Deployment (CD)
    ↓
AWS Lambda
    ↓
API Gateway
    ↓
Customer
🧠 AI / MLOps Pipeline
Technical Manual
       ↓
Document Processing
       ↓
Embeddings
       ↓
Qdrant Vector Database
       ↓
RAG Retrieval
       ↓
Groq LLM
       ↓
Technical Support Answer
📌 Project Classification

DevOps: ✅ Implemented
CI: ✅ Implemented
CD: ✅ Implemented
CT: 🟡 Basic implementation
MLOps: 🟡 Partial implementation
AWS Cloud Deployment: ✅ Implemented
Serverless: ✅ Implemented

-----

📚 Supported Technical Topics

The chatbot can provide information related to topics such as:

Washing Cycle
Starting a wash cycle
Selecting a washing program
Adding detergent
Vibration
Unbalanced laundry
Machine leveling
Stable floor requirements
Door Lock
Door remaining locked
Water remaining in the drum
Safe door-release procedure
Drainage
Drain hose
Drain filter
Water remaining in the drum
Maintenance
Cleaning detergent drawer
Cleaning drain filter
Cleaning/self-clean cycle
Error Codes

The chatbot can answer supported error-code questions when the information is available in the provided documentation.

🚫 Out-of-Scope Questions

The chatbot is designed to stay within the available technical documentation.

For example:

User:
What is the capital of France?

AI:
I don't have enough information in the provided manual.

This prevents the application from presenting unrelated general knowledge as washing-machine technical information.

🖥️ Frontend

The frontend is implemented using:

HTML
CSS
JavaScript
Fetch API
GitHub Pages

The interface provides:

Modern responsive layout
Animated UI effects
AI online indicator
Chat bubbles
Quick question suggestions
Loading/Thinking state
Retry/error handling
Copy-answer button
Clear chat
Conversation history
Character counter
Enter-to-send support

Conversation history is stored locally in the browser so users can continue viewing previous questions and answers.

⚡ FastAPI Backend

FastAPI provides the REST API layer between the frontend and the RAG application.

API Endpoint
POST /chat
Request
{
  "query": "Why is my washing machine vibrating?"
}
Response
{
  "query": "Why is my washing machine vibrating?",
  "answer": "Excessive vibration usually happens when the machine is unbalanced or not level..."
}

FastAPI is integrated with AWS Lambda using the Mangum adapter.

☁️ AWS Serverless Deployment

The backend is deployed using AWS serverless infrastructure.

AWS Components
Amazon API Gateway
        ↓
AWS Lambda
        ↓
Docker Container Image
        ↓
Amazon ECR
Lambda Function
Samsung-Washing-Machine-RAG-Chatbot
API Gateway
Samsung-Washing-Machine-RAG-API
API Route
POST /chat
🐳 Docker

The backend is packaged as a Docker container.

Docker provides a consistent runtime environment containing:

Python
FastAPI
RAG application
Required dependencies
Lambda runtime configuration

The container image is stored in Amazon ECR.

📦 Amazon ECR

The Docker image is pushed to:

samsung-washing-machine-rag-chatbot

Each GitHub Actions build uses the Git commit SHA as the Docker image tag.

Example:

samsung-washing-machine-rag-chatbot:<commit-sha>

This provides a unique image version for each deployment.

🚀 CI/CD with GitHub Actions

The project uses GitHub Actions to automate backend deployment.

Whenever code is pushed to the main branch:

Git Push
   ↓
GitHub Actions
   ↓
Python CI Tests
   ↓
Install Dependencies
   ↓
Compile Python Files
   ↓
Docker Build
   ↓
Amazon ECR
   ↓
Update AWS Lambda
   ↓
Lambda Deployment Complete
🔐 GitHub Actions AWS Authentication

GitHub Actions authenticates with AWS using OIDC.

The workflow uses the IAM role:

GitHubActions-SamsungRAG-Deploy

This avoids storing long-lived AWS access keys directly inside GitHub repository secrets.

🧪 CI Tests

The GitHub Actions pipeline validates the Python application before deployment.

Current checks include:

python -m py_compile app/main.py
python -m py_compile app/rag.py

The Docker image is only built after the CI test job succeeds.

📂 Project Structure
Samsung-Washing-Machine-RAG-Chatbot/
│
├── .github/
│   └── workflows/
│       └── deploy.yml
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── rag.py
│
├── documents/
│   └── technical documentation
│
├── index.html
├── Dockerfile
├── requirements.txt
└── README.md
🔌 API Testing

The backend can be tested using cURL.

Example
curl -X POST \
"https://uji26268tj.execute-api.us-east-1.amazonaws.com/chat" \
-H "Content-Type: application/json" \
-d '{"query":"Why is my washing machine vibrating?"}'
Example Response
{
  "query": "Why is my washing machine vibrating?",
  "answer": "Excessive vibration usually happens when the machine is unbalanced or not level..."
}
🧪 Example Questions
Question
How do I start a wash cycle?
Answer
1. Load the clothes into the drum.
2. Add the appropriate detergent.
3. Close the door securely.
4. Select the desired wash cycle.
5. Press the Start button.
Question
Why is my washing machine vibrating?
Answer
Excessive vibration usually happens when the load is
unbalanced or the machine isn't level.

Redistribute the clothes evenly inside the drum and
check that the machine is installed on a stable,
level surface.
Question
Why is the washing machine door locked?

The chatbot provides the relevant information from the washing-machine documentation.

---

🛠️ Technologies Used

| Technology          | Purpose                              |
| ------------------- | ------------------------------------ |
| Python              | Backend development                  |
| FastAPI             | REST API framework                   |
| RAG                 | Document-grounded question answering |
| Qdrant              | Vector database/search               |
| Groq                | Large Language Model                 |
| Mangum              | FastAPI-to-Lambda adapter            |
| Docker              | Containerization                     |
| AWS Lambda          | Serverless backend                   |
| Amazon API Gateway  | HTTP API                             |
| Amazon ECR          | Docker image registry                |
| GitHub Actions      | CI/CD automation                     |
| GitHub Pages        | Frontend hosting                     |
| HTML/CSS/JavaScript | Customer UI                          |

----

📈 Deployment Architecture
                  GitHub Repository
                         │
                         ▼
                 GitHub Actions
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
          CI Testing          Docker Build
                                    │
                                    ▼
                              Amazon ECR
                                    │
                                    ▼
                                AWS Lambda
                                    │
                                    ▼
                             API Gateway
                                    │
                                    ▼
                             GitHub Pages
                                    │
                                    ▼
                                Customer
🔁 Deployment Process
Backend
1. Modify Python/RAG code
2. Commit changes
3. Push to main
4. GitHub Actions starts
5. CI tests execute
6. Docker image is built
7. Image is pushed to ECR
8. Lambda is updated
9. New backend version becomes available
Frontend
1. Modify index.html
2. Commit changes
3. Push to main
4. GitHub Pages workflow runs
5. Website is updated
🧩 Design Goals

The project focuses on:

Reliable technical answers
Document-grounded responses
Serverless deployment
Automated deployment
Simple customer experience
Scalable architecture
Maintainable code
Modern web UI
📌 Current Deployment Status
Frontend                  ✅
GitHub Pages              ✅
FastAPI Backend           ✅
RAG Pipeline              ✅
Qdrant                    ✅
Groq LLM                  ✅
Docker                    ✅
Amazon ECR                ✅
AWS Lambda                ✅
API Gateway               ✅
CORS                      ✅
Conversation History      ✅
Retry/Error Handling      ✅
GitHub Actions CI/CD      ✅

👨‍💻 Developed by

R. Sugumar, M.B.A.,

