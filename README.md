<div align="center">

<h1 align="center">🧺 Samsung Washing Machine RAG Chatbot</h1>

<p align="center">
  <strong>⚡ Serverless AWS Lambda & RAG</strong><br>
  <srong> **RAG • Qdrant Vector Search • Groq LLM • FastAPI • Docker • AWS Lambda • Amazon API Gateway • Amazon ECR • GitHub Actions CI/CD • DevOps • MLOps</br>**
</p>
### 🌐 [LIVE DEMO](https://sugumarranganathan.github.io/Samsung-Washing-Machine-RAG-Chatbot/)

</div>

---

##  Overview

**Samsung Washing Machine Technical Support AI** is an AI-powered, Retrieval-Augmented Generation (RAG) chatbot that answers washing-machine technical questions using information retrieved from the provided technical documentation.

The application combines **RAG, Qdrant vector search, Groq LLM, FastAPI, Docker, AWS Lambda, Amazon API Gateway, Amazon ECR, GitHub Actions CI/CD, and a modern GitHub Pages frontend**.

The chatbot is designed to stay grounded in the available documentation. When a question is outside the provided manual, it responds that the required information is not available instead of presenting unrelated general knowledge as technical support.

> **Documentation note:** The project uses a synthetic technical-support test document for the RAG demonstration; it is not an official Samsung manual and does not represent specifications or procedures for a real Samsung model.

---

# 🎯 Problem Statement

Washing-machine users often depend on technical manuals to troubleshoot operational problems, error codes, vibration, drainage, door-lock, and maintenance issues.

Traditional manuals can be difficult to search and understand quickly.

This project provides an AI-powered technical-support interface where customers can ask questions in natural language and receive concise answers grounded in the available washing-machine documentation.

---

# 💡 Solution

The application uses **Retrieval-Augmented Generation (RAG)**.

Instead of allowing the LLM to answer entirely from general knowledge, the system:

1. Receives the customer's question.
2. Searches the vector database for relevant manual content.
3. Retrieves the most relevant context.
4. Sends the retrieved context to the Groq LLM.
5. Generates a technical answer based on the available documentation.
6. Returns the answer to the customer through the API.

### 🧠 RAG Flow

```text
Technical Documentation
          ↓
   Document Processing
          ↓
       Text Chunks
          ↓
      Embeddings
          ↓
   Qdrant Vector DB
          ↓
    Customer Question
          ↓
    Similarity Search
          ↓
   Relevant Context
          ↓
       Groq LLM
          ↓
    Final Answer
```

---

# ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **AI Technical Support** | Natural-language support for washing-machine questions |
| 📚 **RAG** | Grounds answers in the available technical documentation |
| 🔎 **Qdrant Search** | Vector similarity search for relevant manual content |
| 🧠 **Groq LLM** | Generates the final technical response |
| 🛡️ **Out-of-Scope Handling** | Avoids answering unrelated questions not supported by the manual |
| ⚡ **FastAPI REST API** | Provides the backend API layer |
| ☁️ **AWS Lambda** | Serverless backend execution |
| 🌐 **API Gateway** | Public HTTPS API endpoint |
| 🐳 **Docker** | Containerized backend deployment |
| 📦 **Amazon ECR** | Stores Lambda container images |
| 🚀 **GitHub Actions** | Automated CI/CD pipeline |
| 💻 **GitHub Pages** | Frontend hosting |
| 💬 **Conversation History** | Stores chat history locally in the browser |
| 📋 **Copy Answer** | Allows customers to copy AI responses |
| 🔄 **Retry & Error Handling** | Handles network, timeout, server, and invalid-response failures |
| ✨ **Animated UI** | Modern responsive customer interface |
| 📱 **Responsive Design** | Works across desktop and mobile layouts |

---

# 🏗️ System Architecture

```text
                         👤 CUSTOMER
                              │
                              ▼
                  ┌───────────────────────┐
                  │    GitHub Pages UI    │
                  │       index.html      │
                  │    HTML / CSS / JS    │
                  └───────────┬───────────┘
                              │
                              │ HTTPS POST /chat
                              ▼
                  ┌───────────────────────┐
                  │   Amazon API Gateway  │
                  │       HTTP API        │
                  └───────────┬───────────┘
                              │
                              ▼
            ┌─────────────────────────────────┐
            │           AWS Lambda             │
            │ Samsung-Washing-Machine-RAG-    │
            │            Chatbot               │
            └───────────────┬─────────────────┘
                            │
                            ▼
                  ┌───────────────────────┐
                  │       FastAPI         │
                  │       REST API        │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │      RAG Pipeline     │
                  └───────────┬───────────┘
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
            ┌──────────────┐      ┌──────────────┐
            │    Qdrant    │      │   Groq LLM   │
            │  Vector DB   │      │ Answer Gen.  │
            └──────────────┘      └──────────────┘
```

---

# 🔄 End-to-End Application Workflow

```text
Customer enters question
          ↓
Frontend sends POST /chat
          ↓
Amazon API Gateway
          ↓
AWS Lambda
          ↓
FastAPI
          ↓
RAG Pipeline
          ↓
Qdrant similarity search
          ↓
Relevant manual context
          ↓
Groq LLM
          ↓
Technical support answer
          ↓
Lambda returns JSON
          ↓
Frontend displays answer
```

---

# 📖 Supported Technical Topics

### 🧺 Washing Cycle
- Starting a wash cycle
- Selecting a washing program
- Adding detergent
- Avoiding overload

### 📳 Vibration & Noise
- Unbalanced laundry
- Machine leveling
- Stable surface requirements

### 🔒 Door Lock
- Door remaining locked
- Water remaining in the drum
- Safe door-release guidance

### 💧 Drainage
- Drain hose
- Drain filter
- Water remaining in the drum

### 🧹 Cleaning & Maintenance
- Cleaning the detergent drawer
- Cleaning the drain filter
- Cleaning/self-clean cycle

### ⚠️ Error Codes
Supported error-code questions can be answered when the relevant information exists in the provided documentation.

---

# 🚫 Out-of-Scope Question Handling

The chatbot is designed to remain within the available technical documentation.

### Example

```text
User:
What is the capital of India?

AI:
I don't have enough information in the provided manual.
```

This prevents the system from presenting unrelated general knowledge as washing-machine technical support.

---

# 🖥️ Customer Frontend

The customer-facing interface is built using:

- HTML
- CSS
- JavaScript
- Fetch API
- GitHub Pages

### UI capabilities

```text
✨ Modern animated interface
💬 Chat bubbles
🤖 AI online indicator
💡 Quick question suggestions
⏳ Thinking/loading state
📋 Copy answer
🗑️ Clear chat
💾 Local conversation history
🔄 Retry failed requests
⚠️ Error handling
🔢 Character counter
↵ Enter-to-send
📱 Responsive layout
```

Conversation history is stored locally in the customer's browser so previous messages can remain visible after a page refresh on the same browser/device.

---

# ⚡ FastAPI Backend

**FastAPI** provides the REST API layer between the frontend and the RAG application.

### Endpoint

```text
POST /chat
```

### Request

```json
{
  "query": "Why is my washing machine vibrating?"
}
```

### Response

```json
{
  "query": "Why is my washing machine vibrating?",
  "answer": "Excessive vibration usually happens when the machine is unbalanced or not level..."
}
```

FastAPI is integrated with AWS Lambda using the **Mangum** adapter.

---

# ☁️ AWS Cloud Deployment

The backend is deployed using AWS serverless infrastructure.

### AWS Components

```text
Amazon API Gateway
        ↓
AWS Lambda
        ↓
Docker Container Image
        ↓
Amazon ECR
```

### Lambda Function

```text
Samsung-Washing-Machine-RAG-Chatbot
```

### API Gateway

```text
Samsung-Washing-Machine-RAG-API
```

### API Route

```text
POST /chat
```

> **Current architecture:** The backend is AWS cloud deployed and serverless. The customer-facing frontend is currently hosted on GitHub Pages.

---

# 🐳 Docker & Amazon ECR

The backend is packaged as a Docker container containing:

- Python
- FastAPI
- RAG application
- Required dependencies
- AWS Lambda runtime configuration

The container image is stored in Amazon ECR.

### ECR Repository

```text
samsung-washing-machine-rag-chatbot
```

GitHub Actions tags each image using the Git commit SHA:

```text
samsung-washing-machine-rag-chatbot:<commit-sha>
```

This provides a unique image version for each deployment.

---

# 🚀 DevOps, MLOps & CI/CD

## Technologies and Practices Used

| Area | Technologies / Practices Used | Status |
|---|---|---|
| **DevOps** | GitHub, GitHub Actions, Docker, AWS, OIDC | ✅ Used |
| **Continuous Integration (CI)** | Dependency installation, Python `py_compile` checks | ✅ Used |
| **Continuous Deployment (CD)** | Docker → Amazon ECR → AWS Lambda automatic deployment | ✅ Used |
| **Continuous Testing (CT)** | Automated Python validation | 🟡 Basic |
| **MLOps** | RAG, embeddings, Qdrant, Groq, Docker, automated deployment | 🟡 Partial |
| **Cloud Deployment** | AWS Lambda, API Gateway, Amazon ECR | ✅ Used |
| **Serverless Architecture** | AWS Lambda, API Gateway | ✅ Used |
| **Frontend Deployment** | GitHub Pages | ✅ Used |

### 📌 Project Classification

```text
DevOps                    ✅ Implemented
Continuous Integration    ✅ Implemented
Continuous Deployment     ✅ Implemented
Continuous Testing        🟡 Basic implementation
MLOps                     🟡 Partial implementation
AWS Cloud Deployment      ✅ Implemented
Serverless Architecture   ✅ Implemented
```

> **MLOps note:** This project uses MLOps-related practices around an AI/RAG application, vector retrieval, containerization, and deployment automation. It does not currently include model training, model registry, automated retraining, or a dedicated model-evaluation pipeline.

---

# 🔄 CI/CD Pipeline

Whenever code is pushed to the `main` branch:

```text
Developer
    ↓
GitHub Repository
    ↓
GitHub Actions
    ↓
┌─────────────────────────────┐
│ Continuous Integration      │
│                             │
│ • Setup Python 3.12         │
│ • Install dependencies      │
│ • Validate Python files     │
└──────────────┬──────────────┘
               │
               ▼
         Docker Build
               ↓
         Amazon ECR
               ↓
┌─────────────────────────────┐
│ Continuous Deployment       │
│                             │
│ • Update Lambda image       │
│ • Wait for Lambda update    │
└──────────────┬──────────────┘
               ↓
          AWS Lambda
               ↓
         API Gateway
               ↓
            Customer
```

---

# 🔐 GitHub Actions AWS Authentication

GitHub Actions authenticates with AWS using **OIDC**.

The workflow uses the IAM role:

```text
GitHubActions-SamsungRAG-Deploy
```

This avoids storing long-lived AWS access keys directly inside the GitHub repository for the deployment workflow.

---

# 🧪 Continuous Integration Checks

The CI pipeline currently validates the Python application before the Docker image is built.

```bash
python -m py_compile app/main.py
python -m py_compile app/rag.py
```

The Docker build/deployment job depends on the successful completion of the CI test job.

---

# 🔌 API Testing

The backend can be tested using cURL.

### Request

```bash
curl -X POST "https://uji26268tj.execute-api.us-east-1.amazonaws.com/chat" -H "Content-Type: application/json" -d '{"query":"Why is my washing machine vibrating?"}'
```

### Response

```json
{
  "query": "Why is my washing machine vibrating?",
  "answer": "Excessive vibration usually happens when the machine is unbalanced or not level..."
}
```

---

# 🧪 Example Questions & Answers

## Example 1 — Start a Wash Cycle

**Question**

```text
How do I start a wash cycle?
```

**Answer**

```text
1. Load the clothes into the drum.
2. Add the appropriate detergent.
3. Close the door securely.
4. Select the desired wash cycle.
5. Press the Start button.
```

---

## Example 2 — Washing Machine Vibration

**Question**

```text
Why is my washing machine vibrating?
```

**Answer**

```text
Excessive vibration usually happens when the load is
unbalanced or the machine isn't level.

Redistribute the clothes evenly inside the drum and
check that the machine is installed on a stable,
level surface.
```

---

## Example 3 — Out-of-Scope Question

**Question**

```text
What is the capital of India?
```

**Answer**

```text
I don't have enough information in the provided manual.
```

---

# 📂 Project Structure

```text
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
```

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| 🐍 **Python 3.12** | Backend development |
| ⚡ **FastAPI** | REST API framework |
| 🧠 **RAG** | Document-grounded question answering |
| 🔎 **Qdrant** | Vector database and similarity search |
| 🤖 **Groq** | Large Language Model |
| 🔌 **Mangum** | FastAPI-to-AWS-Lambda adapter |
| 🐳 **Docker** | Containerization |
| ☁️ **AWS Lambda** | Serverless backend |
| 🌐 **Amazon API Gateway** | HTTP API |
| 📦 **Amazon ECR** | Container image registry |
| 🚀 **GitHub Actions** | CI/CD automation |
| 💻 **GitHub Pages** | Frontend hosting |
| HTML / CSS / JS | Customer-facing UI |

---

# 📈 Deployment Architecture

```text
                     GitHub Repository
                            │
                            ▼
                    GitHub Actions
                       /                                /                                 ▼              ▼
                CI Testing     Docker Build
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
```

---

# 🔁 Deployment Process

## Backend Deployment

```text
1. Modify Python / RAG code
2. Commit changes
3. Push to main
4. GitHub Actions starts
5. CI tests execute
6. Docker image is built
7. Image is pushed to Amazon ECR
8. Lambda function is updated
9. Lambda deployment completes
```

## Frontend Deployment

```text
1. Modify index.html
2. Commit changes
3. Push to main
4. GitHub Pages workflow runs
5. Website is updated
```

---

# 🧩 Design Goals

The project focuses on:

- 🎯 Reliable technical answers
- 📚 Document-grounded responses
- ☁️ Serverless cloud deployment
- 🚀 Automated deployment
- 💬 Simple customer experience
- 📈 Scalable architecture
- 🧹 Maintainable code
- ✨ Modern web UI

---

# 📊 Current Deployment Status

| Component | Status |
|---|---|
| Frontend | ✅ |
| GitHub Pages | ✅ |
| FastAPI Backend | ✅ |
| RAG Pipeline | ✅ |
| Qdrant | ✅ |
| Groq LLM | ✅ |
| Docker | ✅ |
| Amazon ECR | ✅ |
| AWS Lambda | ✅ |
| API Gateway | ✅ |
| CORS | ✅ |
| Conversation History | ✅ |
| Retry / Error Handling | ✅ |
| GitHub Actions CI/CD | ✅ |

---


# 👨‍💻 Developed By

### **R. Sugumar, M.B.A.**

---

<div align="center">

### ⭐ Samsung Washing Machine Technical Support AI

**RAG • FastAPI • Qdrant • Groq • Docker • AWS Lambda • API Gateway • ECR • GitHub Actions • GitHub Pages**

</div>
