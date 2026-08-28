# 🧺 Samsung Washing Machine AI Support Chatbot

<p align="center">

### 🤖 RAG-Based Technical Support Assistant

**An AI-powered customer support chatbot that answers Samsung washing-machine questions using a trusted product manual through Retrieval-Augmented Generation (RAG).**

<br>

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?style=for-the-badge&logo=amazonaws)
![Lambda](https://img.shields.io/badge/AWS%20Lambda-Serverless-FF9900?style=for-the-badge&logo=awslambda)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-red?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM-black?style=for-the-badge)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions)

</p>

---

## 🌟 Project Overview

**Samsung Washing Machine AI Support Chatbot** is a cloud-deployed RAG-based customer support application designed to answer technical questions about washing-machine operation, maintenance, error codes, vibration, drainage, door-lock issues, and other supported topics.

The system retrieves relevant information from the provided washing-machine manual and generates a concise response using an LLM.

The chatbot follows a **manual-grounded approach**, meaning it does not intentionally answer from general world knowledge when the required information is not available in the provided manual.

If the manual does not contain sufficient information, the chatbot responds:

> **I don't have enough information in the provided manual.**

This helps reduce unsupported or hallucinated answers.

---

# 🎯 Problem Statement

Traditional customer support systems often require users to search through lengthy product manuals or contact support teams for simple technical questions.

Customers may ask questions such as:

- How do I start a wash cycle?
- Why is my washing machine vibrating?
- What does error code 5C mean?
- Why is the washing-machine door locked?
- Why is there water remaining in the drum?
- How do I clean the washing machine?

Searching manually through documentation can be time-consuming.

This project provides an **AI-powered technical support interface** that retrieves relevant manual information and generates an easy-to-understand answer.

---

# 💡 Solution

The chatbot combines:

- **Retrieval-Augmented Generation (RAG)**
- **Sentence Transformer embeddings**
- **Qdrant vector search**
- **Groq LLM**
- **FastAPI**
- **Docker**
- **AWS Lambda**
- **Amazon ECR**
- **Amazon API Gateway**
- **CloudWatch**
- **GitHub Actions CI/CD**
- **GitHub Pages frontend**

The overall solution is:

```text
Customer Question
        │
        ▼
   Web Interface
        │
        ▼
    API Gateway
        │
        ▼
   AWS Lambda
        │
        ▼
Generate Query Embedding
        │
        ▼
   Qdrant Search
        │
        ▼
Retrieve Manual Context
        │
        ▼
 Relevance Validation
        │
        ▼
  Strict RAG Prompt
        │
        ▼
     Groq LLM
        │
        ▼
   Final Answer
        │
        ▼
    Customer UI


---

🏗️ System Architecture

                         ┌───────────────────────┐
                         │       CUSTOMER        │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    GitHub Pages UI    │
                         │   HTML / CSS / JS     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     API Gateway       │
                         │       /chat           │
                         └───────────┬───────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          AWS Lambda                  │
                  │  Samsung-Washing-Machine-RAG-       │
                  │          Chatbot                     │
                  └───────────────┬─────────────────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 │                │                │
                 ▼                ▼                ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │ Sentence     │  │   Qdrant     │  │    Groq      │
        │ Transformer  │  │ Vector DB    │  │     LLM      │
        │ Embeddings   │  │ Manual Data  │  │ Answer Gen.  │
        └──────────────┘  └──────────────┘  └──────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Washing Machine  │
                         │     Manual       │
                         └──────────────────┘

---

🔎 RAG Pipeline

The project uses Retrieval-Augmented Generation instead of relying only on the LLM.

RAG Flow

User Question
      │
      ▼
Question Validation
      │
      ▼
Query Embedding
      │
      ▼
Qdrant Vector Search
      │
      ▼
Top Relevant Manual Chunks
      │
      ▼
Similarity / Relevance Check
      │
      ▼
Manual Context
      │
      ▼
Strict RAG Prompt
      │
      ▼
Groq LLM
      │
      ▼
Customer Answer

---

🧠 Manual-Grounded AI

The chatbot is intentionally designed to avoid answering unsupported questions.

For example:

Supported question
Why is my washing machine vibrating?

The chatbot retrieves relevant manual information and responds with troubleshooting guidance.

Unsupported question
What is the capital of France?

The chatbot responds:

I don't have enough information in the provided manual.

This behavior keeps the application focused on the washing-machine support domain.

---

⚙️ Main Features
🤖 AI Technical Support

Answers washing-machine-related technical questions using the RAG pipeline.

📚 Manual-Based Knowledge

The knowledge base is built from washing-machine manual content.

🔎 Semantic Search

Customer questions are converted into embeddings and searched against the Qdrant vector database.

🛡️ Grounded Responses

The LLM is instructed to use only the retrieved manual context.

🚫 Out-of-Domain Protection

Unrelated questions are rejected when relevant manual information cannot be retrieved.

🔄 Retry & Error Handling

The backend includes retry handling for:

Embedding generation
Qdrant requests
Groq requests

---

💬 Conversation History

The customer interface stores conversation history locally for the current browser experience.

📋 Copy Answer

Customers can copy generated answers easily.

🔁 Retry

The UI provides a retry mechanism when an API request fails.

🧹 Clear Chat

Customers can clear the current conversation.

📱 Responsive UI

The frontend is designed for desktop and smaller screens.

---

🎨 Customer Interface

The frontend provides a modern AI-chat experience with:

Samsung-style branding
AI online indicator
Chat bubbles
Welcome screen
Suggested questions
Copy buttons
Retry buttons
Clear-chat functionality
Character counter
Enter-to-send support
Loading state
Error state
Conversation history

Example supported quick questions:

▶ Start a wash cycle
💧 Error 4C
🛠 Error 5C
📳 Vibration
🔒 Door locked

---

🧰 Technology Stack
Category

| Category                 | Technology                     |
| ------------------------ | ------------------------------ |
| Programming Language     | Python 3.12                    |
| Backend                  | FastAPI                        |
| API Style                | REST API                       |
| AI Architecture          | Retrieval-Augmented Generation |
| Embeddings               | Sentence Transformers          |
| Vector Database          | Qdrant                         |
| LLM                      | Groq                           |
| Containerization         | Docker                         |
| Cloud Platform           | AWS                            |
| Compute                  | AWS Lambda                     |
| Container Registry       | Amazon ECR                     |
| API Layer                | Amazon API Gateway             |
| Monitoring               | Amazon CloudWatch              |
| Frontend                 | HTML, CSS, JavaScript          |
| Frontend Hosting         | GitHub Pages                   |
| CI/CD                    | GitHub Actions                 |
| Authentication for CI/CD | AWS IAM + GitHub OIDC          |


---

☁️ AWS Cloud Deployment

The backend is deployed using a serverless AWS architecture.

GitHub Pages
      │
      ▼
API Gateway
      │
      ▼
AWS Lambda
      │
      ▼
Docker Container
      │
      ▼
Amazon ECR
      │
      ├──────────────► Qdrant
      │
      └──────────────► Groq

---

AWS Services Used
AWS Lambda

Runs the containerized FastAPI/RAG application without managing a traditional server.

Amazon ECR

Stores the Docker container image used by Lambda.

Amazon API Gateway

Provides the public REST API endpoint used by the frontend.

Amazon CloudWatch

Stores Lambda execution logs and helps diagnose runtime problems.

AWS IAM

Provides controlled permissions for AWS resources.

GitHub OIDC

Allows GitHub Actions to authenticate with AWS without storing long-lived AWS access keys in GitHub.

---

🐳 Docker Architecture

The application is packaged into a Docker image.

Source Code
    │
    ▼
Dockerfile
    │
    ▼
Docker Image
    │
    ▼
Amazon ECR
    │
    ▼
AWS Lambda

The container includes the application runtime and required Python dependencies.

---

CI/CD Pipeline

The project uses GitHub Actions for automated CI/CD.

Whenever code is pushed to the main branch:

Git Push
   │
   ▼
GitHub Actions
   │
   ▼
CI Tests
   │
   ├── Python setup
   ├── Install dependencies
   └── Python syntax checks
   │
   ▼
Docker Build
   │
   ▼
Amazon ECR
   │
   ▼
Update AWS Lambda
   │
   ▼
Wait for Lambda deployment
   │
   ▼
Verify deployment
   │
   ▼
✅ Deployment Successful

---

🔄 CI/CD Workflow

The workflow performs:

1. Checkout source code
2. Setup Python 3.12
3. Upgrade pip
4. Install requirements
5. Compile-check application files
6. Authenticate to AWS using OIDC
7. Login to Amazon ECR
8. Build Docker image
9. Tag Docker image
10. Push image to ECR
11. Update Lambda function
12. Wait for Lambda update
13. Verify Lambda deployment

This provides automated application delivery from GitHub to AWS.

---

🛠️ DevOps / MLOps / CI / CD / CT

| Area   | Implementation                                                           |
| ------ | ------------------------------------------------------------------------ |
| DevOps | Docker, AWS, GitHub Actions, IAM, CloudWatch                             |
| MLOps  | Embedding model, vector database, RAG pipeline, model loading            |
| CI     | Automated dependency installation and Python validation                  |
| CD     | Automated Docker → ECR → Lambda deployment                               |
| CT     | Automated validation/testing can be extended with a dedicated test suite |

Current implementation: CI and CD are implemented.
MLOps practices are present through the embedding model and RAG pipeline.
Continuous Testing can be expanded further with automated unit/API/integration test suit

---

📂 Project Structure
Samsung-Washing-Machine-RAG-Chatbot/
│
├── app/
│   ├── main.py
│   └── rag.py
│
├── frontend/
│   └── index.html
│
├── models/
│   └── all-MiniLM-L6-v2/
│
├── data/
│   └── washing-machine-manual/
│
├── .github/
│   └── workflows/
│       └── deploy.yml
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore

Folder names may vary depending on the final repository organization.

---

🔌 API

The backend exposes a REST API endpoint:

POST /chat
Request
{
  "query": "Why is my washing machine vibrating?"
}
Response
{
  "query": "Why is my washing machine vibrating?",
  "answer": "Excessive vibration can happen when the load is unbalanced or the machine is not level. - Redistribute the laundry evenly inside the drum. - Verify that the washing machine is installed on a stable, level surface."
}

---

🧪 API Testing

The API can be tested using curl, Postman, or the frontend.

Example:

curl -X POST "YOUR_API_ENDPOINT/chat" \
-H "Content-Type: application/json" \
-d '{"query":"Why is my washing machine vibrating?"}'
Expected behavior
Question
   ↓
API Gateway
   ↓
Lambda
   ↓
RAG
   ↓
Answer

---

🧪 Test Cases
Test 1 — Washing Machine Vibration
Input
Why is my washing machine vibrating?
Expected

The chatbot should explain that vibration can be associated with an unbalanced load or the machine not being level, when supported by the manual.

Test 2 — Wash Cycle
Input
How do I start a wash cycle?
Expected

The chatbot should retrieve the documented wash-cycle instructions.

Test 3 — Error Code
Input
What does error code 5C mean?
Expected

The chatbot should provide the manual-supported explanation and troubleshooting guidance.

Test 4 — Door Locked
Input
Why is the washing machine door locked?
Expected

The chatbot should provide the relevant manual-supported safety information.

Test 5 — Water Remaining
Input
There is water remaining in the drum.
Expected

The chatbot should provide relevant manual-supported drainage troubleshooting.

Test 6 — Washing Machine Cleaning
Input
How do I clean the washing machine?
Expected

The chatbot should provide the documented cleaning/maintenance information.

Test 7 — Out-of-Domain Question
Input
What is the capital of France?
Expected
I don't have enough information in the provided manual.

---

📊 Monitoring

AWS CloudWatch is used for Lambda logging.

The application logs important processing stages such as:

RAG REQUEST STARTED
        ↓
Embedding
        ↓
Qdrant Search
        ↓
Context Building
        ↓
Groq Request
        ↓
RAG REQUEST COMPLETED

CloudWatch helps identify:

Lambda initialization problems
Runtime exceptions
Qdrant failures
Groq failures
Request duration
Memory usage
Deployment/runtime issues

---

Security

The project uses AWS IAM and GitHub OIDC for CI/CD authentication.

GitHub Actions does not need a permanent AWS access key.

The workflow assumes the configured IAM role:

GitHubActions-SamsungRAG-Deploy

through GitHub's OIDC identity federation.

The application also keeps sensitive configuration such as API keys in environment variables rather than hard-coding them into the source code.

---

Cost Awareness

The architecture uses serverless/container-based AWS services.

Cost-related considerations include:

AWS Lambda execution duration
Lambda memory allocation
API Gateway requests
Amazon ECR storage
CloudWatch log storage
External Qdrant usage
Groq API usage

Long-running Lambda cold starts can increase execution duration, so application startup optimization is important.

---

🌐 Frontend Deployment

The customer-facing interface is hosted through GitHub Pages.

GitHub Repository
       │
       ▼
GitHub Pages
       │
       ▼
Customer Browser
       │
       ▼
API Gateway
       │
       ▼
AWS Lambda

The frontend does not directly access the vector database or LLM.

All AI processing happens through the backend API.

---

🧠 Why FastAPI?

FastAPI provides the backend REST API layer.

It handles requests such as:

POST /chat

and connects the frontend with the RAG pipeline.

Frontend
   ↓
FastAPI
   ↓
RAG Pipeline
   ↓
Qdrant + Groq

FastAPI also provides a lightweight and modern Python API framework suitable for containerized cloud deployment.

---

🔍 Why Qdrant?

Qdrant is used as the vector database.

The manual content is converted into vector representations.

When a customer asks a question:

Customer Question
       ↓
Embedding
       ↓
Vector Search
       ↓
Relevant Manual Content

This allows the chatbot to retrieve semantically relevant manual sections rather than relying only on keyword matching.

---

🤖 Why Groq?

Groq provides the LLM inference layer used to generate the final customer-facing answer from the retrieved manual context.

Retrieved Manual Context
          +
Customer Question
          ↓
       Groq LLM
          ↓
   Final Answer

The RAG prompt instructs the model to remain grounded in the supplied manual context.

---

🧩 Why Sentence Transformers?

Sentence Transformers converts the customer's question into a numerical embedding.

For example:

"Why is my washing machine vibrating?"
                  ↓
        Sentence Transformer
                  ↓
             Vector
                  ↓
              Qdrant

The vector is then used to find semantically similar manual content.

---

📈 Current Project Status

| Component                 | Status      |
| ------------------------- | ----------- |
| Customer UI               | ✅ Completed |
| Chat interface            | ✅ Completed |
| Conversation history      | ✅ Completed |
| Copy response             | ✅ Completed |
| Retry handling            | ✅ Completed |
| Error handling            | ✅ Completed |
| RAG pipeline              | ✅ Completed |
| Manual-grounded responses | ✅ Completed |
| Qdrant integration        | ✅ Completed |
| Groq integration          | ✅ Completed |
| FastAPI backend           | ✅ Completed |
| Docker containerization   | ✅ Completed |
| Amazon ECR                | ✅ Completed |
| AWS Lambda                | ✅ Completed |
| API Gateway               | ✅ Completed |
| CloudWatch logging        | ✅ Completed |
| GitHub Actions CI         | ✅ Completed |
| GitHub Actions CD         | ✅ Completed |
| GitHub OIDC               | ✅ Completed |
| AWS cloud deployment      | ✅ Completed |


---

Project Highlights
🔹 AI-Powered

Uses an LLM with retrieval-based grounding.

🔹 RAG Architecture

Retrieves relevant information before generating an answer.

🔹 Domain Specific

Designed specifically for washing-machine technical support.

🔹 Cloud Native

Deployed using AWS serverless/container services.

🔹 Containerized

The backend is packaged as a Docker image.

🔹 Automated CI/CD

GitHub Actions automatically builds, pushes, and deploys the application.

🔹 Secure AWS Authentication

GitHub OIDC is used instead of storing permanent AWS access keys.

🔹 Observable

AWS CloudWatch provides runtime logs.

🔹 Customer Friendly

The frontend provides a modern conversational interface with retry, history, copy, clear-chat, and error states.

---

Deployment Flow

The complete production deployment process is:

Developer
   │
   │ git push
   ▼
GitHub
   │
   ▼
GitHub Actions
   │
   ├── CI validation
   │
   ├── Docker build
   │
   ├── AWS OIDC authentication
   │
   ├── ECR push
   │
   ├── Lambda update
   │
   └── Deployment verification
   │
   ▼
AWS Cloud
   │
   ├── API Gateway
   │
   ├── Lambda
   │
   ├── ECR
   │
   └── CloudWatch
   │
   ▼
Customer

---


