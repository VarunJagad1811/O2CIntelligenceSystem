Agentic O2C
Agentic Intelligence Integration for Explainable Order-to-Cash Process Risk Management
Accepted for presentation at the LION20 Conference

Overview
Order-to-Cash (O2C) processes are historically plagued by manual review bottlenecks, regulatory compliance delays, and fragmented logistics decision-making. Traditional Process Mining identifies these inefficiencies, while predictive machine learning forecasts them. However, a prescriptive gap remains wherein systems cannot autonomously execute corrective actions.

Agentic O2C is a multimodal Decision Intelligence architecture that bridges this gap. By combining a Random Forest predictive engine, Tree SHAP for causal feature attribution, and a Multi-Agent Large Language Model orchestration layer, the system functions as a deterministic, self-correcting state machine. It autonomously retrieves enterprise policies, calculates alternative shipping rates, and formulates compliant, cost-saving business resolutions.

Key Features

Predictive ML and Causal Explainability: Utilizes a Random Forest classifier (94.1 percent test accuracy) cached in RAM for high-speed inference, coupled with exact Tree SHAP values to isolate root causal drivers such as weight or international status.

Parallel Mixture of Experts: Deploys specialized LangChain agents (Logistics Manager and Compliance Officer) running in parallel via a thread pool executor to optimize for both speed and regulatory risk.

Retrieval-Augmented Generation (RAG): Eliminates LLM hallucination. The Compliance Agent autonomously queries a FAISS Vector Database to retrieve exact enterprise policies based on order attributes.

Autonomous Tool Execution: The Executive Orchestrator (Finance Director) executes real-time Python backend functions to interact with simulated ERP APIs and dynamically pull cost-saving logistics alternatives.

Actor-Critic Self-Reflection Loop: A zero-hallucination deployment mechanism. An autonomous Risk Auditor rigorously evaluates drafted action plans against SLAs and budgets. Rejected plans force the system into a deterministic feedback loop for automatic revision before user visibility.

System Architecture
The architecture strictly decouples deterministic machine learning from stochastic agentic reasoning across four main layers:

Frontend Interface: A user-facing dashboard for process visualization and prescriptive simulation.

Cached ML Engine: Holds the Random Forest model and SHAP explainer in RAM for high-speed inference, preventing redundant disk reads.

Tool Execution Layer: Contains the FAISS Vector Database for policy retrieval and the simulated ERP pricing application programming interface.

Agentic Orchestration: The cloud-based inference engine powering the Llama-3.1 agents, managed through LangChain.

The Agentic Workflow

Trigger: The user inputs an order. The predictive model flags manual review risk, and the SHAP explainer isolates the causal drivers.

Parallel Evaluation: The causal drivers are injected into the Logistics and Compliance agents. The Compliance agent uses RAG to fetch grounded policy context.

Executive Synthesis: The Finance Director receives conflicting advice from the parallel agents, triggers the ERP tool for live rates, and drafts a prescriptive business plan.

Audit and Self-Correction: The Risk Auditor evaluates the draft. If rejected due to logic or compliance flaws, it forces a rewrite. If approved, the final plan is rendered to the dashboard.

Tech Stack

Frontend: Streamlit

Machine Learning: Scikit-Learn, SHAP

Agentic Framework: LangChain, Groq API (Llama-3.1)

Knowledge Retrieval: FAISS, HuggingFace Embeddings

Data Processing: Pandas, NumPy, Joblib

Installation and Setup
To run this project locally, begin by downloading or cloning the repository to your local machine. Next, create and activate a Python virtual environment to keep dependencies isolated.

Install all required dependencies listed in the requirements text file using your package manager. You will also need to configure your environment variables by creating a configuration file in the root directory and adding your Groq API key.

Once the environment is configured and dependencies are installed, launch the main application file using Streamlit to view the dashboard in your web browser.

Project Structure
The repository is organized into specific functional directories:

The root directory contains the main application entry point and dependency requirements.

The modules folder holds the core logic, including the agentic AI definitions, machine learning loading scripts, and custom user interface rendering functions.

The models folder stores the pre-trained Random Forest model and data encoders.

The data folder houses the processed datasets and the FAISS vector database index files used for RAG policy retrieval.

Performance Metrics

Predictive Accuracy: 94.1 percent Test Accuracy and 0.95 High-Risk Recall.

Agentic Latency: Approximately 3.7 seconds with a full rewrite loop, and 2.8 seconds with no rewrite required.

Tool Invocation: 100 percent accuracy for Compliance RAG retrieval and 96 percent accuracy for Finance ERP checks.

Self-Correction: 18 percent auditor rejection rate, ensuring strict SLA enforcement and zero-hallucination deployment.
