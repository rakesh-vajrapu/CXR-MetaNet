<p align="center">
  <img src="Assets/Architecture.png" alt="CDSS Architecture" width="700" />
</p>

<h1 align="center">CXR-MetaNet — AI Diagnostic Intelligence</h1>

<p align="center">
  <strong>CXR-MetaNet: Chest X-Ray Meta-Ensemble Neural Network using Hybrid CNN–Transformer Architecture</strong><br/>
  Clinical Decision Support System with GradCAM++ Explainability and Dual-Tier RAG Reporting
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white" alt="React" />
  <img src="https://img.shields.io/badge/Vite-7-646CFF?logo=vite&logoColor=white" alt="Vite" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-4-06B6D4?logo=tailwindcss&logoColor=white" alt="TailwindCSS" />
  <img src="https://img.shields.io/badge/Azure_AI-GPT--4o--mini-0078D4?logo=microsoftazure&logoColor=white" alt="Azure" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/License-Research-yellow" alt="License" />
</p>

---

## 📌 Aim & Outcome

CDSS is an **AI-powered Clinical Decision Support System** that performs automated analysis of chest radiographs to detect and classify three thoracic conditions:

| Class | Description |
|-------|-------------|
| **Normal** | No significant pathology detected |
| **Pneumonia** | Pulmonary consolidation / lung opacity |
| **Pleural Effusion** | Fluid accumulation in the pleural space |

### What It Does

1. **Automated Radiograph Analysis** — A production-grade meta-ensemble of three state-of-the-art deep learning architectures classifies chest X-rays with high confidence in under 60 seconds.
2. **GradCAM++ Explainability** — Visual attention heatmaps show exactly *where* the AI focused during diagnosis, enabling clinical transparency.
3. **Dual-Tier RAG Reporting** — An Azure OpenAI (GPT-4o-mini) agent generates two distinct clinical narratives:
   - **Tier 2 — Radiologist Report**: Formal medical findings using standard clinical terminology, suitable for patient charts.
   - **Tier 1 — Patient Narrative**: An empathetic, plain-language explanation to improve patient health literacy.
4. **Context-Aware AI Assistant** — A chatbot that understands the scan results, heatmap localization, and patient metadata to answer follow-up clinical questions with safety guardrails.

---

## 🎯 Target Users & Clinical Impact

### For Radiologists
- **Reduced Turnaround Time** — Instant AI-assisted preliminary reads, freeing radiologists to focus on complex cases.
- **GradCAM++ Heatmaps** — Transparent attention maps that highlight regions of interest, enabling radiologists to validate AI findings and catch subtle pathologies.
- **Structured Tier 2 Reports** — Auto-generated formal reports with CURB-65 assessment recommendations for pneumonia cases.

### For Patients
- **Health Literacy** — Tier 1 patient narratives translate complex medical findings into accessible, empathetic language.
- **Interactive AI Guidance** — A chatbot that answers questions about the diagnosis in patient-friendly terms with built-in safety guardrails (emergency detection, prescription refusal).

### For Healthcare Facilities
- **High-Volume Screening** — Enables rapid triage of chest X-rays in resource-constrained settings.
- **Edge-Cloud Architecture** — Can run locally on-premises (CPU-only) or scale via cloud deployment.

---

## 🛠 Tech Stack

### Frontend
| Technology | Purpose |
|-----------|---------|
| **React 18** | Component-based UI framework |
| **Vite 7** | Lightning-fast build tool & HMR dev server |
| **Tailwind CSS 4** | Utility-first styling with dark/light theme support |
| **Framer Motion** | Page transitions & micro-animations |
| **Recharts** | Probability distribution visualization |
| **jsPDF + html2canvas** | Client-side PDF report generation |
| **Web Speech API** | Speech-to-text (mic input) & text-to-speech (voice output) |
| **Lucide React** | Premium icon library |

### Backend
| Technology | Purpose |
|-----------|---------|
| **FastAPI** | High-performance async API framework |
| **Uvicorn** | ASGI production server |
| **PyTorch 2.2+** | Deep learning inference engine |
| **timm** | Pre-trained model zoo (DenseNet, ConvNeXt, MaxViT) |
| **pytorch-grad-cam** | GradCAM++ heatmap generation |
| **scikit-learn (joblib)** | Meta-learner serialization |
| **Pillow + SciPy** | Image processing & validation gates |
| **Pandas** | Ground truth dataset management |

### Cloud / AI
| Technology | Purpose |
|-----------|---------|
| **Azure AI Foundry** | Managed LLM endpoint hosting |
| **GPT-4o-mini** | Report generation & chatbot intelligence |
| **Docker** | Backend containerization for deployment |

---

## 📁 Codebase Structure

```
CDSS Project/
│
├── api.py                        # FastAPI backend — all endpoints & ML inference
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Production container image
├── .dockerignore                 # Docker build exclusions
├── .env                          # Backend secrets (Azure keys, CORS origins)
│
├── Models/
│   ├── densenet121/
│   │   └── best_tta.pth          # DenseNet-121 checkpoint (~108 MB)
│   ├── convnext_v2_base/
│   │   └── best_tta.pth          # ConvNeXtV2-Base checkpoint (~1.3 GB)
│   ├── maxvit_base/
│   │   └── best_tta.pth          # MaxViT-Base checkpoint (~1.4 GB)
│   └── meta_learner_logistic.pkl # L2 Logistic Meta-Learner
│
├── Dataset/
│   ├── data.csv                  # Ground truth labels
│   └── images/                   # Chest X-ray images (PNG/JPG)
│
├── Assets/
│   ├── Architecture.png          # System architecture diagram
│   ├── Colab Notebook.pdf        # Model training notebook
│   └── Knowledge Base.pdf        # Clinical knowledge reference
│
└── frontend/
    ├── index.html                # SEO-optimized entry point
    ├── .env                      # Frontend env vars (API base URL)
    ├── package.json              # Node.js dependencies
    └── src/
        ├── main.jsx              # React root with ErrorBoundary
        ├── App.jsx               # Full application (2,849 lines)
        └── index.css             # Design system & animations (2,056 lines)
```

---

## 🧠 Model Architecture & Training

### The "Platinum Trio" Meta-Ensemble

CDSS employs a **stacking ensemble** of three diverse architectures, each bringing complementary inductive biases:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Input: 512×512 Chest X-Ray                  │
│                (ImageNet Normalized, Bicubic Resize)             │
└─────────────────┬──────────────────┬──────────────────┬─────────┘
                  │                  │                  │
          ┌───────▼───────┐  ┌──────▼───────┐  ┌──────▼──────┐
          │  DenseNet-121  │  │ ConvNeXtV2   │  │   MaxViT    │
          │   (108 MB)     │  │   Base       │  │    Base     │
          │                │  │  (1.3 GB)    │  │  (1.4 GB)   │
          │  Dense blocks  │  │  FCMAE pre-  │  │  Multi-axis │
          │  + feature     │  │  trained on  │  │  attention  │
          │  reuse         │  │  IN-22k      │  │  + conv     │
          └──────┬────────┘  └─────┬────────┘  └─────┬───────┘
                 │ [3 probs]       │ [3 probs]       │ [3 probs]
                 └────────┬────────┴────────┬────────┘
                          │  Concatenate     │
                          │  [9-dim vector]  │
                          ▼
                 ┌────────────────────┐
                 │  L2 Logistic       │
                 │  Meta-Learner      │
                 │  (Calibrated       │
                 │   Stacking)        │
                 └────────┬───────────┘
                          │
                  Final Prediction
                  + Confidence Score
```

| Model | Architecture | Params | Pre-training | Strength |
|-------|-------------|--------|-------------|----------|
| **DenseNet-121** | Dense connectivity with feature reuse | ~8M | ImageNet-1k | Efficient feature extraction, low memory |
| **ConvNeXtV2-Base** | Modernized ConvNet with FCMAE | ~89M | ImageNet-22k → 1k | Strong spatial priors from large-scale pre-training |
| **MaxViT-Base** | Multi-axis Vision Transformer | ~120M | ImageNet-1k | Global self-attention + local window attention |

### Training Details
- **Input Resolution**: 512×512 pixels
- **Augmentation**: Test-Time Augmentation (TTA) applied during validation
- **Ensemble Strategy**: L2-regularized Logistic Regression Meta-Learner trained on held-out TTA probabilities
- **GradCAM++ Extraction**: Applied to DenseNet-121's final feature layer to generate visual attention heatmaps

### Image Validation Pipeline (4 Gates)
Before inference, every uploaded image passes through a multi-gate validation system:

| Gate | Check | Threshold | Rejects |
|------|-------|-----------|---------|
| 1 | Color channel deviation | > 15.0 | Color photographs, selfies |
| 2 | HSV saturation ratio | > 8% | Colorful non-medical images |
| 3 | Intensity std deviation | < 15.0 | Blank / flat images |
| 4 | Sobel edge density | < 2% or > 60% | Text documents, smooth photos |

---

## 📊 Performance Metrics

> Evaluated on the **held-out 20% TTA test set** (4,161 samples: 1,200 Normal · 1,561 Pneumonia · 1,400 Pleural Effusion).

### Meta-Ensemble (L2 Logistic Stacking) — Overall

| Metric | Value |
|--------|:-----:|
| **Overall Accuracy** | **89.33%** |
| **Macro-Avg Sensitivity** | **89.53%** |
| **Macro-Avg Specificity** | **94.70%** |
| **Macro-Avg F1 Score** | **89.28%** |
| **Micro-Avg AUC-ROC** | **0.9702** |

### Per-Class Breakdown

| Class | Sensitivity | Specificity | F1 Score | AUC-ROC |
|-------|:----------:|:----------:|:--------:|:-------:|
| **Normal** | 90.25% | 93.92% | 87.94% | 0.9680 |
| **Pneumonia** | 85.14% | 95.69% | 88.54% | 0.9554 |
| **Pleural Effusion** | 93.21% | 94.49% | 91.35% | 0.9763 |

### Inference Performance
| Metric | Target | Status |
|--------|--------|--------|
| End-to-end inference (CPU) | < 60 seconds | ✅ Achieved |
| Image validation | < 1 second | ✅ Achieved |
| Report generation (GPT-4o-mini) | < 10 seconds | ✅ Achieved |

---

## 🏗 Deployment Architecture

CDSS follows a **decoupled edge-cloud architecture** optimized for clinical environments:

```
┌──────────────────────────────────────────────────────────────────┐
│                         BROWSER (Client)                         │
│                                                                  │
│   React + Vite SPA  ←──  Deployed on Vercel / Static Hosting     │
│   - Upload X-rays                                                │
│   - Display results, heatmaps, reports                           │
│   - Chat with AI assistant                                       │
│   - Env var: VITE_API_BASE_URL → points to backend               │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTPS (REST API)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     BACKEND (Docker Container)                    │
│                                                                  │
│   FastAPI + Uvicorn  ←──  Deployed on Azure / Docker Host         │
│   - PyTorch inference (CPU-optimized)                             │
│   - GradCAM++ heatmap generation                                  │
│   - Image validation (6-gate pipeline)                            │
│   - CORS restricted to allowed origins                            │
│                                                                  │
│   Env vars: FOUNDRY_API_KEY, FOUNDRY_ENDPOINT, ALLOWED_ORIGINS    │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTPS
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    AZURE AI FOUNDRY (Cloud)                       │
│                                                                  │
│   GPT-4o-mini  ←──  Managed LLM Endpoint                         │
│   - Dual-tier report generation (Radiologist + Patient)           │
│   - Context-aware medical chatbot                                 │
│   - Safety guardrails (emergency detection, prompt injection)     │
└──────────────────────────────────────────────────────────────────┘
```

### Deployment Options

| Component | Local Dev | Production |
|-----------|-----------|------------|
| **Frontend** | `npm run dev` (Vite, port 5173) | Vercel / Netlify / Azure Static Web Apps |
| **Backend** | `uvicorn api:app` (port 8000) | Docker → Azure Container Apps / AWS ECS |
| **AI/LLM** | Azure AI Foundry (same) | Azure AI Foundry (same) |

---

## 🚀 Local Run Instructions

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** and **npm**
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/rakesh-vajrapu/CXR-MetaNet.git
cd CXR-MetaNet
```

### 1b. Extract Dataset Images
The dataset images are stored as a compressed archive via Git LFS. Run:
```bash
python setup_dataset.py
```
> This extracts 20,805 chest X-ray images (~2.6 GB) from `Dataset/images.zip` into `Dataset/images/`.

### 2. Backend Setup
```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root (or edit the existing one):
```env
FOUNDRY_API_KEY=your_azure_api_key
FOUNDRY_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:4173
```

### 4. Start the Backend
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
> ⏳ First startup takes ~60 seconds to load all 3 models (~2.8 GB total) into memory.

### 5. Frontend Setup
```bash
cd frontend
npm install
```

Create a `.env` file inside the `frontend/` folder:
```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### 6. Start the Frontend
```bash
npm run dev
```

### 7. Open in Browser
Navigate to **http://localhost:5173** — the CDSS UI will appear. Wait for the model loading banner to disappear before uploading X-rays.

### Continuous Deployment (GitHub Actions)
The backend is fully automated via CI/CD. When you push to the `main` branch, a GitHub Action automatically triggers:
1. Provisions an Ubuntu runner
2. Downloads the 847MB `images.zip` dataset directly from **Azure Blob Storage** (bypassing Git LFS limitations)
3. Extracts the images into the repository space
4. Builds the Docker container natively
5. Pushes to **Azure Container Registry (ACR)**
6. Restarts the Azure App Service with the new image

*Required GitHub repository secrets for this pipeline:*
- `REGISTRY_USERNAME` (ACR login)
- `REGISTRY_PASSWORD` (ACR password)
- `AZURE_WEBAPP_PUBLISH_PROFILE` (XML profile from Azure App Service)

> **Note on "Always On"**: For zero-wait inference, ensure "Always On" is enabled under *Configuration > General settings* in your Azure App Service. Otherwise, the container scales down after 20 minutes of inactivity, causing a 30-second "Cold Start" model loading delay for the next user.

---

## 🛠️ Performance & Scalability Features

### 1. Smart CPU Thread Allocation
To maximize inference speed without crashing the host OS or freezing the browser via CSS animation starvation, the backend dynamically sizes PyTorch threads based on its environment:
- **Production (Azure App Service)**: Detects the `WEBSITE_SITE_NAME` environment variable and allocates **90% of available CPU cores** (e.g., 3 threads on a 4-core machine) to maximize inference throughput.
- **Local Dev (VM / Laptop)**: Allocates **75% of available CPU cores** to leave generous headroom for the developer's Web Browser, IDE, and OS.

### 2. Strict AI Medical Scope Restriction
The CDSS Assistant relies on Azure OpenAI's GPT-4o-mini but features a hardcoded **Scope Restriction Guardrail**:
- The AI is instructed to *strictly* refuse answering questions unrelated to chest X-rays, clinical diagnoses, respiratory conditions, or the CDSS application itself.
- Off-topic queries (e.g., about general tech companies, politics, or coding) trigger a polite, guiding refusal, ensuring the assistant remains a dedicated clinical tool.

---

## ⚠️ Medical Disclaimer

> This system is a **research prototype** and is **NOT** a certified medical device. It is intended to **assist** — not replace — qualified healthcare professionals. All AI-generated diagnoses, reports, and chatbot responses must be reviewed, validated, and confirmed by a licensed radiologist or physician before any clinical decision is made. **Do not use this system for self-diagnosis or treatment.**

---

<p align="center">
  Built with ❤️ by the CDSS Research Team
</p>
