# Colorectal Cancer Prediction Pipeline with MLflow, DagsHub, and Kubeflow


End-to-end MLOps pipeline for colorectal cancer prediction using Kubeflow Pipelines on Minikube with MLflow tracking through DagsHub.

##  Technologies Used
- **Orchestration**: Kubeflow Pipelines
- **Experiment Tracking**: MLflow + DagsHub
- **Local Kubernetes**: Minikube
- **Containerization**: Docker
- **Model Serving**: KServe/KFServing

##  Prerequisites
1. [Minikube](https://minikube.sigs.k8s.io/docs/start/) v1.25+
2. [kubectl](https://kubernetes.io/docs/tasks/tools/)
3. [Docker](https://docs.docker.com/get-docker/) 20.10+
4. Python 3.8+
5. [Kubeflow](https://www.kubeflow.org/docs/started/installing-minikube/) 1.7+

### Key Features:
1. **Complete Setup Guide** - From Minikube to Kubeflow installation
2. **DagsHub Integration** - Ready-to-use MLflow configuration
3. **Production-Ready Structure** - Standard MLOps directory layout
4. **Visual Workflow** - Mermaid diagram of pipeline architecture
5. **Copy-Paste Friendly** - Single code block for GitHub
6. **Placeholder Management** - Clear indications for user-specific values

### Before Using:
1. Replace all `<PLACEHOLDER>` values with your actual credentials
2. Add a real architecture diagram (remove placeholder image)
3. Customize the training parameters in the experiment example
4. Update maintainer information at the bottom
