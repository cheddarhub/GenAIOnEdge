# Generative AI on the Edge: PromptAI and Performance Evaluation

This project is focused on **Generative AI experimentation on Edge devices**, specifically utilizing a **Raspberry Pi** cluster as the testbed. The goal of this project is to deploy a distributed **PromptAI** system using **Docker containers** and a **Kubernetes (K3s) cluster** on Raspberry Pi devices.

The **PromptAI** system consists of interconnected services, including a prompt front-end, proxy, and model containers. These services are developed and deployed using modern technologies such as **FastAPI** (for the front-end), **Node.js** (for the API layer), and containerized using **Docker** for easy deployment and management. Docker images are also available on DockerHub https://hub.docker.com/u/myicap.

Users can choose from a pool of **Large Language Models (LLMs)** and interact with their selected model. The requests are forwarded through a **proxy service**, which directs the traffic to the appropriate model container running on the Raspberry Pi nodes.

In addition to deployment and interaction, the project also focuses on evaluating the performance of these models using the **Llama.cpp** framework.

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

Clone the repository and install dependencies.

### Clone the repository:
```bash
git clone https://github.com/cheddarhub/GenAIOnEdge.git

```

## Acknowledgments
This work is supported by the Communications Hub for Empowering Distributed Cloud Computing Applications and Research (CHEDDAR) (https://cheddarhub.org/), a hub dedicated to advancing future communications. CHEDDAR is funded by the Engineering and Physical Sciences Research Council (EPSRC) – UK Research and Innovation (UKRI) via the Technology Missions Fund (TMF).



