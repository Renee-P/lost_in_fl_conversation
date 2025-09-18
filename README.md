# Lost in Filipino Conversation

A repository for the code written for our thesis entitled **Evaluating Large Language Models in Filipino Multi-turn, Underspecified Conversations**. 
This repo implements methods inspired by *Laban et al. (2025)'s Lost in Conversation* study and uses the *Batayan dataset (Montalan et al., 2025)*.

## Status
Work in progress 🚧

## Setting Up
This project uses [Ollama](https://ollama.com/) to run local LLMs (7-9B parameters). You must have Ollama installed, running, and the necessary models pulled before using the code.

### 1. Install Ollama
Download and install Ollama from: [https://ollama.com/download](https://ollama.com/download). Ollama runs as a background service after installation.
(Optional) Run the following to confirm Ollama is active and responding:
```bash
curl http://localhost:11434/api/tags
```
You should see a JSON response listing installed models.

### 2. Install the Python client
In your Python environment, install the Ollama Python package:
```bash
pip install ollama
```

### 3. Pull the necessary models 
Make sure the models are downloaded locally: *(placeholder - will change for the complete model list later)*
```bash
ollama pull sailor2:1b
```
You can list available models with:
```bash
ollama list
```

## References
1. Laban, P., Hayashi, H., Zhou, Y., & Neville, J. (2025). *LLMs get lost in multi-turn conversation*. arXiv. https://arxiv.org/abs/2505.06120
   - code: https://github.com/microsoft/lost_in_conversation/tree/main
3. Susanto, Y., Hulagadri, A. V., Montalan, J. R., Ngui, J. G., Yong, X. B., Leong, W., Rengarajan, H., Limkonchotiwat, P., Mai, Y., & Tjhi, W. C. (2025). *SEA-HELM: Southeast Asian holistic evaluation of language models*. arXiv. https://arxiv.org/abs/2502.14301
   - code: https://github.com/aisingapore/SEA-HELM/tree/main
  

