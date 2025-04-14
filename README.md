# Scott's AI Model Collection

This repository contains a collection of hosted AI models I find differentuated and useful for building Agentic AI applications. Each model is set up to utilize the full allowed context length without any rate limiting constraints.

Unlike ALL commercial API providers that impose various limits even after payment, these models are configured without:

- RPM (Requests per minute) limits
- RPD (Requests per day) limits
- TPM (Tokens per minute) limits
- TPD (Tokens per day) limits

This makes them ideal for development, testing, and production use cases where predictable performance is essential.

## Deployment

The models are deployed on a containerized serverless capability. To save on costs, each model will scale to zero when not in use.  If a model has scaled down, it will take a few minutes for the model and associated container to cold start on fresh use.  If you want to warm up the model before use, you can send a POST to the /docs endpoint rather than the OpenAI API /vi endpoint.  Containers currently run for 10 minutes without an active call before scaling down.  In my experience, as you are interating on code, you usually execute, think for 3 to 6 minutes, modify code, and execute again so 10 minutes is more than enough time to keep the container warm and keep you out of cold start hell.

## Testing

Each model has been painstakenly tested and deployed for the best price and performance GPU configuration.  You will notice by looking at the various hugging face repo's that some of these models might memorywise fit on a single GPU, but are deployed with multiple GPUs.  This is because I wasn't satisfied with the single GPU inference response times.  In addition, there are no advanced Nvidia features in use here, I found after wasting hundreds of hours, the most performant delivery of a model is always exclusive GPU's attached per container.

## Performance

These models are hosted on both AWS and GCP and will scale to thousands of GPU's and containers if you apply pressure to the endpoint.  Of course I ask you to be responsible as we all know GPU's are not free and the GPU price per ounce costs are more expensive than gold.  I do have budgets in place if a mistake is made, and you scale up 10k containers so no need to worry about that on your end.

## Security

It is stupid simple to inject callbacks into model hosting to capture/store prompts and responses sent and received from models.  You have my word that there is none of that here.  Your request goes directly to the model and is returned back to you.  There are no AI gateways in play, no model API wrappers or OpenAI class modifications.  Just pure inference.  Further, when the container scales to zero, the model and its KV cache is completely destroyed and unrecoverable.

## A note from the developer

Developing AI applications can be frusterating.  I found developing agentic AI apps over the past year using local models via Ollama or LMStudio often relegated me to make concessions on quantization levels or dealing with the limitations of not having the ability to scale.  In addition, I found dealing with the API providers was rate limited, expensive, and very frusterating when I didn't have complete control over the model, the engine it was running, and its configuration.  Plus I find very few of the API providers other than the frontier models (GPT, Gemini, Claude, Cohere, etc) deliver models that will handle tool calling.  One of my favorite API hosting organizations, [SambaNova](https://sambanova.ai/), doesn't host a single model that supports tool calling.  My other favorite provider [Groq](https://groq.com/), rate limits to the moon and some of the model context configuration is lack luster.  But they do both still deliver value when inference speed is paramount as they have processing hardware that isn't available in the market.

## Model Specifications

The table below provides detailed specifications for each model in this collection:

| Model Name | Inference Engine | Tool Calling | Reasoning | FlashInfer | Quantization | Context Window | GPU Type | GPU Count |
|------------|-------------|--------------|-----------|------------|--------------|----------------|----------|---------|
| [Granite-3.2-8b-instruct](#granite-32-8b-instruct) | VLLM 0.8.3 v1 | Yes | No | 0.2.5 | None | 131K | L40S | 1 |
| [DeepHermes-3-Mistral-24B-Preview](#deephermes-3-mistral-24b-preview) | VLLM 0.8.2 v0 | No | Yes | 0.2.0.post2 | None | 32K | A100-80GB | 1 |
| [Qwen2.5-Coder-32B-Instruct](#qwen25-coder-32b-instruct) | VLLM 0.8.3 v1 | Yes | No | 0.2.0.post2 | GPTQ-Int4 (gptq_marlin) | 32K | A100-40GB | 1 |
| [QwQ-32B-AWQ](#qwq-32b-awq) | VLLM 0.8.2 v0 | Yes | Yes | 0.2.0.post2 | AWQ | 32K | L40S | 1 |
| [DeepSeek-R1-Distill-Llama-8B](#deepseek-r1-distill-llama-8b) | VLLM 0.8.3 v0 | No | Yes | Yes | None | 32K | L4 | 2 |
| [BGE-large-en-v1.5](#bge-large-en-v15) | VLLM 0.8.3 v0 | No | No | No | None | N/A | T4 | 1 |
| [Jina Reranker V2 Base Multilingual](#jina-reranker-v2-base-multilingual) | VLLM 0.8.3 v0 | No | No | No | None | N/A | L4 | 1 |

## Usage

Once deployed, you can interact with the models using the OpenAI Python client library or any HTTP client that supports the OpenAI API format. Authentication is handled via API keys.

Example:
```python
from openai import OpenAI

client = OpenAI(
    api_key="your_api_key",
    base_url="https://your-deployment-url"
)

response = client.chat.completions.create(
    model="granite-3.2-8b-instruct",
    messages=[
        {"role": "user", "content": "Hello, how can you help me today?"}
    ]
)

print(response.choices[0].message.content)
```

## Roadmap

I plan to add more models to this collection over time.  If you have a model you'd like me to add, please let me know.

1. ✅ Deepseek R1 Distill model - Added!
2. ✅ BGE-large-en-v1.5 - Added!
3. Cogoto hybrid reasoning model


## Model Notes

### Granite-32-8b-instruct

**API Endpoint**: `https://smpnet74-1--granite-3-2-8b-instruct-serve.modal.run/v1`

**Model Card**: [IBM/granite-3.2-8b-instruct](https://huggingface.co/IBM/granite-3.2-8b-instruct)

The Granite-3.2-8b-instruct model is IBM's 8B parameter instruction-tuned model that excels at following instructions and tool calling.

I use this model almost exclusively for tool calling because its one of the few tool calling models that works flawlessly at 8b parameters.  The llama and qwen model family has struggled for me with tool calling in the smaller versions.

### DeepHermes-3-Mistral-24B-Preview

**API Endpoint**: `https://smpnet74-1--deephermes-3-mistral-24b-preview-serve.modal.run/v1`

**Model Card**: [NousResearch/DeepHermes-3-Mistral-24B-Preview](https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview)

The DeepHermes-3-Mistral-24B-Preview model is NousResearch's 24B parameter model based on Mistral architecture. Key features include:

- High-quality general purpose model with strong reasoning capabilities

What is compelling to me is this model is trained as the first LLM model to unify both "intuitive", traditional mode responses and long chain of thought reasoning responses into a single model, toggled by a system prompt.

Add this exact content to the start of your system prompt and the model will act as a thinking model:

You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.

Do not include that section in your system prompt and it will act as a traditional model.

### Qwen2.5-Coder-32B-Instruct

**API Endpoint**: `https://smpnet74-1--qwen2-5-coder-32b-instruct-gptq-int4-serve.modal.run/v1`

**Model Card**: [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)

The Qwen2.5-Coder-32B-Instruct model is Alibaba's code-specialized large language model with 32B parameters. This version uses GPTQ 4-bit quantization to reduce memory requirements while maintaining high performance.

This model I have found works fantastic as a coding replacement model for Sonnet 3.5-7.  If you are into vibe coding with technologies such as Aider, Roo Code, or Cline, you'll find its near impossible to find a model that works as well as Sonnet 3.5 or 3.7.  I have tested at least 20 different model architectures and parameter sizes, and Qwen25-Coder-32B-Instruct always comes out on top.

I also have found that the int4 awq quantized version works just as well as full precision.  So in order to save on model hosting costs and fit it onto a 40gb card, I run it with awq quantization.  No need to spend extra pennies when not required.  I did run this for a moment on an L40S rather than the A100-40GB, but the performance was subpar so I reverted back to A100-40GB.  Quite honestly when used for vibe coding, it probably could use a bit more oomph than the A100 but I am cheap and willing to wait a little longer to save a buck.

### QwQ-32B-AWQ

**API Endpoint**: `https://smpnet74-1--qwq-32b-awq-serve.modal.run/v1`

**Model Card**: [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B)

QwQ-32B-AWQ is Qwen's specialized reasoning model with 32B parameters, quantized using AWQ to 4-bit precision. This model is particularly strong at mathematical reasoning, scientific analysis, and complex problem-solving tasks.

What makes this model special is its dual capabilities - it combines both tool calling and reasoning in a single model. The reasoning is implemented using the DeepSeek-R1 reasoning parser, which allows the model to show its step-by-step thinking process before providing a final answer.

### DeepSeek-R1-Distill-Llama-8B

**API Endpoint**: `https://smpnet74-1--deepseek-r1-distill-llama-8b-serve.modal.run/v1`

**Model Card**: [unsloth/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B)

The DeepSeek-R1-Distill-Llama-8B model is a distilled version of DeepSeek's R1 reasoning model, based on the Llama architecture with 8B parameters. This model is specifically designed to provide strong reasoning capabilities in a more efficient package.

Key features include:
- Enabled reasoning/thinking capabilities with the DeepSeek R1 reasoning parser
- 32K token context window
- Optimized for showing step-by-step thinking when prompted with reasoning questions

This model is particularly useful for applications requiring detailed reasoning processes while maintaining reasonable inference speeds and resource requirements. When prompted with "Please show your thinking" or similar instructions, the model will demonstrate its reasoning process before providing a final answer.

### BGE-large-en-v1.5

**API Endpoint**: `https://smpnet74-1--bge-large-en-v1-5-serve.modal.run/v1`

**Model Card**: [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)

The BGE-large-en-v1.5 model is a state-of-the-art embedding model developed by BAAI (Beijing Academy of Artificial Intelligence). This model is specifically designed to convert text into high-quality vector embeddings for various natural language processing tasks.

Key features include:
- 1024-dimensional dense vector embeddings
- Optimized for semantic search and text similarity tasks
- Supports English language content
- Deployed with vLLM's embedding capabilities for efficient inference

This model is particularly useful for:
- Building semantic search systems
- Document retrieval applications
- Text clustering and classification
- RAG (Retrieval-Augmented Generation) systems

Example usage involves sending text to the `/v1/embeddings` endpoint with a JSON payload containing an "input" field with your text, which returns vector embeddings that can be used for semantic search and similarity comparisons.

### Jina Reranker V2 Base Multilingual

**API Endpoint**: `https://smpnet74-1--jina-reranker-v2-base-multilingual-serve-dev.modal.run/v1`

**Model Card**: [jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)

The Jina Reranker V2 Base Multilingual model is a powerful cross-encoder designed for reranking search results across multiple languages. This model evaluates the relevance between a query and document pairs, providing scores that can be used to reorder search results for improved precision.

Key features include:
- Multilingual support for over 100 languages
- Optimized for document reranking tasks
- Based on a cross-encoder architecture for direct relevance scoring
- Deployed with vLLM's scoring capabilities for efficient inference

This model is particularly useful for:
- Improving search result quality in multilingual applications
- Reranking candidate documents in RAG (Retrieval-Augmented Generation) systems
- Enhancing document retrieval precision in information retrieval systems
- Fine-tuning search results for domain-specific applications

Example usage involves sending a query and a list of documents to the `/v1/rerank` endpoint, which returns the documents reordered by relevance score.
