# LangChain & LangGraph Interview Questions for AI/ML Roles
---

## Table of Contents

- [Part 1: LangChain Interview Questions (170 Questions)](#part-1-langchain-interview-questions)
  - [LangChain Basics](#langchain-basics)
  - [Chains & LCEL](#chains--lcel)
  - [Agents and Tools](#agents-and-tools)
  - [Memory](#memory)
  - [Retrieval and Vector Stores](#retrieval-and-vector-stores)
  - [Evaluation and Metrics](#evaluation-and-metrics)
  - [Deployment and Inference](#deployment-and-inference)
  - [Debugging and Error Handling](#debugging-and-error-handling)
  - [Visualization and Interpretation](#visualization-and-interpretation)
  - [Best Practices and Optimization](#best-practices-and-optimization)
- [Part 2: LangGraph Interview Questions (100 Questions)](#part-2-langgraph-interview-questions)
  - [LangGraph Basics](#langgraph-basics)
  - [State Management](#state-management)
  - [Nodes and Edges](#nodes-and-edges)
  - [Agents and Tool Calling](#langgraph-agents-and-tool-calling)
  - [Memory and Persistence](#langgraph-memory-and-persistence)
  - [Human-in-the-Loop](#human-in-the-loop)
  - [Multi-Agent Systems](#multi-agent-systems)
  - [Streaming and Async](#streaming-and-async)
  - [Deployment and Production](#langgraph-deployment-and-production)
  - [Debugging and Observability](#langgraph-debugging-and-observability)

---

# Part 1: LangChain Interview Questions

> 170 questions covering core LangChain concepts for NLP and Generative AI roles.

---

## LangChain Basics

### Basic

**1. What is LangChain, and how is it used in NLP applications?**  
LangChain is a framework for building applications powered by LLMs. It provides composable building blocks—models, prompts, chains, memory, retrievers, and agents—for context-aware NLP tasks like chatbots, summarizers, and RAG pipelines.

> **Interview tip:** Emphasize that LangChain 1.0 (released late 2025) consolidated its architecture around LCEL and LangGraph, and that the framework is model-agnostic (OpenAI, Anthropic, Ollama, HuggingFace, etc.).

```python
# Modern: use langchain_openai, not langchain.llms
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke([HumanMessage(content="What is NLP?")])
print(response.content)
```

---

**2. How do you install LangChain and its dependencies?**  
LangChain is now split into separate packages. You install the core library and provider-specific packages separately.

> **Interview tip:** Interviewers often ask about the package split. `langchain` is the orchestration layer; `langchain-core` holds base abstractions; `langchain-openai`, `langchain-anthropic`, `langchain-community` are provider integrations.

```bash
pip install langchain langchain-openai langchain-community
# For vector stores, embeddings, etc.:
pip install faiss-cpu chromadb pinecone-client
```

---

**3. What are the core components of LangChain?**  
The main components are: **Chat Models / LLMs** (text generation), **Prompt Templates** (structured inputs), **Chains / LCEL Runnables** (composable workflows), **Memory** (conversation state), **Retrievers** (document search), **Agents** (dynamic tool-using reasoning), and **Tools** (external capabilities).

> **Interview tip:** In modern LangChain, "Chains" are built via LCEL using the `|` pipe operator. `LLMChain` is deprecated.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LCEL chain: prompt | model | parser
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("Summarize: {text}")
chain = prompt | llm | StrOutputParser()
```

---

**4. How do you configure an LLM in LangChain for text generation?**  
Set model name, temperature, and API keys (via environment variables or constructor arguments).

> **Interview tip:** Prefer `ChatOpenAI` over `OpenAI` for chat-optimized models. Use environment variables (`OPENAI_API_KEY`) rather than hardcoded keys.

```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "your-api-key"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
response = llm.invoke("Explain transformers in one sentence.")
print(response.content)
```

---

**5. What is a PromptTemplate in LangChain, and how is it used?**  
`PromptTemplate` structures prompts with dynamic variables. `ChatPromptTemplate` is used with chat models and supports system/human/AI message roles.

> **Interview tip:** Know the difference: `PromptTemplate` produces a plain string; `ChatPromptTemplate` produces a list of messages. Chat models require messages format.

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# For chat models (preferred)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Explain {topic} in simple terms.")
])

# For plain-text models
text_prompt = PromptTemplate.from_template("Explain {topic} in simple terms.")
```

---

**6. How do you save and load a LangChain chain for reuse?**  
Modern LCEL chains don't use `.save("chain.json")` (that was the old `LLMChain` API). Instead, save configuration as YAML/JSON and reconstruct, or use LangSmith for hosted versioning.

> **Interview tip:** The old `chain.save()` / `load_chain()` approach is deprecated. Modern best practice is to version your chain configuration code directly (treat chains as code).

```python
import json
from langchain_core.prompts import ChatPromptTemplate

# Save prompt config
prompt_config = {
    "template": "Summarize: {text}",
    "model": "gpt-4o-mini",
    "temperature": 0.0
}
with open("chain_config.json", "w") as f:
    json.dump(prompt_config, f)

# Reload and reconstruct
with open("chain_config.json") as f:
    cfg = json.load(f)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(cfg["template"])
llm = ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"])
chain = prompt | llm | StrOutputParser()
```

---

### Intermediate

**7. Write a function to create a simple LangChain chain for text summarization.**  
Uses LCEL pipe syntax to build a summarization chain.

> **Interview tip:** `LLMChain` is deprecated since LangChain 0.1.17. The LCEL equivalent is `prompt | llm | parser`. The chain is a `RunnableSequence` and supports `.invoke()`, `.batch()`, `.stream()`, and `.ainvoke()` natively.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_summary_chain():
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in 2-3 sentences:\n\n{text}"
    )
    return prompt | llm | StrOutputParser()

chain = create_summary_chain()
result = chain.invoke({"text": "LangChain is a framework for building LLM apps..."})
```

---

**8. How do you use LangChain to integrate with external APIs like OpenAI?**  
Use `langchain_openai` package and configure via environment variables.

```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "your-api-key"
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("What are the benefits of RAG?")
print(response.content)
```

---

**9. Explain the role of callbacks in LangChain for monitoring.**  
Callbacks hook into chain lifecycle events (`on_chain_start`, `on_llm_end`, `on_tool_error`, etc.) for logging, tracing, and monitoring.

```python
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[StdOutCallbackHandler()])
prompt = ChatPromptTemplate.from_template("Define: {word}")
chain = prompt | llm | StrOutputParser()
chain.invoke({"word": "embedding"})
```

---

**10. How do you handle API rate limits in LangChain?**  
Use `max_retries` on the model and optionally wrap with `tenacity` or use LangSmith's built-in retry handling.

```python
from langchain_openai import ChatOpenAI

# Built-in retries
llm = ChatOpenAI(model="gpt-4o-mini", max_retries=3)

# Or use a fallback chain with .with_fallbacks()
from langchain_openai import ChatOpenAI
fast_llm = ChatOpenAI(model="gpt-4o-mini")
fallback_llm = ChatOpenAI(model="gpt-3.5-turbo")
robust_llm = fast_llm.with_fallbacks([fallback_llm])
```

---

**11. Write a function to generate text completions with LangChain.**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def generate_completion(prompt_text: str, model: str = "gpt-4o-mini") -> str:
    llm = ChatOpenAI(model=model)
    response = llm.invoke([HumanMessage(content=prompt_text)])
    return response.content

result = generate_completion("Explain embeddings in one paragraph.")
```

---

**12. How do you use LangChain to switch between different LLM providers?**  
LangChain's unified interface lets you swap providers by changing the import and model name.

> **Interview tip:** This is a key selling point of LangChain. All ChatModel providers implement the same `BaseChatModel` interface, so your chain code doesn't change.

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# HuggingFace (local/hosted)
from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="google/flan-t5-base")

# Ollama (local)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2")

# The chain code below works with ALL of the above
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
chain = ChatPromptTemplate.from_template("Define {word}") | llm | StrOutputParser()
```

---

### Advanced

**13. Implement a custom LangChain LLM wrapper for a local model.**  
Subclass `BaseChatModel` (for chat models) or `BaseLLM` (for completion models).

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any

class CustomLocalLLM(BaseChatModel):
    model_name: str = "custom-v1"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs,
    ) -> ChatResult:
        # Replace with your actual model call
        last_msg = messages[-1].content
        output_text = f"[Custom model response to: {last_msg}]"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=output_text))])

    @property
    def _llm_type(self) -> str:
        return "custom-local"

llm = CustomLocalLLM()
response = llm.invoke("Hello!")
print(response.content)
```

---

**14. Write a function to optimize LangChain LLM calls for cost efficiency.**  
Use `.batch()` for parallel requests and caching to avoid redundant API calls.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List

def batch_llm_calls(prompts: List[str], model: str = "gpt-4o-mini") -> List[str]:
    llm = ChatOpenAI(model=model)
    chain = ChatPromptTemplate.from_template("{text}") | llm | StrOutputParser()
    # .batch() sends all requests concurrently
    return chain.batch([{"text": p} for p in prompts])

results = batch_llm_calls(["Define AI", "Define ML", "Define NLP"])
```

---

**15. How do you implement a custom prompt template with dynamic inputs?**

```python
from langchain_core.prompts import ChatPromptTemplate
from typing import List

def create_dynamic_prompt(input_keys: List[str]) -> ChatPromptTemplate:
    # Build a template string from dynamic keys
    content = " | ".join([f"{k}: {{{k}}}" for k in input_keys])
    return ChatPromptTemplate.from_template(f"Answer based on the following — {content}")

prompt = create_dynamic_prompt(["context", "question", "language"])
formatted = prompt.format_messages(context="Python docs", question="What is a list?", language="English")
```

---

**16. Write a function to validate LangChain LLM outputs.**

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def validated_chain(min_length: int = 20):
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template("Explain {topic}.")
    base_chain = prompt | llm | StrOutputParser()

    def validate_and_run(inputs: dict) -> str:
        output = base_chain.invoke(inputs)
        if len(output.strip()) < min_length:
            raise ValueError(f"Output too short: {len(output)} chars")
        return output

    return validate_and_run
```

---

**17. How do you use LangChain to handle multi-lingual NLP tasks?**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a multilingual assistant. Always respond in {language}."),
    ("human", "{text}")
])
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"text": "What is machine learning?", "language": "Spanish"})
```

---

**18. Implement a LangChain callback for logging execution time.**

```python
from langchain_core.callbacks.base import BaseCallbackHandler
from typing import Any, Dict
import time

class TimingCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        self._start_time = time.perf_counter()

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        elapsed = time.perf_counter() - self._start_time
        print(f"Chain completed in {elapsed:.3f}s")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        print(f"LLM error: {error}")

# Usage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

handler = TimingCallbackHandler()
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[handler])
chain = ChatPromptTemplate.from_template("Define {word}") | llm | StrOutputParser()
chain.invoke({"word": "embedding"})
```

---

## Chains & LCEL

### Basic

**19. What is a chain in LangChain, and how is it used in NLP?**  
A chain is a sequence of composable steps that process inputs and produce outputs. In modern LangChain, chains are built using **LCEL (LangChain Expression Language)** via the `|` (pipe) operator. Each step is a `Runnable`.

> **Interview tip:** The key insight is that LCEL chains are lazy—they describe the computation graph but don't execute until `.invoke()`, `.batch()`, or `.stream()` is called. This enables streaming and async out of the box.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
# Pipe operator builds a RunnableSequence
chain = ChatPromptTemplate.from_template("Define {word}") | llm | StrOutputParser()
result = chain.invoke({"word": "algorithm"})
```

---

**20. How do you create a sequential chain in LangChain?**  
Use LCEL to pipe the output of one step into the next. `SimpleSequentialChain` and `SequentialChain` are deprecated.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Step 1: Summarize
summarize_chain = ChatPromptTemplate.from_template("Summarize: {text}") | llm | parser

# Step 2: Explain the summary — pass output of step1 as input
explain_chain = ChatPromptTemplate.from_template("Explain this summary: {summary}") | llm | parser

# Compose using a lambda to rename the key
full_chain = summarize_chain | (lambda summary: {"summary": summary}) | explain_chain

result = full_chain.invoke({"text": "Transformers use attention mechanisms..."})
```

---

**21. What is the difference between LLMChain and LCEL RunnableSequence?**  
`LLMChain` (deprecated) was a class-based wrapper that required explicit `input_variables` and `output_key`. LCEL `RunnableSequence` uses the `|` operator, is composable, supports streaming/async by default, and handles input/output routing automatically.

> **Interview tip:** This is a very common question. Summarize it as: LCEL = declarative + lazy + streaming-first vs. LLMChain = imperative + eager + boilerplate-heavy.

```python
# OLD (deprecated) — Do NOT use
# from langchain.chains import LLMChain
# chain = LLMChain(llm=llm, prompt=prompt)

# NEW (correct)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

chain = ChatPromptTemplate.from_template("Define {word}") | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
result = chain.invoke({"word": "neural network"})
```

---

**22. How do you pass inputs to a LangChain chain?**

```python
# Single input
result = chain.invoke({"word": "algorithm"})

# Batch input
results = chain.batch([{"word": "AI"}, {"word": "ML"}, {"word": "NLP"}])

# Async
import asyncio
result = asyncio.run(chain.ainvoke({"word": "embedding"}))
```

---

**23. What is the role of output parsers in LangChain chains?**  
Output parsers convert raw LLM output (a `AIMessage` or string) into structured formats like strings, JSON, lists, or Pydantic models.

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Plain string
str_chain = ChatPromptTemplate.from_template("Name 5 {category}") | llm | StrOutputParser()

# CSV list
list_chain = ChatPromptTemplate.from_template("List 5 {category} as comma-separated values") | llm | CommaSeparatedListOutputParser()

# JSON
json_chain = (
    ChatPromptTemplate.from_template('Return a JSON object with "name" and "description" for {topic}')
    | llm
    | JsonOutputParser()
)
```

---

**24. How do you debug a LangChain chain?**

```python
# Option 1: verbose via callbacks
from langchain_core.callbacks import StdOutCallbackHandler
chain.invoke({"word": "AI"}, config={"callbacks": [StdOutCallbackHandler()]})

# Option 2: use LangSmith tracing (best for production)
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"

# Option 3: inspect intermediate steps with RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
debug_chain = RunnablePassthrough() | (lambda x: print("Input:", x) or x) | chain
```

---

### Intermediate

**25. Write a function to create a LangChain chain for question answering.**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_qa_chain():
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based only on the provided context. If unsure, say 'I don't know'."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    return prompt | llm | StrOutputParser()

qa_chain = create_qa_chain()
result = qa_chain.invoke({
    "context": "LangChain was created by Harrison Chase in 2022.",
    "question": "Who created LangChain?"
})
```

---

**26. How do you implement a chain with multiple prompts in LangChain?**  
Use `RunnablePassthrough` and dictionaries to pass multiple outputs through the pipeline.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

summarize = ChatPromptTemplate.from_template("Summarize: {text}") | llm | parser
key_points = ChatPromptTemplate.from_template("List key points from: {summary}") | llm | parser

# Multi-step chain: pass original text AND summary forward
multi_chain = (
    RunnablePassthrough.assign(summary=summarize)
    | RunnablePassthrough.assign(points=key_points)
)

result = multi_chain.invoke({"text": "Transformers are the backbone of modern NLP..."})
print(result["summary"])
print(result["points"])
```

---

**27. Write a function to chain text generation and parsing.**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class Answer(BaseModel):
    answer: str = Field(description="The direct answer")
    confidence: str = Field(description="high/medium/low confidence level")

def create_structured_chain():
    llm = ChatOpenAI(model="gpt-4o-mini")
    parser = JsonOutputParser(pydantic_object=Answer)
    prompt = ChatPromptTemplate.from_template(
        "Answer the question and provide a confidence level.\n"
        "Question: {question}\n"
        "{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

chain = create_structured_chain()
result = chain.invoke({"question": "What is the capital of France?"})
```

---

**28. How do you use LangChain to create a conversational chain?**  
Use `ChatPromptTemplate` with a `MessagesPlaceholder` for history, plus `RunnableWithMessageHistory`.

> **Interview tip:** `ConversationChain` with `ConversationBufferMemory` is deprecated. Modern approach is `RunnableWithMessageHistory`.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
chain = prompt | llm | StrOutputParser()

store = {}  # session_id -> ChatMessageHistory

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

response = conversation.invoke(
    {"input": "Hi, my name is Alex."},
    config={"configurable": {"session_id": "user_1"}}
)
```

---

**29. Implement a chain to handle batch processing in LangChain.**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List

def batch_chain_processing(inputs: List[dict]) -> List[str]:
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = ChatPromptTemplate.from_template("Define {word}") | llm | StrOutputParser()
    # .batch() runs requests concurrently with a thread pool
    return chain.batch(inputs, config={"max_concurrency": 5})

results = batch_chain_processing([{"word": "AI"}, {"word": "ML"}, {"word": "NLP"}])
```

---

**30. How do you handle errors in LangChain chains?**

```python
from langchain_core.runnables import RunnableLambda

def safe_run(inputs: dict) -> str:
    try:
        return chain.invoke(inputs)
    except Exception as e:
        return f"Error: {e}"

# Or use .with_fallbacks() for automatic fallback chains
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

primary = ChatPromptTemplate.from_template("Define {word}") | ChatOpenAI(model="gpt-4o") | StrOutputParser()
fallback = ChatPromptTemplate.from_template("Define {word}") | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
robust_chain = primary.with_fallbacks([fallback])
```

---

### Advanced

**31. Write a function to implement a custom LangChain chain (Runnable).**

```python
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_custom_runnable(preprocessing_fn, postprocessing_fn) -> RunnableSequence:
    """Wraps custom pre/post processing around an LLM call."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template("{input}")
    
    pre = RunnableLambda(preprocessing_fn)
    core = prompt | llm | StrOutputParser()
    post = RunnableLambda(postprocessing_fn)
    
    return pre | core | post

# Example usage
preprocess = lambda x: {"input": x["raw_text"].strip().upper()}
postprocess = lambda x: {"output": x, "word_count": len(x.split())}

custom_chain = build_custom_runnable(preprocess, postprocess)
result = custom_chain.invoke({"raw_text": "  explain neural networks  "})
```

---

**32. How do you optimize LangChain chains for low-latency NLP tasks?**

```python
import langchain
from langchain_community.cache import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Enable global LLM response caching
langchain.llm_cache = InMemoryCache()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # temperature=0 = deterministic = cacheable
chain = ChatPromptTemplate.from_template("Define {word}") | llm | StrOutputParser()

# First call hits API
result1 = chain.invoke({"word": "embedding"})
# Second identical call is served from cache instantly
result2 = chain.invoke({"word": "embedding"})
```

---

**33. Write a function to implement parallel chain execution in LangChain.**

```python
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_parallel_chain():
    llm = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()

    summary_chain = ChatPromptTemplate.from_template("Summarize: {text}") | llm | parser
    sentiment_chain = ChatPromptTemplate.from_template("Sentiment of: {text} (positive/negative/neutral)") | llm | parser
    keywords_chain = ChatPromptTemplate.from_template("Extract 5 keywords from: {text}") | llm | parser

    # RunnableParallel runs all branches concurrently
    return RunnableParallel(
        summary=summary_chain,
        sentiment=sentiment_chain,
        keywords=keywords_chain
    )

parallel_chain = create_parallel_chain()
result = parallel_chain.invoke({"text": "LangChain is an amazing framework for building LLM applications."})
print(result["summary"], result["sentiment"], result["keywords"])
```

---

**34. How do you implement a chain with dynamic routing in LangChain?**  
Use `RunnableBranch` or a conditional lambda for routing.

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

summarize_chain = ChatPromptTemplate.from_template("Summarize: {text}") | llm | parser
translate_chain = ChatPromptTemplate.from_template("Translate to French: {text}") | llm | parser
default_chain = ChatPromptTemplate.from_template("Process: {text}") | llm | parser

router = RunnableBranch(
    (lambda x: x.get("task") == "summarize", summarize_chain),
    (lambda x: x.get("task") == "translate", translate_chain),
    default_chain  # fallback
)

result = router.invoke({"text": "Hello world", "task": "translate"})
```

---

**35. Implement a chain to handle multi-step reasoning in LangChain.**

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Chain-of-Thought reasoning pipeline
cot_chain = (
    RunnablePassthrough.assign(
        step1=ChatPromptTemplate.from_template("Break down the problem step by step: {problem}") | llm | parser
    )
    | RunnablePassthrough.assign(
        step2=ChatPromptTemplate.from_template("Based on this analysis:\n{step1}\nNow solve: {problem}") | llm | parser
    )
    | RunnablePassthrough.assign(
        final=ChatPromptTemplate.from_template("Given the solution:\n{step2}\nProvide a concise final answer.") | llm | parser
    )
)

result = cot_chain.invoke({"problem": "If a train travels 60mph for 2.5 hours, how far does it go?"})
```

---

**36. How do you monitor chain performance in production?**

```python
import time
import logging
from langchain_core.callbacks.base import BaseCallbackHandler
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionMonitorCallback(BaseCallbackHandler):
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        self._t0 = time.perf_counter()

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        logger.info(f"Chain latency: {time.perf_counter() - self._t0:.3f}s")

    def on_chain_error(self, error: Exception, **kwargs):
        logger.error(f"Chain error: {error}", exc_info=True)

    def on_llm_end(self, response: Any, **kwargs):
        # Log token usage
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            logger.info(f"Tokens used: {usage}")
```

---

## Agents and Tools

### Basic

**37. What is a LangChain agent, and how is it used in NLP?**  
An agent uses an LLM as a reasoning engine to decide which tools to call and in what order. Unlike a fixed chain, agents handle dynamic, multi-step tasks where the sequence isn't predetermined.

> **Interview tip:** In modern LangChain, use `create_react_agent` + `AgentExecutor` or, even better, build agents in LangGraph for production use. `initialize_agent` is deprecated.

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

@tool
def search(query: str) -> str:
    """Search the web for information about the query."""
    return f"Search results for: {query}"

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search]
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "What is LangChain?"})
```

---

**38. How do you create a tool in LangChain for an agent?**  
Use the `@tool` decorator for simple functions, or subclass `BaseTool` for more control.

```python
from langchain_core.tools import tool, BaseTool
from typing import Type
from pydantic import BaseModel, Field

# Simple approach: @tool decorator
@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression. Input should be a valid Python math expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

# Advanced approach: subclass BaseTool
class SearchInput(BaseModel):
    query: str = Field(description="The search query")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for current information"
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        return f"Results for '{query}': [mock result]"
```

---

**39. What is the ReAct framework in LangChain agents?**  
ReAct (Reasoning + Acting) is an agent strategy where the LLM interleaves **Thought** (reasoning), **Action** (tool call), and **Observation** (tool result) steps until it reaches a final answer. This makes reasoning transparent and debuggable.

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 25°C"

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = hub.pull("hwchase17/react")  # Standard ReAct prompt
agent = create_react_agent(llm, [get_weather], prompt)
executor = AgentExecutor(agent=agent, tools=[get_weather], verbose=True)
# The agent will Thought→Action→Observation loop
```

---

**40. How do you use LangChain agents for web search integration?**

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

search_tool = DuckDuckGoSearchRun()
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, [search_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[search_tool])
result = executor.invoke({"input": "What are the latest developments in AI?"})
```

---

**41. What is the role of the AgentExecutor in LangChain?**  
`AgentExecutor` is the runtime loop that calls the agent to get the next action, executes the tool, feeds back the observation, and repeats until the agent produces a final answer or hits `max_iterations`.

```python
from langchain.agents import AgentExecutor

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,              # Prevent infinite loops
    handle_parsing_errors=True,   # Gracefully handle malformed tool calls
    return_intermediate_steps=True  # Include thought/action/obs trace
)
result = executor.invoke({"input": "Solve this step by step: what is 15% of 240?"})
```

---

**42. How do you handle tool failures in LangChain agents?**

```python
from langchain.agents import AgentExecutor

# handle_parsing_errors catches LLM output that doesn't match tool schema
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    max_execution_time=30  # timeout in seconds
)

# Make tools fault-tolerant
from langchain_core.tools import tool

@tool
def safe_search(query: str) -> str:
    """Search for information. Returns error message if search fails."""
    try:
        # actual search logic
        return f"Results for: {query}"
    except Exception as e:
        return f"Search failed: {str(e)}. Try rephrasing your query."
```

---

### Intermediate

**43. Write a function to create a LangChain agent with custom tools.**

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

def create_custom_agent():
    @tool
    def summarize_text(text: str) -> str:
        """Summarizes a given text to its key points."""
        return text[:100] + "... [summarized]"

    @tool
    def count_words(text: str) -> str:
        """Counts the number of words in a text."""
        return str(len(text.split()))

    llm = ChatOpenAI(model="gpt-4o-mini")
    tools = [summarize_text, count_words]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
```

---

**44. How do you implement a LangChain agent with memory?**  
Use `RunnableWithMessageHistory` or pass `chat_history` to the agent prompt.

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

@tool
def get_time() -> str:
    """Returns the current time."""
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_time]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

store = {}
agent_with_memory = RunnableWithMessageHistory(
    executor,
    lambda sid: store.setdefault(sid, ChatMessageHistory()),
    input_messages_key="input",
    history_messages_key="chat_history"
)
```

---

**45. Write a function to integrate a LangChain agent with a database tool.**

```python
from langchain_core.tools import tool
import sqlite3

def create_db_tool(db_path: str):
    @tool
    def query_database(sql_query: str) -> str:
        """Execute a read-only SQL SELECT query on the database and return results.
        Only SELECT queries are allowed for safety."""
        if not sql_query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are permitted."
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            conn.close()
            return str(rows[:10])  # limit to 10 rows
        except Exception as e:
            return f"Database error: {e}"
    return query_database
```

---

**46. How do you use LangChain agents for multi-tool workflows?**

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def search(query: str) -> str:
    """Search for information on a topic."""
    return f"Found: key facts about '{query}'"

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))

@tool
def summarizer(text: str) -> str:
    """Summarize a long piece of text."""
    return text[:200] + "..."

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search, calculator, summarizer]
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the available tools to answer questions."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

---

**47. Implement a LangChain agent to handle API-based tools.**

```python
from langchain_core.tools import tool
import httpx

def create_api_tool(base_url: str, api_key: str):
    @tool
    def call_api(endpoint: str, params: str = "") -> str:
        """Call an external API endpoint. Provide endpoint path and optional query params."""
        try:
            url = f"{base_url}/{endpoint}"
            headers = {"Authorization": f"Bearer {api_key}"}
            response = httpx.get(url, params={"q": params}, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text[:500]
        except httpx.HTTPError as e:
            return f"API error: {e}"
    return call_api
```

---

**48. How do you debug LangChain agent decision-making?**

```python
from langchain.agents import AgentExecutor

# verbose=True shows Thought/Action/Observation
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
result = executor.invoke({"input": "What is 15% of 200?"})

# Inspect the reasoning trace
for step in result["intermediate_steps"]:
    action, observation = step
    print(f"Tool: {action.tool}")
    print(f"Input: {action.tool_input}")
    print(f"Output: {observation}\n")
```

---

### Advanced

**49. Write a function to implement a custom LangChain agent.**

```python
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_core.tools import tool
from typing import List, Tuple, Union
import re

def create_custom_agent(llm, tools):
    """A simple custom agent that always uses the first available tool."""
    tool_names = [t.name for t in tools]
    tool_map = {t.name: t for t in tools}

    def agent_logic(inputs: dict) -> Union[AgentAction, AgentFinish]:
        user_input = inputs["input"]
        # Simple heuristic: if input contains "search", use search tool
        for tool_name in tool_names:
            if tool_name.lower() in user_input.lower():
                return AgentAction(tool=tool_name, tool_input=user_input, log=f"Using {tool_name}")
        return AgentFinish(return_values={"output": f"No tool needed. Answer: {user_input}"}, log="")

    from langchain_core.runnables import RunnableLambda
    agent = RunnableLambda(agent_logic)
    return AgentExecutor(agent=agent, tools=tools)
```

---

**50. How do you optimize LangChain agents for complex NLP tasks?**

```python
from langchain.agents import AgentExecutor

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,              # Prevent runaway loops
    max_execution_time=60,         # Wall-clock timeout
    return_intermediate_steps=True, # For debugging
    handle_parsing_errors=True,    # Don't crash on malformed tool calls
    early_stopping_method="generate"  # Let LLM generate a final answer if stuck
)
```

---

**51. Write a function to implement a multi-agent system in LangChain.**

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_specialized_agent(name: str, system_prompt: str, tools: list):
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, name=name)

@tool
def search_papers(query: str) -> str:
    """Search academic papers."""
    return f"Papers on '{query}': [paper1, paper2]"

@tool
def write_summary(content: str) -> str:
    """Write a structured summary."""
    return f"Summary: {content[:200]}"

researcher = create_specialized_agent("Researcher", "You are a research specialist.", [search_papers])
writer = create_specialized_agent("Writer", "You are a technical writer.", [write_summary])

def orchestrate(query: str) -> str:
    research = researcher.invoke({"input": query})["output"]
    summary = writer.invoke({"input": research})["output"]
    return summary
```

---

**52. How do you implement a LangChain agent with structured tool outputs?**

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="Name of the city")
    unit: str = Field(default="celsius", description="Temperature unit: celsius or fahrenheit")

def get_weather_func(city: str, unit: str = "celsius") -> str:
    return f"Weather in {city}: 22°{unit[0].upper()}"

weather_tool = StructuredTool.from_function(
    func=get_weather_func,
    name="get_weather",
    description="Get current weather for a city",
    args_schema=WeatherInput,
)
```

---

**53. Write a function to handle tool timeouts in LangChain agents.**

```python
import asyncio
from langchain_core.tools import tool
from functools import wraps

def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to a tool function."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    asyncio.coroutine(func)(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                return f"Tool timed out after {timeout_seconds}s. Please try a simpler query."
        return async_wrapper
    return decorator

@tool
def slow_api_call(query: str) -> str:
    """Calls an external API (with timeout protection applied at executor level)."""
    import time
    time.sleep(2)
    return f"API result for: {query}"
```

---

**54. How do you monitor LangChain agent performance in production?**

```python
from langchain_core.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List
import time
import logging

logger = logging.getLogger(__name__)

class AgentMonitorCallback(BaseCallbackHandler):
    def __init__(self):
        self.tool_calls = []
        self.start_time = None

    def on_agent_action(self, action: Any, **kwargs):
        self.tool_calls.append({
            "tool": action.tool,
            "input": str(action.tool_input)[:100],
            "timestamp": time.time()
        })
        logger.info(f"Agent calling tool: {action.tool}")

    def on_agent_finish(self, finish: Any, **kwargs):
        duration = time.time() - (self.start_time or time.time())
        logger.info(f"Agent finished. Tools used: {len(self.tool_calls)}, Duration: {duration:.2f}s")

    def on_chain_start(self, serialized: Dict, inputs: Dict, **kwargs):
        self.start_time = time.time()
```

---

## Memory

### Basic

**55. What is memory in LangChain, and how is it used in NLP?**  
Memory allows chains and agents to persist conversation history across turns, enabling contextual, multi-turn conversations. Modern LangChain uses `RunnableWithMessageHistory` with `BaseChatMessageHistory` implementations.

> **Interview tip:** `ConversationBufferMemory` and other `Memory` classes are deprecated. The modern approach decouples message storage (history store) from the chain using `RunnableWithMessageHistory`.

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# In-memory store per session
store: dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

---

**56. How do you add memory to a LangChain chain?**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
chain = prompt | llm | StrOutputParser()
store = {}

chain_with_memory = RunnableWithMessageHistory(
    chain,
    lambda sid: store.setdefault(sid, ChatMessageHistory()),
    input_messages_key="input",
    history_messages_key="history"
)

# Turn 1
r1 = chain_with_memory.invoke({"input": "My name is Alex."}, config={"configurable": {"session_id": "s1"}})
# Turn 2 — remembers "Alex"
r2 = chain_with_memory.invoke({"input": "What is my name?"}, config={"configurable": {"session_id": "s1"}})
```

---

**57. What is ChatMessageHistory in LangChain?**  
`ChatMessageHistory` is the modern replacement for `ConversationBufferMemory`. It stores a list of `HumanMessage` / `AIMessage` objects and is used as the history store in `RunnableWithMessageHistory`.

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

history = ChatMessageHistory()
history.add_user_message("Hi there!")
history.add_ai_message("Hello! How can I help?")
history.add_user_message("What's the capital of France?")

print(history.messages)
# Access all messages
for msg in history.messages:
    print(f"{type(msg).__name__}: {msg.content}")
```

---

**58. How do you retrieve memory from a LangChain conversation?**

```python
from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("What is LangChain?")
history.add_ai_message("LangChain is a framework for building LLM apps.")

# Get all messages
messages = history.messages

# Get just the last exchange
last_human = [m for m in messages if isinstance(m, type(messages[0]))][-1]

# Clear history
history.clear()
```

---

**59. What is the role of session IDs in LangChain memory?**  
Session IDs (also called `thread_id` in LangGraph) uniquely identify a conversation thread, allowing the same chain to serve multiple users with isolated memory.

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

chain_with_memory = RunnableWithMessageHistory(
    chain,
    lambda session_id: store.setdefault(session_id, ChatMessageHistory()),
    input_messages_key="input",
    history_messages_key="history"
)

# Different users have isolated histories
user_a = chain_with_memory.invoke({"input": "Hi, I'm Alice."}, config={"configurable": {"session_id": "alice"}})
user_b = chain_with_memory.invoke({"input": "Hi, I'm Bob."}, config={"configurable": {"session_id": "bob"}})
```

---

**60. How do you clear memory in a LangChain conversation?**

```python
from langchain_community.chat_message_histories import ChatMessageHistory

# In-memory
history = ChatMessageHistory()
history.add_user_message("Test")
history.clear()

# For session-based store
def clear_session(store: dict, session_id: str):
    if session_id in store:
        store[session_id].clear()
        print(f"Cleared history for session: {session_id}")
```

---

### Intermediate

**61. Write a function to create a LangChain chain with summary memory.**  
Use a summarization prompt to compress long histories before sending them to the LLM.

> **Interview tip:** `ConversationSummaryMemory` is deprecated. Implement summary compression manually in the chain using `RunnableLambda`.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory

llm = ChatOpenAI(model="gpt-4o-mini")

def summarize_history(messages, max_messages: int = 10) -> str:
    if len(messages) <= max_messages:
        return "\n".join([f"{m.type}: {m.content}" for m in messages])
    # Summarize older messages
    older = messages[:-max_messages]
    recent = messages[-max_messages:]
    summary_prompt = f"Summarize this conversation history in 2 sentences:\n" + \
                     "\n".join([f"{m.type}: {m.content}" for m in older])
    summary = llm.invoke(summary_prompt).content
    recent_text = "\n".join([f"{m.type}: {m.content}" for m in recent])
    return f"[Summary of earlier conversation: {summary}]\n\n{recent_text}"
```

---

**62. How do you implement token-limited memory in LangChain?**  
Trim message history to fit within a token budget using `trim_messages`.

```python
from langchain_core.messages import trim_messages, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",      # Keep the most recent messages
    token_counter=llm,    # Use the LLM's tokenizer for accurate counting
    include_system=True,  # Always keep system message
    allow_partial=False,
    start_on="human"      # Always start on a human message
)

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Tell me about AI."),
    AIMessage("AI stands for artificial intelligence..."),
    HumanMessage("What about machine learning?"),
]
trimmed = trimmer.invoke(messages)
```

---

**63. Write a function to store conversation history in a database.**  
Use `SQLChatMessageHistory` for persistent storage.

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_sql_history(session_id: str, db_url: str = "sqlite:///chat_history.db"):
    """Returns a persistent chat history backed by SQLite."""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=db_url
    )

# Usage with RunnableWithMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_db_memory = RunnableWithMessageHistory(
    chain,
    lambda sid: get_sql_history(sid),
    input_messages_key="input",
    history_messages_key="history"
)
```

---

**64. How do you use LangChain to implement entity-based memory?**  
Track specific named entities mentioned in the conversation.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

llm = ChatOpenAI(model="gpt-4o-mini")

entity_extraction_chain = (
    ChatPromptTemplate.from_template(
        "Extract named entities (people, places, organizations) from this text as JSON.\n"
        "Text: {text}\nReturn: {{\"entities\": [{{\"name\": ..., \"type\": ...}}]}}"
    )
    | llm
    | StrOutputParser()
)

def extract_entities(text: str) -> dict:
    result = entity_extraction_chain.invoke({"text": text})
    try:
        return json.loads(result)
    except:
        return {"entities": []}
```

---

**65. Implement a function to merge multiple memory contexts.**

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage
from typing import List

def merge_histories(histories: List[ChatMessageHistory]) -> ChatMessageHistory:
    """Merge multiple chat histories into one, ordered by creation order."""
    merged = ChatMessageHistory()
    for history in histories:
        for message in history.messages:
            merged.messages.append(message)
    return merged

# Example
history1 = ChatMessageHistory()
history1.add_user_message("What is ML?")
history1.add_ai_message("ML is machine learning.")

history2 = ChatMessageHistory()
history2.add_user_message("What is DL?")
history2.add_ai_message("DL is deep learning.")

combined = merge_histories([history1, history2])
print(len(combined.messages))  # 4
```

---

**66. How do you handle memory overflow in LangChain?**

```python
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini")
trimmer = trim_messages(max_tokens=2000, strategy="last", token_counter=llm, include_system=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

# Apply trimmer before the prompt
chain = (
    RunnablePassthrough.assign(messages=lambda x: trimmer.invoke(x["messages"]))
    | prompt
    | llm
    | StrOutputParser()
)
```

---

### Advanced

**67. Write a function to implement a custom memory type in LangChain.**

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List
import redis
import json

class RedisChatMessageHistory(BaseChatMessageHistory):
    """Redis-backed message history for production use."""
    
    def __init__(self, session_id: str, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.session_id = session_id
        self.redis_client = redis.Redis.from_url(redis_url)
        self.ttl = ttl
        self.key = f"chat_history:{session_id}"

    @property
    def messages(self) -> List[BaseMessage]:
        data = self.redis_client.get(self.key)
        if not data:
            return []
        raw = json.loads(data)
        return [HumanMessage(m["content"]) if m["type"] == "human" 
                else AIMessage(m["content"]) for m in raw]

    def add_message(self, message: BaseMessage) -> None:
        current = self.messages
        current.append(message)
        serialized = [{"type": "human" if isinstance(m, HumanMessage) else "ai",
                       "content": m.content} for m in current]
        self.redis_client.setex(self.key, self.ttl, json.dumps(serialized))

    def clear(self) -> None:
        self.redis_client.delete(self.key)
```

---

**68. How do you implement memory with vector stores in LangChain?**  
Use vector store-backed retrieval to fetch only the most relevant past messages.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

class VectorMemory:
    """Stores conversation turns as embeddings for semantic retrieval."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.turns = []

    def add_turn(self, human: str, ai: str):
        doc = Document(page_content=f"Human: {human}\nAI: {ai}", metadata={"turn": len(self.turns)})
        self.turns.append(doc)
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents([doc], self.embeddings)
        else:
            self.vectorstore.add_documents([doc])

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        if not self.vectorstore:
            return ""
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n".join([d.page_content for d in docs])
```

---

**69. Write a function to compress conversation memory.**

```python
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage, AIMessage

def compress_history(history: ChatMessageHistory, max_messages: int = 6) -> ChatMessageHistory:
    """Summarizes older messages to keep the history under max_messages."""
    messages = history.messages
    if len(messages) <= max_messages:
        return history

    llm = ChatOpenAI(model="gpt-4o-mini")
    to_compress = messages[:-max_messages]
    recent = messages[-max_messages:]
    
    conversation_text = "\n".join([f"{m.type}: {m.content}" for m in to_compress])
    summary = llm.invoke(f"Summarize this conversation concisely:\n\n{conversation_text}").content

    new_history = ChatMessageHistory()
    new_history.add_message(AIMessage(content=f"[Earlier conversation summary: {summary}]"))
    for msg in recent:
        new_history.add_message(msg)
    return new_history
```

---

**70. How do you implement memory for multi-user conversations in LangChain?**

```python
from langchain_community.chat_message_histories import ChatMessageHistory, SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import Dict

# In-memory store (development)
user_histories: Dict[str, ChatMessageHistory] = {}

def get_user_history(user_id: str) -> ChatMessageHistory:
    if user_id not in user_histories:
        user_histories[user_id] = ChatMessageHistory()
    return user_histories[user_id]

# Production: use SQL or Redis
def get_persistent_history(user_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=user_id,
        connection_string="postgresql://user:pass@localhost/chatdb"
    )

multi_user_chain = RunnableWithMessageHistory(
    chain,
    get_user_history,  # swap with get_persistent_history for production
    input_messages_key="input",
    history_messages_key="history"
)
```

---

**71. Write a function to synchronize memory across LangChain agents.**

```python
from langchain_community.chat_message_histories import ChatMessageHistory

def sync_histories(source: ChatMessageHistory, targets: list[ChatMessageHistory]):
    """Copy all messages from source history to all target histories."""
    for target in targets:
        for msg in source.messages:
            target.add_message(msg)

# Example: coordinator shares context with specialist agents
coordinator_history = ChatMessageHistory()
coordinator_history.add_user_message("User wants a travel plan to Paris.")
coordinator_history.add_ai_message("Planning Paris trip...")

research_history = ChatMessageHistory()
booking_history = ChatMessageHistory()

sync_histories(coordinator_history, [research_history, booking_history])
```

---

**72. How do you optimize memory for long conversations in LangChain?**

```python
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

llm = ChatOpenAI(model="gpt-4o-mini")

# Strategy 1: Sliding window (keep last N messages)
trimmer = trim_messages(max_tokens=1500, strategy="last", token_counter=llm, include_system=True)

# Strategy 2: Token-aware trim with partial messages allowed
trimmer_partial = trim_messages(max_tokens=1500, strategy="last", token_counter=llm, allow_partial=True)

# Strategy 3: Persistent storage to avoid re-loading full history
from langchain_community.chat_message_histories import SQLChatMessageHistory
history = SQLChatMessageHistory(session_id="user_123", connection_string="sqlite:///chats.db")
```

---

## Retrieval and Vector Stores

### Basic

**73. What is retrieval-augmented generation (RAG) in LangChain?**  
RAG enhances LLM responses by first retrieving relevant documents from a knowledge base (vector store), then including them as context in the prompt. This reduces hallucination and enables the model to answer questions about proprietary or recent data.

> **Interview tip:** RAG has three phases: **Indexing** (chunk → embed → store), **Retrieval** (embed query → vector search → top-k docs), **Generation** (stuff docs into prompt → LLM → answer). Know all three.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Indexing
texts = ["LangChain was created in 2022.", "RAG combines retrieval with generation."]
vectorstore = FAISS.from_texts(texts, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# RAG Chain
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("Context: {context}\n\nQuestion: {question}")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result = rag_chain.invoke("When was LangChain created?")
```

---

**74. How do you create a vector store in LangChain?**

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# From raw texts
vectorstore = FAISS.from_texts(
    ["Document 1 content", "Document 2 content"],
    embeddings
)

# From Document objects (preserves metadata)
docs = [
    Document(page_content="LangChain basics", metadata={"source": "docs", "chapter": 1}),
    Document(page_content="LCEL guide", metadata={"source": "docs", "chapter": 2}),
]
vectorstore = FAISS.from_documents(docs, embeddings)
```

---

**75. What is the role of embeddings in LangChain retrieval?**  
Embeddings are dense vector representations of text that capture semantic meaning. Similar texts have similar vectors. During retrieval, the query is embedded and compared to stored document vectors using cosine similarity.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Single text embedding (1536-dim vector for text-embedding-3-small)
query_vector = embeddings.embed_query("What is machine learning?")
print(f"Embedding dimensions: {len(query_vector)}")

# Batch embedding (more efficient)
doc_vectors = embeddings.embed_documents(["Document 1", "Document 2", "Document 3"])
```

---

**76. How do you use LangChain to query a vector store?**

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

vectorstore = FAISS.from_texts(["AI basics", "ML fundamentals", "DL overview"], OpenAIEmbeddings())

# Simple similarity search
docs = vectorstore.similarity_search("machine learning", k=2)

# With scores (lower = more similar for L2, higher = more similar for cosine)
docs_with_scores = vectorstore.similarity_search_with_score("machine learning", k=2)
for doc, score in docs_with_scores:
    print(f"Score: {score:.4f} | Content: {doc.page_content}")

# As a retriever (standard interface)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke("AI definition")
```

---

**77. What is the `create_retrieval_chain` in modern LangChain?**  
`create_retrieval_chain` (replacing the deprecated `RetrievalQA`) builds a full RAG chain with retrieval and answer generation.

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

vectorstore = FAISS.from_texts(["Paris is the capital of France."], OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context below.\n\nContext: {context}"),
    ("human", "{input}")
])
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

result = rag_chain.invoke({"input": "What is the capital of France?"})
print(result["answer"])
```

---

**78. How do you save and load a vector store in LangChain?**

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(["Sample text 1", "Sample text 2"], embeddings)

# Save
vectorstore.save_local("./faiss_index")

# Load
loaded_vs = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
```

---

### Intermediate

**79. Write a function to create a LangChain RAG pipeline.**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def create_rag_pipeline(raw_texts: List[str]):
    # 1. Chunk documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(raw_texts)
    
    # 2. Embed and store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 3. Build RAG chain
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the context provided. If unsure, say 'I don't know'.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)
```

---

**80. How do you implement a custom retriever in LangChain?**

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List

class KeywordRetriever(BaseRetriever):
    """Simple keyword-based retriever (no embeddings needed)."""
    documents: List[Document]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:3]]

docs = [Document(page_content="LangChain is a framework"), Document(page_content="Python is great")]
retriever = KeywordRetriever(documents=docs)
results = retriever.invoke("What is LangChain?")
```

---

**81. Write a function to update a LangChain vector store with new documents.**

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List

def update_vectorstore(vectorstore: FAISS, new_texts: List[str], metadata: List[dict] = None):
    embeddings = OpenAIEmbeddings()
    new_docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(new_texts, metadata or [{}] * len(new_texts))
    ]
    # add_documents returns the IDs of the added documents
    ids = vectorstore.add_documents(new_docs)
    print(f"Added {len(ids)} documents")
    return vectorstore
```

---

**82. How do you use LangChain to implement semantic search?**

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="The Eiffel Tower is in Paris.", metadata={"country": "France"}),
    Document(page_content="Mount Fuji is Japan's highest peak.", metadata={"country": "Japan"}),
    Document(page_content="The Colosseum is a famous landmark in Rome.", metadata={"country": "Italy"}),
]
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# Semantic search — finds conceptually related results
results = vectorstore.similarity_search("What are famous European monuments?", k=2)
for doc in results:
    print(doc.page_content, "-", doc.metadata)
```

---

**83. Implement a function to combine multiple vector stores in LangChain.**

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def merge_faiss_stores(*stores: FAISS) -> FAISS:
    """Merges multiple FAISS vector stores into the first one."""
    base_store = stores[0]
    for store in stores[1:]:
        base_store.merge_from(store)
    return base_store

embeddings = OpenAIEmbeddings()
store1 = FAISS.from_texts(["Document A", "Document B"], embeddings)
store2 = FAISS.from_texts(["Document C", "Document D"], embeddings)
merged = merge_faiss_stores(store1, store2)
```

---

**84. How do you optimize vector store retrieval in LangChain?**

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Use a faster index type for large collections
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
texts = ["doc1", "doc2", "doc3"]  # in practice, thousands of docs

# FAISS with IVF index for large-scale (approximate but fast)
# Use Chroma for persistence without serialization overhead
from langchain_community.vectorstores import Chroma
persistent_vs = Chroma.from_texts(
    texts,
    embeddings,
    persist_directory="./chroma_db"
)
persistent_vs.persist()

# Use MMR for diverse results (avoids redundant top-k)
retriever = persistent_vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
)
```

---

### Advanced

**85. Write a function to implement a hybrid search in LangChain.**  
Combines keyword (BM25) and semantic (vector) search for better coverage.

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List

def create_hybrid_retriever(documents: List[str]) -> EnsembleRetriever:
    docs = [Document(page_content=d) for d in documents]
    
    # BM25: keyword-based (exact match, good for proper nouns)
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 3
    
    # FAISS: semantic (good for meaning-based queries)
    faiss_vs = FAISS.from_documents(docs, OpenAIEmbeddings())
    faiss_retriever = faiss_vs.as_retriever(search_kwargs={"k": 3})
    
    # Ensemble combines both with Reciprocal Rank Fusion
    return EnsembleRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[0.4, 0.6]  # weight semantic higher
    )

retriever = create_hybrid_retriever(["AI fundamentals", "Machine learning basics", "Deep learning intro"])
```

---

**86. How do you implement a LangChain retriever with metadata filtering?**

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="Python tutorial", metadata={"language": "Python", "level": "beginner"}),
    Document(page_content="Advanced Python patterns", metadata={"language": "Python", "level": "advanced"}),
    Document(page_content="JavaScript basics", metadata={"language": "JavaScript", "level": "beginner"}),
]
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

# Filter by metadata
advanced_python_retriever = vectorstore.as_retriever(
    search_kwargs={"filter": {"language": "Python", "level": "advanced"}, "k": 3}
)
results = advanced_python_retriever.invoke("programming patterns")
```

---

**87. Write a function to implement a self-querying retriever in LangChain.**  
The LLM auto-generates both a search query and a metadata filter from the user's natural language question.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

def create_self_query_retriever():
    docs = [
        Document(page_content="Python guide for beginners", metadata={"language": "Python", "level": "beginner", "year": 2024}),
        Document(page_content="Advanced Rust programming", metadata={"language": "Rust", "level": "advanced", "year": 2023}),
    ]
    vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
    
    metadata_field_info = [
        AttributeInfo(name="language", description="Programming language covered", type="string"),
        AttributeInfo(name="level", description="Skill level: beginner/intermediate/advanced", type="string"),
        AttributeInfo(name="year", description="Year of publication", type="integer"),
    ]
    
    return SelfQueryRetriever.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        vectorstore=vectorstore,
        document_contents="Programming tutorials",
        metadata_field_info=metadata_field_info
    )

retriever = create_self_query_retriever()
# The LLM will auto-filter to language=Python, level=beginner
results = retriever.invoke("Show me beginner Python tutorials")
```

---

**88. How do you handle large-scale vector stores in LangChain?**

```python
# Pinecone for production-scale (millions of vectors)
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

pc = Pinecone(api_key="your-pinecone-key")
index = pc.Index("my-index")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# Add documents in batches
from langchain_core.documents import Document
batch_size = 100
documents = [Document(page_content=f"Doc {i}") for i in range(1000)]
for i in range(0, len(documents), batch_size):
    vectorstore.add_documents(documents[i:i+batch_size])
```

---

**89. Write a function to implement a contextual compression retriever.**  
Filters out irrelevant content from retrieved documents before passing to the LLM.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def create_compression_retriever():
    docs = [
        Document(page_content="LangChain is a framework. It also handles scheduling and payroll tasks for large enterprises."),
        Document(page_content="LCEL is LangChain's expression language for building composable chains."),
    ]
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    base_retriever = vectorstore.as_retriever()

    # Option 1: LLM-based extraction (most precise, more expensive)
    compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-4o-mini"))
    
    # Option 2: Embedding filter (faster, cheaper)
    # compressor = EmbeddingsFilter(embeddings=OpenAIEmbeddings(), similarity_threshold=0.75)
    
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

retriever = create_compression_retriever()
# Only the relevant sentences are returned, not the whole document
```

---

**90. How do you evaluate retrieval performance in LangChain?**

```python
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List

def evaluate_retriever(retriever, queries: List[str], relevant_contents: List[List[str]]) -> dict:
    """Compute precision@k and recall@k for a retriever."""
    precisions, recalls = [], []

    for query, relevant in zip(queries, relevant_contents):
        retrieved = [doc.page_content for doc in retriever.invoke(query)]
        
        true_positives = sum(1 for r in retrieved if any(rel in r for rel in relevant))
        precision = true_positives / len(retrieved) if retrieved else 0
        recall = true_positives / len(relevant) if relevant else 0
        
        precisions.append(precision)
        recalls.append(recall)

    return {
        "avg_precision": sum(precisions) / len(precisions),
        "avg_recall": sum(recalls) / len(recalls),
        "avg_f1": 2 * (sum(precisions) / len(precisions)) * (sum(recalls) / len(recalls)) /
                  ((sum(precisions) + sum(recalls)) / len(precisions) + 1e-9)
    }
```

---

## Evaluation and Metrics

### Basic

**91. What metrics are used to evaluate LangChain applications?**  
Common metrics: **BLEU** (n-gram overlap), **ROUGE** (recall-oriented), **BERTScore** (semantic similarity), **Faithfulness** (RAG hallucination), **Answer Relevance**, **Context Precision**. LangSmith provides built-in evaluators.

```python
from langchain.evaluation import load_evaluator

# String distance evaluator
evaluator = load_evaluator("string_distance")
result = evaluator.evaluate_strings(
    prediction="The capital is Paris.",
    reference="Paris is the capital of France."
)
print(result)
```

---

**92. How do you evaluate the quality of LangChain LLM outputs?**

```python
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI

# LLM-based criteria evaluator (no reference needed)
evaluator = load_evaluator(
    "criteria",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    criteria="helpfulness"
)
result = evaluator.evaluate_strings(
    input="What is LangChain?",
    prediction="LangChain is a framework for building LLM applications."
)
print(result["score"], result["reasoning"])
```

---

**93. What is the role of human evaluation in LangChain?**  
Human eval is the gold standard for subjective quality (tone, helpfulness, safety) that automated metrics miss. LangSmith provides annotation queues for human labelers.

```python
def collect_human_feedback(query: str, response: str) -> dict:
    """Simple human evaluation interface."""
    print(f"Query: {query}")
    print(f"Response: {response}")
    score = input("Rate (1-5): ")
    comment = input("Comment: ")
    return {"query": query, "response": response, "score": int(score), "comment": comment}
```

---

**94. How do you use LangChain to compute BLEU scores?**

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu(prediction: str, reference: str) -> float:
    """Compute sentence-level BLEU score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = prediction.lower().split()
    smoother = SmoothingFunction().method1  # avoid zero for short sentences
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)

score = compute_bleu("LangChain is a great framework", "LangChain is an excellent framework")
print(f"BLEU: {score:.3f}")
```

---

**95. What is a LangChain evaluator, and how is it used?**  
Evaluators are callable objects that score LLM outputs. They implement `.evaluate_strings()` or `.evaluate_run()`. Types include: `string_distance`, `criteria`, `embedding_distance`, `qa` (checks factual accuracy), `labeled_pairwise_string` (for A/B comparison).

```python
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_openai import ChatOpenAI

# Embedding-based semantic similarity
embedding_evaluator = load_evaluator(EvaluatorType.EMBEDDING_DISTANCE)

# LLM-based correctness with reference
qa_evaluator = load_evaluator(
    EvaluatorType.QA,
    llm=ChatOpenAI(model="gpt-4o-mini")
)
result = qa_evaluator.evaluate_strings(
    input="What is RAG?",
    prediction="RAG combines retrieval and generation.",
    reference="RAG stands for Retrieval Augmented Generation."
)
```

---

**96. How do you log evaluation results in LangChain?**

```python
import logging
import json
from datetime import datetime

logging.basicConfig(filename="eval_results.log", level=logging.INFO)

def log_evaluation(query: str, prediction: str, metrics: dict):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "prediction": prediction[:200],
        "metrics": metrics
    }
    logging.info(json.dumps(record))
    return record
```

---

### Intermediate

**97. Write a function to evaluate LangChain chain outputs.**

```python
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List

def evaluate_chain_outputs(chain, test_cases: List[dict], references: List[str]) -> dict:
    """Evaluate chain predictions against reference answers."""
    evaluator = load_evaluator("embedding_distance", embeddings=OpenAIEmbeddings())
    scores = []

    for case, ref in zip(test_cases, references):
        prediction = chain.invoke(case)
        result = evaluator.evaluate_strings(prediction=prediction, reference=ref)
        scores.append(result["score"])

    return {
        "mean_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "n_samples": len(scores)
    }
```

---

**98. How do you implement a custom evaluator in LangChain?**

```python
from langchain.evaluation import StringEvaluator
from typing import Optional

class LengthPenaltyEvaluator(StringEvaluator):
    """Penalizes responses that are too short or too long."""
    
    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return False

    @property
    def evaluation_name(self) -> str:
        return "length_penalty"

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs
    ) -> dict:
        length = len(prediction.split())
        if length < 10:
            score, reason = 0.3, "Response too short"
        elif length > 500:
            score, reason = 0.6, "Response too long"
        else:
            score, reason = 1.0, "Length appropriate"
        return {"score": score, "reasoning": reason}

evaluator = LengthPenaltyEvaluator()
result = evaluator.evaluate_strings(prediction="AI is a field of computer science.")
```

---

**99. Write a function to compute ROUGE scores for LangChain outputs.**

```python
from rouge_score import rouge_scorer
from typing import List, Dict

def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    all_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in all_scores:
            all_scores[key].append(scores[key].fmeasure)

    return {k: sum(v) / len(v) for k, v in all_scores.items()}

preds = ["LangChain is a framework for LLM apps.", "RAG improves accuracy."]
refs = ["LangChain is an LLM application framework.", "RAG reduces hallucination."]
print(compute_rouge_scores(preds, refs))
```

---

**100. How do you evaluate LangChain agent performance?**

```python
from langchain.agents import AgentExecutor
from typing import List

def evaluate_agent(executor: AgentExecutor, tasks: List[str], expected_keywords: List[List[str]]) -> dict:
    """Check if agent responses contain expected keywords."""
    results = []
    for task, keywords in zip(tasks, expected_keywords):
        try:
            result = executor.invoke({"input": task})
            output = result.get("output", "").lower()
            hit = any(kw.lower() in output for kw in keywords)
            results.append({"task": task, "success": hit, "output": output[:200]})
        except Exception as e:
            results.append({"task": task, "success": False, "error": str(e)})

    success_rate = sum(r["success"] for r in results) / len(results)
    return {"success_rate": success_rate, "details": results}
```

---

**101. Implement a function to perform A/B testing for LangChain chains.**

```python
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
from typing import List

def ab_test_chains(chain_a, chain_b, test_inputs: List[dict], references: List[str]) -> dict:
    evaluator = load_evaluator("labeled_pairwise_string", llm=ChatOpenAI(model="gpt-4o-mini"))
    results = {"chain_a_wins": 0, "chain_b_wins": 0, "ties": 0}

    for inputs, ref in zip(test_inputs, references):
        pred_a = chain_a.invoke(inputs)
        pred_b = chain_b.invoke(inputs)
        result = evaluator.evaluate_string_pairs(
            prediction=pred_a,
            prediction_b=pred_b,
            input=str(inputs),
            reference=ref
        )
        if result["value"] == "A":
            results["chain_a_wins"] += 1
        elif result["value"] == "B":
            results["chain_b_wins"] += 1
        else:
            results["ties"] += 1

    return results
```

---

**102. How do you use LangChain to evaluate factual accuracy?**

```python
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI

def evaluate_factual_accuracy(chain, qa_pairs: list) -> dict:
    """Uses a judge LLM to score factual accuracy against reference answers."""
    evaluator = load_evaluator("qa", llm=ChatOpenAI(model="gpt-4o-mini"))
    scores = []

    for pair in qa_pairs:
        prediction = chain.invoke({"input": pair["question"]})
        if isinstance(prediction, dict):
            prediction = prediction.get("answer", str(prediction))
        result = evaluator.evaluate_strings(
            input=pair["question"],
            prediction=prediction,
            reference=pair["answer"]
        )
        scores.append(result["score"])

    return {"accuracy": sum(scores) / len(scores), "n_questions": len(scores)}
```

---

### Advanced

**103. Write a function to evaluate LangChain RAG performance.**  
Measures retrieval quality (context recall, precision) and generation quality (faithfulness, answer relevance). This is often called **RAGAS evaluation**.

```python
from rouge_score import rouge_scorer
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
from typing import List

def evaluate_rag_pipeline(rag_chain, test_data: List[dict]) -> dict:
    """
    test_data: [{"question": ..., "reference_answer": ..., "reference_docs": [...]}]
    """
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    faithfulness_evaluator = load_evaluator("criteria", llm=ChatOpenAI(model="gpt-4o-mini"), criteria="factuality")
    
    scores = {"rougeL": [], "faithfulness": []}
    
    for item in test_data:
        result = rag_chain.invoke({"input": item["question"]})
        answer = result.get("answer", "")
        context = "\n".join([d.page_content for d in result.get("context", [])])
        
        # Generation quality
        r_score = rouge.score(item["reference_answer"], answer)["rougeL"].fmeasure
        scores["rougeL"].append(r_score)
        
        # Faithfulness (does the answer stick to retrieved context?)
        faith = faithfulness_evaluator.evaluate_strings(
            input=f"Context: {context[:500]}\nQuestion: {item['question']}",
            prediction=answer
        )
        scores["faithfulness"].append(faith["score"])

    return {k: sum(v) / len(v) for k, v in scores.items()}
```

---

**104. How do you implement a multi-metric evaluation in LangChain?**

```python
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def multi_metric_eval(prediction: str, reference: str, input_text: str = "") -> dict:
    # BLEU
    bleu = sentence_bleu(
        [reference.split()],
        prediction.split(),
        smoothing_function=SmoothingFunction().method1
    )
    
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = scorer.score(reference, prediction)["rougeL"].fmeasure
    
    # Embedding distance (semantic)
    emb_evaluator = load_evaluator("embedding_distance", embeddings=OpenAIEmbeddings())
    emb_score = emb_evaluator.evaluate_strings(prediction=prediction, reference=reference)["score"]
    
    return {"bleu": bleu, "rouge_l": rouge_l, "embedding_distance": emb_score}
```

---

**105–108. (Evaluation continued)**

```python
# 105: Evaluate conversational coherence
def evaluate_coherence(chain_with_memory, dialogue_turns: list) -> dict:
    from langchain_openai import ChatOpenAI
    from langchain.evaluation import load_evaluator
    
    judge = load_evaluator("criteria", llm=ChatOpenAI(model="gpt-4o-mini"), criteria="coherence")
    scores = []
    for i in range(1, len(dialogue_turns)):
        context = "\n".join([f"Human: {t['human']}\nAI: {t['ai']}" for t in dialogue_turns[:i]])
        current_q = dialogue_turns[i]["human"]
        current_a = chain_with_memory.invoke({"input": current_q})
        result = judge.evaluate_strings(input=f"Previous: {context}\nQuestion: {current_q}", prediction=current_a)
        scores.append(result["score"])
    return {"avg_coherence": sum(scores) / len(scores)}

# 106: Evaluate output diversity
def evaluate_diversity(outputs: list) -> dict:
    from collections import Counter
    all_tokens = [token for out in outputs for token in out.lower().split()]
    total = len(all_tokens)
    unique = len(set(all_tokens))
    type_token_ratio = unique / total if total else 0
    return {"type_token_ratio": type_token_ratio, "unique_tokens": unique, "total_tokens": total}

# 107: Adversarial evaluation
def adversarial_eval(chain, adversarial_inputs: list) -> dict:
    successes = 0
    for input_text in adversarial_inputs:
        try:
            result = chain.invoke({"input": input_text})
            if result and len(str(result)) > 0:
                successes += 1
        except Exception as e:
            pass  # failure
    return {"robustness_rate": successes / len(adversarial_inputs)}

# 108: Automate evaluation pipelines
def run_eval_pipeline(chain, test_suite: list) -> list:
    results = []
    for test in test_suite:
        pred = chain.invoke(test["input"])
        metrics = multi_metric_eval(str(pred), test["reference"], str(test["input"]))
        results.append({"test": test["input"], "metrics": metrics})
    return results
```

---

## Deployment and Inference

### Basic

**109. How do you deploy a LangChain application for production?**  
Use FastAPI (or Flask) to wrap chains as REST APIs. For LangChain-specific deployment, use **LangServe** which auto-generates FastAPI routes with built-in streaming and playground.

```python
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="LangChain API")
chain = ChatPromptTemplate.from_template("Define {word}") | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

# LangServe adds /invoke, /batch, /stream, /stream_log endpoints automatically
add_routes(app, chain, path="/define")

# Run with: uvicorn app:app --reload
```

---

**110–114. Basic deployment operations**

```python
# 110: Invoke a deployed chain
result = chain.invoke({"word": "algorithm"})

# 111: Save chain config (modern approach — save as code/config)
import json
config = {"model": "gpt-4o-mini", "template": "Define {word}", "temperature": 0}
with open("chain_config.json", "w") as f:
    json.dump(config, f)

# 112: Load and reconstruct chain from config
with open("chain_config.json") as f:
    cfg = json.load(f)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
chain = ChatPromptTemplate.from_template(cfg["template"]) | ChatOpenAI(model=cfg["model"]) | StrOutputParser()

# 113: Use environment variables for secrets
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"  # or use python-dotenv
from dotenv import load_dotenv
load_dotenv()  # reads .env file

# 114: Batch inference
results = chain.batch([{"word": "AI"}, {"word": "ML"}, {"word": "NLP"}], config={"max_concurrency": 5})
```

---

### Intermediate

**115. Write a function to deploy a LangChain chain with Flask.**

```python
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
chain = ChatPromptTemplate.from_template("Define {word}") | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

@app.route("/define", methods=["POST"])
def define_word():
    data = request.get_json()
    if not data or "word" not in data:
        return jsonify({"error": "Missing 'word' field"}), 400
    result = chain.invoke({"word": data["word"]})
    return jsonify({"definition": result})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8000)
```

---

**116–120. Intermediate deployment patterns**

```python
# 116: Enable caching for low latency
import langchain
from langchain_community.cache import InMemoryCache, SQLiteCache
langchain.llm_cache = InMemoryCache()  # dev
# langchain.llm_cache = SQLiteCache(".langchain.db")  # production

# 117: Real-time streaming inference
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

async def stream_response(word: str):
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    chain = ChatPromptTemplate.from_template("Explain {word} in detail.") | llm
    async for chunk in chain.astream({"word": word}):
        yield chunk.content  # stream tokens as they arrive

# 118: Secure endpoint with FastAPI authentication
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/define", dependencies=[Depends(verify_token)])
async def define_word(word: str):
    return {"definition": chain.invoke({"word": word})}

# 119: Monitor inference performance
import time
def monitored_invoke(chain, inputs: dict) -> dict:
    start = time.perf_counter()
    try:
        result = chain.invoke(inputs)
        return {"result": result, "latency_s": time.perf_counter() - start, "status": "ok"}
    except Exception as e:
        return {"result": None, "latency_s": time.perf_counter() - start, "status": "error", "error": str(e)}

# 120: Version control chains
import os
def save_versioned_chain_config(config: dict, version: str):
    os.makedirs("chain_versions", exist_ok=True)
    with open(f"chain_versions/chain_v{version}.json", "w") as f:
        import json; json.dump(config, f, indent=2)
```

---

### Advanced

**121–126. Advanced deployment patterns**

```python
# 121: A/B test two deployed chains
def ab_test_deployments(chain_a, chain_b, inputs: list) -> dict:
    results_a = chain_a.batch(inputs)
    results_b = chain_b.batch(inputs)
    return {"chain_a_outputs": results_a, "chain_b_outputs": results_b, "n": len(inputs)}

# 122: Distributed / concurrent inference
import asyncio
async def distributed_inference_async(chain, inputs: list) -> list:
    tasks = [chain.ainvoke(inp) for inp in inputs]
    return await asyncio.gather(*tasks)  # true async concurrency

# 123: Failover with .with_fallbacks()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

primary_llm = ChatOpenAI(model="gpt-4o")
fallback_llm = ChatOpenAI(model="gpt-4o-mini")
robust_llm = primary_llm.with_fallbacks([fallback_llm])
robust_chain = ChatPromptTemplate.from_template("Define {word}") | robust_llm | StrOutputParser()

# 124: Continuous learning via vector store updates
def online_learning_step(vectorstore, new_doc: str, metadata: dict = None):
    from langchain_core.documents import Document
    vectorstore.add_documents([Document(page_content=new_doc, metadata=metadata or {})])

# 125: Cost-optimized batch inference
def cost_optimized_batch(chain, inputs: list, batch_size: int = 10) -> list:
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        results.extend(chain.batch(batch, config={"max_concurrency": batch_size}))
    return results

# 126: Simple load balancing across model replicas
import random
def load_balanced_invoke(chains: list, inputs: dict) -> str:
    # Round-robin or random selection across identical model instances
    chain = random.choice(chains)
    return chain.invoke(inputs)
```

---

## Debugging and Error Handling

### Basic

**127–132. Basic debugging techniques**

```python
# 127: Enable verbose logging
from langchain_core.callbacks import StdOutCallbackHandler
result = chain.invoke({"word": "AI"}, config={"callbacks": [StdOutCallbackHandler()]})

# 128: Try-except for chain execution
try:
    result = chain.invoke({"word": "algorithm"})
except Exception as e:
    print(f"Chain failed: {type(e).__name__}: {e}")

# 129: Validate inputs before chain execution
from pydantic import BaseModel, ValidationError

class ChainInput(BaseModel):
    word: str
    def validate_word(cls, v):
        if not v.strip():
            raise ValueError("word cannot be empty")
        return v.strip()

def safe_invoke(inputs: dict) -> str:
    try:
        validated = ChainInput(**inputs)
        return chain.invoke(validated.dict())
    except ValidationError as e:
        return f"Invalid input: {e}"

# 130: LangSmith tracing (best debugging tool)
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "my-app"
# All chain invocations are now traced automatically

# 131: Model-level retries
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", max_retries=3)

# 132: Structured error logging
import logging, json
logger = logging.getLogger(__name__)
def log_chain_error(error: Exception, inputs: dict):
    logger.error(json.dumps({"error": str(error), "type": type(error).__name__, "inputs": str(inputs)[:200]}))
```

---

### Intermediate

**133–138. Intermediate debugging**

```python
# 133: Retry with exponential backoff
import time
def retry_chain(chain, inputs: dict, max_attempts: int = 3) -> str:
    for attempt in range(max_attempts):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait = 2 ** attempt
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

# 134: Inspect agent reasoning trace
from langchain.agents import AgentExecutor
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
result = executor.invoke({"input": "What is 2+2?"})
for action, obs in result["intermediate_steps"]:
    print(f"Tool: {action.tool}, Input: {action.tool_input}, Output: {obs}")

# 135: Validate output format
import json
def validate_json_output(output: str) -> dict:
    output = output.strip().lstrip("```json").rstrip("```").strip()
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON output: {e}\nRaw: {output[:200]}")

# 136: Profile chain latency
import time
def profile_chain(chain, inputs: dict) -> dict:
    start = time.perf_counter()
    result = chain.invoke(inputs)
    elapsed = time.perf_counter() - start
    return {"result": result, "latency_ms": elapsed * 1000}

# 137: Handle memory errors gracefully
def safe_memory_add(history, human_msg: str, ai_msg: str, max_messages: int = 50):
    history.add_user_message(human_msg)
    history.add_ai_message(ai_msg)
    if len(history.messages) > max_messages:
        # Keep system messages + last N messages
        history.messages = history.messages[-max_messages:]

# 138: Debug retriever results
def debug_retriever(retriever, query: str):
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents for query: '{query}'")
    for i, doc in enumerate(docs, 1):
        print(f"\n[Doc {i}] Score metadata: {doc.metadata}")
        print(f"Content: {doc.page_content[:200]}...")
    return docs
```

---

### Advanced

**139–144. Advanced error handling**

```python
# 139: Custom callback error handler
from langchain_core.callbacks.base import BaseCallbackHandler

class ResilientCallbackHandler(BaseCallbackHandler):
    def __init__(self, fallback_fn):
        self.fallback_fn = fallback_fn
        
    def on_chain_error(self, error: Exception, **kwargs):
        print(f"Chain error caught: {error}")
        return {"output": self.fallback_fn(error)}

# 140: Circuit breaker pattern
class CircuitBreaker:
    def __init__(self, fail_max=3, reset_timeout=60):
        self.failures = 0
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.last_failure_time = None
        self.state = "closed"  # closed=normal, open=failing, half-open=testing

    def call(self, chain, inputs):
        import time
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN")
        try:
            result = chain.invoke(inputs)
            self.failures = 0
            self.state = "closed"
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.fail_max:
                self.state = "open"
            raise

# 141: Detect and flag potential hallucinations
def detect_hallucination(chain, query: str, trusted_facts: list) -> dict:
    response = chain.invoke({"input": query})
    response_lower = str(response).lower()
    contradictions = [f for f in trusted_facts if not any(
        word in response_lower for word in f.lower().split()[:3]
    )]
    return {
        "response": response,
        "hallucination_risk": "HIGH" if contradictions else "LOW",
        "missing_facts": contradictions
    }

# 142: Distributed logging with structlog
import structlog
logger = structlog.get_logger()
def log_chain_execution(chain_name: str, inputs: dict, outputs: dict, latency: float):
    logger.info("chain_executed", chain=chain_name, latency_ms=round(latency*1000), 
                input_keys=list(inputs.keys()), output_keys=list(outputs.keys()))

# 143: Version compatibility check
import langchain, langchain_core
def check_langchain_version(min_version: str = "0.2.0"):
    from packaging import version
    current = version.parse(langchain.__version__)
    required = version.parse(min_version)
    if current < required:
        raise EnvironmentError(f"LangChain {min_version}+ required, found {langchain.__version__}")
    print(f"LangChain {langchain.__version__} — OK")

# 144: Debug memory growth in long conversations  
def debug_memory_size(history) -> dict:
    messages = history.messages
    total_chars = sum(len(m.content) for m in messages)
    return {
        "message_count": len(messages),
        "total_chars": total_chars,
        "approx_tokens": total_chars // 4,
        "last_5": [(type(m).__name__, m.content[:50]) for m in messages[-5:]]
    }
```

---

## Visualization and Interpretation

### Basic

**145–150. Basic visualization**

```python
# 145: Log chain steps with verbose
result = chain.invoke({"word": "AI"}, config={"callbacks": [StdOutCallbackHandler()]})

# 146: Standard logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("langchain_app")
logger.info("Chain invoked with input: %s", {"word": "AI"})

# 147: Log agent tool usage
from langchain.agents import AgentExecutor
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

# 148: Display conversation history
def print_history(history):
    for i, msg in enumerate(history.messages, 1):
        role = "Human" if "Human" in type(msg).__name__ else "AI"
        print(f"[{i}] {role}: {msg.content[:100]}...")

# 149: Callbacks for step-by-step logging
from langchain_core.callbacks import StdOutCallbackHandler
chain.invoke(inputs, config={"callbacks": [StdOutCallbackHandler()]})

# 150: Plot evaluation metrics
import matplotlib.pyplot as plt
def plot_metrics(metric_list: list, metric_name: str = "score"):
    scores = [m[metric_name] for m in metric_list if metric_name in m]
    plt.figure(figsize=(10, 4))
    plt.plot(scores, marker="o")
    plt.title(f"{metric_name} over time")
    plt.xlabel("Evaluation index")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{metric_name}_trend.png")
    plt.close()
```

---

### Intermediate

**151–156. Intermediate visualization**

```python
# 151: Visualize chain structure
def describe_chain(chain):
    print(f"Chain type: {type(chain).__name__}")
    if hasattr(chain, "steps"):
        for i, step in enumerate(chain.steps):
            print(f"  Step {i+1}: {type(step).__name__}")

# 152: FastAPI metrics dashboard
from fastapi import FastAPI
from collections import deque
import time

app = FastAPI()
request_log = deque(maxlen=100)  # Keep last 100 requests

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    request_log.append({"path": request.url.path, "latency": time.time() - start, "status": response.status_code})
    return response

@app.get("/metrics")
def get_metrics():
    return {"recent_requests": list(request_log)[-10:], "total_tracked": len(request_log)}

# 153: Track memory growth
def track_memory_growth(history, new_messages: list) -> list:
    sizes = []
    for msg in new_messages:
        history.add_user_message(msg)
        history.add_ai_message("response")
        sizes.append(sum(len(m.content) for m in history.messages))
    return sizes

# 154: Visualize retrieval results
def display_retrieval(retriever, query: str, max_chars: int = 150):
    docs = retriever.invoke(query)
    print(f"\nQuery: {query}\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.page_content[:max_chars]}{'...' if len(doc.page_content) > max_chars else ''}")
        if doc.metadata:
            print(f"       Metadata: {doc.metadata}")

# 155: Plot ROUGE trend
def plot_rouge_trend(results: list, metric: str = "rougeL"):
    scores = [r.get(metric, 0) for r in results]
    plt.figure()
    plt.plot(scores, label=metric, color="steelblue")
    plt.fill_between(range(len(scores)), scores, alpha=0.2)
    plt.title(f"{metric} over evaluations")
    plt.legend()
    plt.savefig(f"{metric}_trend.png"); plt.close()

# 156: Inspect agent reasoning trace
def display_agent_trace(result: dict):
    print(f"\nFinal Answer: {result.get('output', 'N/A')}\n")
    for i, (action, obs) in enumerate(result.get("intermediate_steps", []), 1):
        print(f"Step {i}:")
        print(f"  Thought: {action.log[:200]}")
        print(f"  Tool: {action.tool}({action.tool_input})")
        print(f"  Observation: {str(obs)[:200]}\n")
```

---

### Advanced

**157–162. Advanced visualization**

```python
# 157: Visualize embedding clusters with t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from langchain_openai import OpenAIEmbeddings

def visualize_embeddings(texts: list, labels: list = None):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents(texts)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(texts)-1))
    reduced = tsne.fit_transform(np.array(vectors))
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=range(len(texts)), cmap="tab10")
    for i, txt in enumerate(texts):
        plt.annotate(txt[:20], (reduced[i, 0], reduced[i, 1]), fontsize=8)
    plt.title("Document Embeddings (t-SNE)")
    plt.savefig("embeddings_tsne.png"); plt.close()

# 158: Real-time WebSocket streaming
from fastapi import FastAPI, WebSocket
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

ws_app = FastAPI()
streaming_chain = ChatPromptTemplate.from_template("Explain {topic} in detail.") | ChatOpenAI(model="gpt-4o-mini", streaming=True)

@ws_app.websocket("/stream")
async def stream_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        topic = await ws.receive_text()
        async for chunk in streaming_chain.astream({"topic": topic}):
            await ws.send_text(chunk.content)
        await ws.send_text("[DONE]")

# 159: Plot chain latency histogram
def benchmark_chain(chain, inputs: list) -> dict:
    import time
    latencies = []
    for inp in inputs:
        start = time.perf_counter()
        chain.invoke(inp)
        latencies.append((time.perf_counter() - start) * 1000)
    plt.hist(latencies, bins=20, color="steelblue", edgecolor="white")
    plt.xlabel("Latency (ms)"); plt.ylabel("Count"); plt.title("Chain Latency Distribution")
    plt.savefig("latency_hist.png"); plt.close()
    return {"p50": sorted(latencies)[len(latencies)//2], "p99": sorted(latencies)[int(len(latencies)*0.99)], "mean": sum(latencies)/len(latencies)}

# 160: Error rate monitoring
def plot_error_rate(results: list):
    errors = [1 if r.get("status") == "error" else 0 for r in results]
    cumulative = [sum(errors[:i+1]) / (i+1) for i in range(len(errors))]
    plt.plot(cumulative, color="red", label="Error Rate")
    plt.axhline(0.05, linestyle="--", color="orange", label="5% threshold")
    plt.legend(); plt.title("Cumulative Error Rate"); plt.savefig("error_rate.png"); plt.close()

# 161: Graphviz workflow diagram
def create_workflow_diagram(chain_steps: list):
    try:
        from graphviz import Digraph
        dot = Digraph(comment="LangChain Workflow")
        dot.attr(rankdir="LR")
        for i, step in enumerate(chain_steps):
            dot.node(str(i), step)
            if i > 0:
                dot.edge(str(i-1), str(i))
        dot.render("workflow", format="png", cleanup=True)
    except ImportError:
        print("Install graphviz: pip install graphviz")

# 162: Interactive Plotly metrics dashboard
def interactive_eval_dashboard(results: list):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    bleu = [r.get("bleu", 0) for r in results]
    rouge = [r.get("rouge_l", 0) for r in results]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["BLEU", "ROUGE-L"])
    fig.add_trace(go.Scatter(y=bleu, name="BLEU", mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(y=rouge, name="ROUGE-L", mode="lines+markers", line=dict(color="orange")), row=1, col=2)
    fig.update_layout(title="Evaluation Dashboard")
    fig.write_html("eval_dashboard.html")
```

---

## Best Practices and Optimization

### Basic

**163–168. Foundational best practices**

```python
# 163: Modular chain structure
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)

def build_summary_chain():
    return ChatPromptTemplate.from_template("Summarize: {text}") | get_llm() | StrOutputParser()

def build_qa_chain():
    return ChatPromptTemplate.from_messages([
        ("system", "Answer concisely."), ("human", "{question}")
    ]) | get_llm() | StrOutputParser()

# 164: Reproducibility with temperature=0
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # deterministic outputs

# 165: Enable LLM response caching
import langchain
from langchain_community.cache import InMemoryCache, SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain_cache.db")  # persists across restarts

# 166: Batch processing for large datasets
def process_large_dataset(texts: list, chain, batch_size: int = 20) -> list:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = [{"text": t} for t in texts[i:i+batch_size]]
        results.extend(chain.batch(batch, config={"max_concurrency": batch_size}))
    return results

# 167: Environment configuration with dotenv
from dotenv import load_dotenv
load_dotenv()  # reads .env file
# .env file: OPENAI_API_KEY=sk-... LANGCHAIN_API_KEY=...

# 168: Documentation with type hints and docstrings
from typing import List
def create_rag_chain(documents: List[str], model: str = "gpt-4o-mini"):
    """
    Build a RAG chain from a list of plain text documents.
    
    Args:
        documents: Raw text documents to index
        model: OpenAI model name to use for generation
    
    Returns:
        A retrieval chain that can be invoked with {"input": "your question"}
    """
    pass  # implementation here
```

---

### Intermediate

**169. Write a function to optimize LangChain memory usage.**

```python
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

def optimized_conversation_step(chain, history: ChatMessageHistory, user_input: str, max_tokens: int = 2000) -> str:
    """Run a conversation step with automatic history trimming."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    trimmer = trim_messages(max_tokens=max_tokens, strategy="last", token_counter=llm, include_system=True)
    
    history.add_user_message(user_input)
    trimmed_messages = trimmer.invoke(history.messages)
    
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("messages")
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"messages": trimmed_messages})
    history.add_ai_message(response)
    return response
```

---

**170. How do you implement unit tests for LangChain chains?**

```python
import unittest
from unittest.mock import patch, MagicMock
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

class TestSummaryChain(unittest.TestCase):
    def setUp(self):
        self.chain = ChatPromptTemplate.from_template("Summarize: {text}") | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

    @patch("langchain_openai.ChatOpenAI.invoke")
    def test_chain_returns_string(self, mock_invoke):
        """Chain should return a non-empty string."""
        mock_invoke.return_value = AIMessage(content="This is a summary.")
        result = self.chain.invoke({"text": "Long text about AI..."})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_chain_with_real_llm(self):
        """Integration test — requires OPENAI_API_KEY."""
        result = self.chain.invoke({"text": "LangChain is a framework for building LLM apps."})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)

if __name__ == "__main__":
    unittest.main()
```

---

---

# Part 2: LangGraph Interview Questions

> 100 questions covering LangGraph for building stateful, multi-step AI agents.

> **What is LangGraph?** LangGraph is a library built on top of LangChain that models agentic workflows as **directed graphs**. Each node is a Python function (or LLM call), edges define transitions, and a central **State** object carries data between nodes. It supports cycles (loops), conditional branching, checkpointing (persistence), and human-in-the-loop patterns — things that are impossible or awkward with linear LCEL chains.

---

## LangGraph Basics

### Basic

**1. What is LangGraph and how does it differ from LangChain LCEL?**

LangGraph models workflows as **stateful directed graphs** (nodes + edges + shared state), enabling loops, conditional branching, and fault tolerance. LCEL is a **linear pipeline** (step1 | step2 | step3) with no built-in cycles or persistent state.

> **Interview tip:** LangGraph is the preferred framework for production AI agents. Use LCEL for simple, linear inference; use LangGraph when your workflow needs loops, retries, branching, or memory that persists across turns.

```python
# LCEL: linear, no loops
chain = prompt | llm | parser  # A → B → C, always

# LangGraph: graph with possible cycles
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    messages: list
    attempts: int

graph = StateGraph(State)
graph.add_node("process", process_node)
graph.add_node("retry", retry_node)
graph.add_edge(START, "process")
graph.add_conditional_edges("process", should_retry, {"retry": "retry", "done": END})
graph.add_edge("retry", "process")  # cycle!
app = graph.compile()
```

---

**2. How do you install LangGraph?**

```bash
pip install langgraph langchain-openai

# For persistence (PostgreSQL checkpointer)
pip install langgraph-checkpoint-postgres

# For Redis checkpointer
pip install langgraph-checkpoint-redis
```

---

**3. What is a StateGraph in LangGraph?**  
`StateGraph` is the core class for building stateful graphs. It takes a **State schema** (TypedDict or Pydantic model) that defines what data flows through the graph.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # reducer: appends messages
    context: str
    iteration: int

graph_builder = StateGraph(ChatState)
```

---

**4. What are nodes in LangGraph?**  
Nodes are Python functions (or runnables) that receive the current **State**, perform some computation (LLM call, tool use, data transformation), and return a **partial state update** (dict with keys to update).

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot_node(state: ChatState) -> dict:
    """A node that calls the LLM and returns updated messages."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # add_messages reducer will append this

graph_builder.add_node("chatbot", chatbot_node)
```

---

**5. What are edges in LangGraph?**  
Edges define transitions between nodes. **Normal edges** always go from A to B. **Conditional edges** use a routing function to choose among multiple destinations.

```python
from langgraph.graph import StateGraph, START, END

# Normal edge: always A → B
graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")
graph.add_edge("node_b", END)

# Conditional edge: routing function decides next node
def route(state) -> str:
    if state["confidence"] > 0.8:
        return "high_confidence_path"
    return "low_confidence_path"

graph.add_conditional_edges(
    "node_a",
    route,
    {"high_confidence_path": "node_b", "low_confidence_path": "node_c"}
)
```

---

**6. How do you compile and run a LangGraph graph?**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class SimpleState(TypedDict):
    message: str
    count: int

def increment(state: SimpleState) -> dict:
    return {"count": state["count"] + 1}

def format_output(state: SimpleState) -> dict:
    return {"message": f"Count is now: {state['count']}"}

graph = StateGraph(SimpleState)
graph.add_node("increment", increment)
graph.add_node("format", format_output)
graph.add_edge(START, "increment")
graph.add_edge("increment", "format")
graph.add_edge("format", END)

app = graph.compile()
result = app.invoke({"message": "", "count": 0})
print(result)  # {"message": "Count is now: 1", "count": 1}
```

---

**7. What is the START and END node in LangGraph?**  
`START` and `END` are special sentinel nodes. `START` marks the entry point of the graph; `END` marks the exit. A graph can have multiple paths to `END`.

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)
graph.add_node("A", node_a)
graph.add_node("B", node_b)
graph.add_edge(START, "A")     # Entry point
graph.add_edge("A", "B")
graph.add_edge("B", END)       # Exit
# You can also add a direct shortcut:
# graph.add_conditional_edges("A", route, {"fast": END, "slow": "B"})
app = graph.compile()
```

---

### Intermediate

**8. How do you build a simple chatbot with LangGraph?**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
app = graph.compile()

result = app.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
print(result["messages"][-1].content)
```

---

**9. What is the `add_messages` reducer and why is it important?**  
`add_messages` is a **reducer** for the `messages` state key. Instead of replacing the list, it **appends** new messages to the existing list. Without it, each node would overwrite the message history.

> **Interview tip:** Reducers define *how* state gets updated when a node returns partial state. The default is full replacement. `add_messages` is the canonical example. You can write custom reducers for any list-like accumulation.

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Without Annotated[list, add_messages]: node output REPLACES messages
    # With Annotated[list, add_messages]:    node output APPENDS to messages
    messages: Annotated[list, add_messages]
    counter: int  # no reducer: each update REPLACES this value
```

---

**10. How do you visualize a LangGraph graph?**

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# In Jupyter Notebook
app = graph.compile()
display(Image(app.get_graph().draw_mermaid_png()))

# Get Mermaid diagram text
mermaid_text = app.get_graph().draw_mermaid()
print(mermaid_text)  # Paste into mermaid.live

# ASCII representation
print(app.get_graph().print_ascii())
```

---

### Advanced

**11. Implement a LangGraph graph with conditional routing.**

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: str

llm = ChatOpenAI(model="gpt-4o-mini")

def classify_intent(state: State) -> dict:
    """Classify user intent to determine routing."""
    last_msg = state["messages"][-1].content
    prompt = f"Classify this as 'question', 'task', or 'chat': '{last_msg}'. Reply with one word."
    intent = llm.invoke(prompt).content.strip().lower()
    return {"route": intent}

def answer_question(state: State) -> dict:
    response = llm.invoke([SystemMessage("You are an expert. Answer precisely.")] + state["messages"])
    return {"messages": [response]}

def execute_task(state: State) -> dict:
    response = llm.invoke([SystemMessage("You are a task executor. Be direct.")] + state["messages"])
    return {"messages": [response]}

def casual_chat(state: State) -> dict:
    response = llm.invoke([SystemMessage("Be friendly and casual.")] + state["messages"])
    return {"messages": [response]}

def router(state: State) -> str:
    return {"question": "answer", "task": "execute", "chat": "casual"}.get(state["route"], "casual")

graph = StateGraph(State)
graph.add_node("classify", classify_intent)
graph.add_node("answer", answer_question)
graph.add_node("execute", execute_task)
graph.add_node("casual", casual_chat)
graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", router, {"answer": "answer", "execute": "execute", "casual": "casual"})
for node in ["answer", "execute", "casual"]:
    graph.add_edge(node, END)
app = graph.compile()
```

---

**12. How do you implement a ReAct agent loop in LangGraph?**  
LangGraph's `create_react_agent` helper (or manual implementation) creates a loop: call LLM → if tool calls exist, execute tools → feed results back → repeat until no more tool calls.

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"{city}: Sunny, 22°C"

@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, tools=[get_weather, calculate])
result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Paris and what's 25 * 4?"}]})
print(result["messages"][-1].content)
```

---

---

## State Management

### Basic

**13. What is State in LangGraph and how is it defined?**  
State is a **TypedDict** (or Pydantic BaseModel) that acts as the shared data store flowing through the graph. Every node receives a copy of the current State and returns a partial update.

```python
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]  # accumulates with reducer
    user_id: str                              # plain value, replaced on update
    tools_used: Annotated[List[str], lambda a, b: a + b]  # custom list reducer
    error: str                               # None if no error
```

---

**14. What are reducers in LangGraph state?**  
Reducers determine how state fields get updated when multiple nodes (or parallel branches) write to the same key. Without a reducer, the last write wins. With a reducer, the values are merged according to the reducer function.

```python
from typing import TypedDict, Annotated
from operator import add  # simple addition reducer

class State(TypedDict):
    # Default: replace
    name: str
    
    # Reducer: append to list
    items: Annotated[list, lambda a, b: a + b]
    
    # Reducer: sum integers
    total: Annotated[int, add]
    
    # Reducer: merge dicts
    metadata: Annotated[dict, lambda a, b: {**a, **b}]
```

---

**15. How do you update state in LangGraph nodes?**  
Nodes return a **dict** containing only the keys they want to update. Keys not returned are left unchanged.

```python
def process_node(state: State) -> dict:
    # Only update what changed — other keys are preserved
    return {
        "messages": [AIMessage(content="Done!")],  # appended via reducer
        "iteration": state["iteration"] + 1,        # replaced
        # "user_id" NOT returned — stays unchanged
    }
```

---

**16. How do you use Pydantic models for LangGraph state?**

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated

class AgentState(BaseModel):
    messages: Annotated[List, add_messages] = Field(default_factory=list)
    plan: Optional[str] = None
    step: int = 0
    errors: List[str] = Field(default_factory=list)

graph = StateGraph(AgentState)
```

---

**17. What is a channel in LangGraph?**  
Internally, each state key corresponds to a **channel** that manages how values are stored and combined. The `Annotated[type, reducer]` syntax in TypedDict configures the channel's reducer. Channels enable concurrent node writes to be safely merged.

```python
from langgraph.graph import StateGraph
from langgraph.channels import LastValue, BinaryOperatorAggregate
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    # BinaryOperatorAggregate channel (reducer = add)
    score: Annotated[int, add]
    
    # LastValue channel (default: last write wins)
    status: str
```

---

### Intermediate

**18. How do you pass data between nodes in LangGraph?**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

class PipelineState(TypedDict):
    raw_text: str
    cleaned_text: str
    summary: str
    keywords: List[str]

def clean_text(state: PipelineState) -> dict:
    cleaned = state["raw_text"].strip().lower()
    return {"cleaned_text": cleaned}

def summarize(state: PipelineState) -> dict:
    # Uses cleaned_text from previous node
    summary = state["cleaned_text"][:100] + "..."
    return {"summary": summary}

def extract_keywords(state: PipelineState) -> dict:
    words = state["cleaned_text"].split()
    keywords = list(set(words))[:5]
    return {"keywords": keywords}

graph = StateGraph(PipelineState)
graph.add_node("clean", clean_text)
graph.add_node("summarize", summarize)
graph.add_node("keywords", extract_keywords)
graph.add_edge(START, "clean")
graph.add_edge("clean", "summarize")
graph.add_edge("clean", "keywords")  # parallel branches from clean
graph.add_edge("summarize", END)
graph.add_edge("keywords", END)
app = graph.compile()
```

---

**19. How do you implement a counter/loop with state?**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class LoopState(TypedDict):
    count: int
    result: list

def increment_and_process(state: LoopState) -> dict:
    new_result = state["result"] + [f"Processed item {state['count']}"]
    return {"count": state["count"] + 1, "result": new_result}

def should_continue(state: LoopState) -> str:
    return "continue" if state["count"] < 5 else "stop"

graph = StateGraph(LoopState)
graph.add_node("process", increment_and_process)
graph.add_edge(START, "process")
graph.add_conditional_edges("process", should_continue, {"continue": "process", "stop": END})
app = graph.compile()
result = app.invoke({"count": 0, "result": []})
print(result["count"])   # 5
print(result["result"])  # ['Processed item 0', ..., 'Processed item 4']
```

---

**20. How do you handle optional state fields in LangGraph?**

```python
from typing import TypedDict, Optional, List

class FlexibleState(TypedDict, total=False):
    # total=False makes all fields optional
    messages: List
    context: str
    error: Optional[str]
    retry_count: int

def safe_node(state: FlexibleState) -> dict:
    # Use .get() for fields that might not exist
    retry_count = state.get("retry_count", 0)
    error = state.get("error", None)
    
    if error:
        return {"retry_count": retry_count + 1, "error": None}
    return {"context": "processed"}
```

---

### Advanced

**21. Implement a LangGraph workflow with parallel branches.**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI

class ParallelState(TypedDict):
    question: str
    answers: Annotated[List[str], lambda a, b: a + b]  # merge answers

llm = ChatOpenAI(model="gpt-4o-mini")

def expert_1(state: ParallelState) -> dict:
    answer = llm.invoke(f"As a technical expert: {state['question']}").content
    return {"answers": [f"Technical: {answer}"]}

def expert_2(state: ParallelState) -> dict:
    answer = llm.invoke(f"As a business expert: {state['question']}").content
    return {"answers": [f"Business: {answer}"]}

def expert_3(state: ParallelState) -> dict:
    answer = llm.invoke(f"As an end-user: {state['question']}").content
    return {"answers": [f"User: {answer}"]}

def synthesize(state: ParallelState) -> dict:
    combined = "\n".join(state["answers"])
    synthesis = llm.invoke(f"Synthesize these perspectives:\n{combined}").content
    return {"answers": [f"Synthesis: {synthesis}"]}

graph = StateGraph(ParallelState)
graph.add_node("expert_1", expert_1)
graph.add_node("expert_2", expert_2)
graph.add_node("expert_3", expert_3)
graph.add_node("synthesize", synthesize)
# Fan out from START to all three experts
graph.add_edge(START, "expert_1")
graph.add_edge(START, "expert_2")
graph.add_edge(START, "expert_3")
# Fan in: all experts → synthesize
graph.add_edge("expert_1", "synthesize")
graph.add_edge("expert_2", "synthesize")
graph.add_edge("expert_3", "synthesize")
graph.add_edge("synthesize", END)
app = graph.compile()
```

---

**22. How do you implement a map-reduce pattern in LangGraph?**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from typing import TypedDict, Annotated, List

class MapReduceState(TypedDict):
    documents: List[str]
    summaries: Annotated[List[str], lambda a, b: a + b]
    final_summary: str

class DocumentState(TypedDict):
    document: str
    summaries: Annotated[List[str], lambda a, b: a + b]

def summarize_document(state: DocumentState) -> dict:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")
    summary = llm.invoke(f"Summarize in one sentence: {state['document']}").content
    return {"summaries": [summary]}

def combine_summaries(state: MapReduceState) -> dict:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")
    combined = "\n".join(state["summaries"])
    final = llm.invoke(f"Combine these summaries: {combined}").content
    return {"final_summary": final}

def dispatch_documents(state: MapReduceState) -> List[Send]:
    # Send each document to the summarize_document node independently
    return [Send("summarize_doc", {"document": doc, "summaries": []}) for doc in state["documents"]]

graph = StateGraph(MapReduceState)
graph.add_node("summarize_doc", summarize_document)
graph.add_node("combine", combine_summaries)
graph.add_conditional_edges(START, dispatch_documents, ["summarize_doc"])
graph.add_edge("summarize_doc", "combine")
graph.add_edge("combine", END)
app = graph.compile()
```

---

---

## Nodes and Edges

### Basic

**23. How do you add a node to a LangGraph graph?**

```python
from langgraph.graph import StateGraph

def my_node(state: dict) -> dict:
    return {"processed": True}

graph = StateGraph(dict)
graph.add_node("my_node", my_node)

# Shortcut: if function name matches desired node name
graph.add_node(my_node)  # node name = "my_node"
```

---

**24. How do you create conditional edges in LangGraph?**

```python
from langgraph.graph import StateGraph, END

def router(state) -> str:
    """Returns the name of the next node."""
    if state.get("error"):
        return "error_handler"
    elif state.get("complete"):
        return END
    else:
        return "process"

graph.add_conditional_edges(
    "current_node",           # source node
    router,                   # routing function
    {                         # mapping: return value → destination node
        "error_handler": "error_handler",
        "process": "process",
        END: END
    }
)
```

---

**25. What is `add_edge` vs `add_conditional_edges`?**

| Feature | `add_edge` | `add_conditional_edges` |
|---|---|---|
| Routing | Fixed: always goes to the same node | Dynamic: routing function decides |
| Use case | Linear sequential steps | Branching, loops, if/else logic |
| Performance | Slightly faster (no function call) | Adds one LLM/function call overhead |

```python
# Fixed: A always goes to B
graph.add_edge("A", "B")

# Dynamic: routing function decides where A goes
graph.add_conditional_edges("A", decide_next, {"B": "B", "C": "C", "__end__": END})
```

---

**26. How do you add entry and finish points?**

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)
graph.add_node("first", first_node)
graph.add_node("last", last_node)

# Method 1: explicit edges from START and to END
graph.add_edge(START, "first")
graph.add_edge("last", END)

# Method 2: set_entry_point / set_finish_point (older API, still works)
graph.set_entry_point("first")    # equivalent to add_edge(START, "first")
graph.set_finish_point("last")    # equivalent to add_edge("last", END)
```

---

### Intermediate

**27. How do you implement a self-correction loop in LangGraph?**

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Optional

class QualityState(TypedDict):
    question: str
    draft: Optional[str]
    critique: Optional[str]
    final: Optional[str]
    iterations: int

llm = ChatOpenAI(model="gpt-4o-mini")

def generate_draft(state: QualityState) -> dict:
    draft = llm.invoke(f"Answer: {state['question']}").content
    return {"draft": draft, "iterations": state.get("iterations", 0) + 1}

def critique_draft(state: QualityState) -> dict:
    critique = llm.invoke(
        f"Rate this answer 1-10 and explain issues:\nQ: {state['question']}\nA: {state['draft']}"
    ).content
    return {"critique": critique}

def should_refine(state: QualityState) -> str:
    # Simple heuristic: if critique mentions low score, refine
    if "low" in state["critique"].lower() or state["iterations"] < 2:
        return "refine" if state["iterations"] < 3 else "accept"
    return "accept"

def refine_answer(state: QualityState) -> dict:
    improved = llm.invoke(
        f"Improve this answer based on critique:\nQ: {state['question']}\nA: {state['draft']}\nCritique: {state['critique']}"
    ).content
    return {"draft": improved}

def accept_answer(state: QualityState) -> dict:
    return {"final": state["draft"]}

graph = StateGraph(QualityState)
graph.add_node("generate", generate_draft)
graph.add_node("critique", critique_draft)
graph.add_node("refine", refine_answer)
graph.add_node("accept", accept_answer)
graph.add_edge(START, "generate")
graph.add_edge("generate", "critique")
graph.add_conditional_edges("critique", should_refine, {"refine": "refine", "accept": "accept"})
graph.add_edge("refine", "generate")  # loop back!
graph.add_edge("accept", END)
app = graph.compile()
```

---

**28. How do you use `Send` for dynamic fan-out in LangGraph?**  
`Send` allows you to dynamically dispatch work to a node for each item in a list — like a parallel map operation.

```python
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List

class BatchState(TypedDict):
    items: List[str]
    results: Annotated[List[str], lambda a, b: a + b]

class ItemState(TypedDict):
    item: str
    results: Annotated[List[str], lambda a, b: a + b]

def process_item(state: ItemState) -> dict:
    return {"results": [f"Processed: {state['item'].upper()}"]}

def fan_out(state: BatchState) -> List[Send]:
    return [Send("process_item", {"item": item, "results": []}) for item in state["items"]]

graph = StateGraph(BatchState)
graph.add_node("process_item", process_item)
graph.add_conditional_edges(START, fan_out, ["process_item"])
graph.add_edge("process_item", END)
app = graph.compile()
result = app.invoke({"items": ["apple", "banana", "cherry"], "results": []})
print(result["results"])  # ['Processed: APPLE', 'Processed: BANANA', 'Processed: CHERRY']
```

---

**29. How do you implement error handling nodes in LangGraph?**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional

class ResilientState(TypedDict):
    input: str
    output: Optional[str]
    error: Optional[str]
    attempts: int

def risky_node(state: ResilientState) -> dict:
    try:
        # Simulate risky operation
        if state["attempts"] < 2:
            raise ValueError("Simulated transient failure")
        return {"output": f"Success after {state['attempts']} attempts", "error": None}
    except Exception as e:
        return {"error": str(e), "attempts": state["attempts"] + 1}

def error_handler(state: ResilientState) -> dict:
    # Log error, maybe alert, then decide to retry or fail
    print(f"Handling error: {state['error']}")
    return {"error": None}  # clear error for retry

def route_on_error(state: ResilientState) -> str:
    if state["error"] and state["attempts"] < 3:
        return "handle_error"
    elif state["error"]:
        return "fail"
    return "done"

graph = StateGraph(ResilientState)
graph.add_node("risky", risky_node)
graph.add_node("handle_error", error_handler)
graph.add_edge(START, "risky")
graph.add_conditional_edges("risky", route_on_error, {"handle_error": "handle_error", "fail": END, "done": END})
graph.add_edge("handle_error", "risky")
app = graph.compile()
```

---

### Advanced

**30. How do you implement subgraphs in LangGraph?**  
Subgraphs let you compose complex agents from smaller, reusable graph modules.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Define a reusable subgraph
class SubgraphState(TypedDict):
    data: str

def preprocess(state: SubgraphState) -> dict:
    return {"data": state["data"].strip().lower()}

def transform(state: SubgraphState) -> dict:
    return {"data": f"[processed] {state['data']}"}

sub_builder = StateGraph(SubgraphState)
sub_builder.add_node("preprocess", preprocess)
sub_builder.add_node("transform", transform)
sub_builder.add_edge(START, "preprocess")
sub_builder.add_edge("preprocess", "transform")
sub_builder.add_edge("transform", END)
subgraph = sub_builder.compile()

# Use the subgraph as a node in a parent graph
class ParentState(TypedDict):
    data: str
    final: str

def finalize(state: ParentState) -> dict:
    return {"final": state["data"].upper()}

parent = StateGraph(ParentState)
parent.add_node("subgraph", subgraph)  # compiled subgraph as node
parent.add_node("finalize", finalize)
parent.add_edge(START, "subgraph")
parent.add_edge("subgraph", "finalize")
parent.add_edge("finalize", END)
app = parent.compile()
```

---

---

## LangGraph Agents and Tool Calling

### Basic

**31. How do you create a tool-calling agent in LangGraph?**

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search for information on the web."""
    return f"Results for '{query}': [top results here]"

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    return f"{ticker}: $150.00"

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, tools=[search_web, get_stock_price])
result = agent.invoke({"messages": [{"role": "user", "content": "What is AAPL's stock price?"}]})
```

---

**32. How does tool calling work internally in LangGraph agents?**  
The LLM returns a message with **tool_calls** (structured JSON). The `tools_node` in LangGraph intercepts these, executes the corresponding Python functions, and returns `ToolMessage` results. The loop continues until the LLM produces a final answer with no tool calls.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import TypedDict, Annotated

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [multiply]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_llm(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

def should_use_tool(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else END

tool_node = ToolNode(tools)

graph = StateGraph(State)
graph.add_node("llm", call_llm)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_use_tool, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")  # loop back after tool execution
app = graph.compile()
```

---

**33. What is `ToolNode` in LangGraph?**  
`ToolNode` is a prebuilt LangGraph node that automatically executes all tool calls in the last `AIMessage`, returns `ToolMessage` results, and appends them to the message list.

```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

@tool
def to_upper(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()

tool_node = ToolNode([add, to_upper])
# tool_node receives state with messages including an AIMessage with tool_calls
# It executes all tool calls and returns {"messages": [ToolMessage(...), ...]}
```

---

**34. How do you add custom system prompts to a LangGraph agent?**

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results: {query}"

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(
    llm,
    tools=[search],
    state_modifier=SystemMessage(
        "You are a helpful research assistant. Always cite your sources. "
        "Be concise and factual."
    )
)
```

---

**35. How do you limit the number of iterations in a LangGraph agent loop?**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

class BoundedState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int

def call_llm(state: BoundedState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response], "steps": state["steps"] + 1}

def should_continue(state: BoundedState) -> str:
    last = state["messages"][-1]
    if state["steps"] >= 10:  # hard limit
        return "force_stop"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "use_tool"
    return "done"

# Also: use recursion_limit at compile time
app = graph.compile()
result = app.invoke(
    {"messages": [("user", "Research AI trends")], "steps": 0},
    config={"recursion_limit": 10}  # raises GraphRecursionError if exceeded
)
```

---

### Intermediate

**36. How do you implement a multi-tool agent with error recovery?**

```python
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

@tool
def search(query: str) -> str:
    """Search the web."""
    if "error" in query.lower():
        raise ValueError("Search service unavailable")
    return f"Found: {query} results"

@tool
def calculate(expr: str) -> str:
    """Calculate math expression."""
    return str(eval(expr, {"__builtins__": {}}, {}))

class State(TypedDict):
    messages: Annotated[list, add_messages]
    errors: Annotated[list, lambda a, b: a + b]

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([search, calculate])

def llm_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

class SafeToolNode(ToolNode):
    def _run(self, *args, **kwargs):
        try:
            return super()._run(*args, **kwargs)
        except Exception as e:
            return {"messages": [ToolMessage(content=f"Tool error: {e}", tool_call_id="error")], "errors": [str(e)]}

safe_tools = SafeToolNode([search, calculate])
```

---

**37. How do you pass context to tools in LangGraph?**  
Use `InjectedState` to inject state values into tool arguments automatically.

```python
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from langgraph.prebuilt import InjectedState

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    permissions: list

@tool
def get_user_data(
    query: str,
    user_id: Annotated[str, InjectedState("user_id")],  # Injected from state
    permissions: Annotated[list, InjectedState("permissions")]
) -> str:
    """Get data for the current user. user_id and permissions are injected automatically."""
    if "admin" not in permissions:
        return "Access denied"
    return f"Data for user {user_id}: {query}"
```

---

**38. How do you implement streaming tool results in LangGraph?**

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def long_running_task(description: str) -> str:
    """Performs a long computation."""
    import time; time.sleep(1)
    return f"Completed: {description}"

agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), [long_running_task])

# Stream updates as they happen
for chunk in agent.stream({"messages": [{"role": "user", "content": "Run a task about AI"}]}):
    if "agent" in chunk:
        messages = chunk["agent"]["messages"]
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                print(f"Agent: {msg.content}")
    elif "tools" in chunk:
        for msg in chunk["tools"]["messages"]:
            print(f"Tool result: {msg.content}")
```

---

### Advanced

**39. Implement a planning agent in LangGraph (Plan-and-Execute).**

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict, List, Optional

class PlanState(TypedDict):
    goal: str
    plan: Optional[List[str]]
    current_step: int
    results: List[str]
    final_answer: Optional[str]

llm = ChatOpenAI(model="gpt-4o-mini")

def planner(state: PlanState) -> dict:
    """Break the goal into actionable steps."""
    prompt = f"Create a 3-5 step plan to accomplish: '{state['goal']}'\nReturn JSON: {{\"steps\": [\"step1\", ...]}}"
    response = llm.invoke(prompt).content
    import json
    plan = json.loads(response)["steps"]
    return {"plan": plan, "current_step": 0}

def executor(state: PlanState) -> dict:
    """Execute the current step."""
    step = state["plan"][state["current_step"]]
    result = llm.invoke(f"Execute this step: {step}").content
    return {"results": state["results"] + [result], "current_step": state["current_step"] + 1}

def synthesizer(state: PlanState) -> dict:
    """Combine all step results into a final answer."""
    steps_summary = "\n".join([f"{i+1}. {r}" for i, r in enumerate(state["results"])])
    answer = llm.invoke(f"Goal: {state['goal']}\nResults:\n{steps_summary}\nProvide final answer:").content
    return {"final_answer": answer}

def route_execution(state: PlanState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "execute"
    return "synthesize"

graph = StateGraph(PlanState)
graph.add_node("plan", planner)
graph.add_node("execute", executor)
graph.add_node("synthesize", synthesizer)
graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_conditional_edges("execute", route_execution, {"execute": "execute", "synthesize": "synthesize"})
graph.add_edge("synthesize", END)
app = graph.compile()
```

---

**40. How do you implement tool choice and routing in LangGraph?**

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"Docs result for: {query}"

@tool  
def query_database(sql: str) -> str:
    """Query the database with SQL."""
    return f"DB result for: {sql}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

tools = [search_docs, query_database, send_email]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools, tool_choice="auto")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

def route(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

tool_node = ToolNode(tools)
graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()
```

---

---

## LangGraph Memory and Persistence

### Basic

**41. What is a checkpointer in LangGraph?**  
A checkpointer saves the graph's state after each node execution to a persistent store (in-memory, SQLite, PostgreSQL, Redis). This enables: **multi-turn conversations** (memory between `.invoke()` calls), **fault tolerance** (resume from last checkpoint on failure), and **human-in-the-loop** (pause and resume workflows).

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

# thread_id scopes the checkpoint to a specific conversation
config = {"configurable": {"thread_id": "user_123_session_1"}}
result = app.invoke({"messages": [{"role": "user", "content": "Hi!"}]}, config=config)
# State is saved after each node. Next call resumes from saved state.
result2 = app.invoke({"messages": [{"role": "user", "content": "What did I just say?"}]}, config=config)
```

---

**42. How do you use `InMemorySaver` for development?**

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_time() -> str:
    """Get the current time."""
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")

memory = InMemorySaver()
agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), [get_time], checkpointer=memory)

session_config = {"configurable": {"thread_id": "session_42"}}

# Turn 1
r1 = agent.invoke({"messages": [("user", "What time is it?")]}, config=session_config)
# Turn 2 — agent remembers context from turn 1
r2 = agent.invoke({"messages": [("user", "What did you just tell me?")]}, config=session_config)
```

---

**43. How do you use SQLite for persistent checkpointing?**

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Use as a context manager for proper connection handling
with SqliteSaver.from_conn_string("./checkpoints.db") as saver:
    app = graph.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "persistent_thread_1"}}
    result = app.invoke({"messages": [("user", "Remember this: my favorite color is blue.")]}, config=config)

# Later run — state is loaded from SQLite
with SqliteSaver.from_conn_string("./checkpoints.db") as saver:
    app = graph.compile(checkpointer=saver)
    result = app.invoke({"messages": [("user", "What is my favorite color?")]}, config=config)
```

---

**44. How do you use PostgreSQL for production checkpointing?**

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

DB_URI = "postgresql://user:password@localhost:5432/langgraph_db"

with ConnectionPool(conninfo=DB_URI) as pool:
    saver = PostgresSaver(pool)
    saver.setup()  # Creates required tables on first run
    
    app = graph.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "prod_thread_abc123"}}
    result = app.invoke(initial_state, config=config)
```

---

**45. How do you retrieve the current state of a LangGraph thread?**

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "thread_1"}}
app.invoke({"messages": [("user", "Hello!")]}, config=config)

# Get current state without invoking the graph
state = app.get_state(config)
print(state.values)         # Current state dict
print(state.next)           # Next nodes to execute (if any)
print(state.metadata)       # Checkpoint metadata

# Get state history (all checkpoints)
for snapshot in app.get_state_history(config):
    print(snapshot.metadata["step"], snapshot.values)
```

---

### Intermediate

**46. How do you implement cross-session memory in LangGraph?**  
Combine thread-scoped checkpoints (short-term) with a vector store (long-term semantic memory).

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List

class LongTermMemoryState(TypedDict):
    messages: Annotated[List, add_messages]
    user_id: str
    retrieved_memories: List[str]

# Long-term memory store (shared across sessions)
long_term_store = FAISS.from_texts(["placeholder"], OpenAIEmbeddings())

def retrieve_memories(state: LongTermMemoryState) -> dict:
    query = state["messages"][-1].content
    docs = long_term_store.similarity_search(query, k=3)
    memories = [d.page_content for d in docs if d.page_content != "placeholder"]
    return {"retrieved_memories": memories}

def save_to_long_term(state: LongTermMemoryState) -> dict:
    # Save important information to long-term store
    last_ai = next((m for m in reversed(state["messages"]) if hasattr(m, "content")), None)
    if last_ai:
        long_term_store.add_documents([
            Document(page_content=last_ai.content, metadata={"user_id": state["user_id"]})
        ])
    return {}
```

---

**47. How do you update state manually in LangGraph?**

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage

memory = InMemorySaver()
app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "thread_1"}}

# Run the graph
app.invoke({"messages": [("user", "Hi!")]}, config=config)

# Manually update state (e.g., inject context, fix a mistake)
app.update_state(
    config,
    {"messages": [AIMessage(content="I want to add this context to the history.")]},
    as_node="chatbot"  # attributes the update to this node
)

# Verify the update
state = app.get_state(config)
print(state.values["messages"])
```

---

**48. How do you implement thread-based multi-user memory?**

```python
from langgraph.checkpoint.sqlite import SqliteSaver

def get_agent_for_user(user_id: str):
    """Returns an agent pre-configured with the user's session."""
    with SqliteSaver.from_conn_string("chat.db") as saver:
        app = agent_graph.compile(checkpointer=saver)
        
        def invoke(message: str, session_id: str = "default"):
            thread_id = f"{user_id}_{session_id}"
            config = {"configurable": {"thread_id": thread_id}}
            return app.invoke({"messages": [("user", message)]}, config=config)
        
        return invoke

user_alice = get_agent_for_user("alice")
user_bob = get_agent_for_user("bob")

# Completely isolated conversation histories
alice_response = user_alice("My name is Alice.")
bob_response = user_bob("My name is Bob.")
```

---

### Advanced

**49. How do you implement time-travel debugging with LangGraph?**

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "debug_thread"}}

# Run several turns
app.invoke({"messages": [("user", "Step 1")]}, config=config)
app.invoke({"messages": [("user", "Step 2")]}, config=config)
app.invoke({"messages": [("user", "Step 3")]}, config=config)

# Get all checkpoints
history = list(app.get_state_history(config))
print(f"Total checkpoints: {len(history)}")

# Roll back to an earlier state and re-run from there
step_1_checkpoint = history[-2]  # second-to-last (oldest)
rollback_config = {"configurable": {"thread_id": "debug_thread", "checkpoint_id": step_1_checkpoint.config["configurable"]["checkpoint_id"]}}

# Fork from step 1 with a different input
fork_config = {"configurable": {"thread_id": "debug_thread_fork"}}
app.invoke({"messages": [("user", "Different Step 2")]}, config=fork_config)
```

---

**50. How do you implement persistent long-term memory with LangGraph + a memory store?**

```python
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Annotated
from langgraph.prebuilt import InjectedStore

# LangGraph's built-in memory store for cross-thread persistence
store = InMemoryStore()

@tool
def save_memory(memory: str, user_id: Annotated[str, InjectedState("user_id")], mem_store: Annotated[InMemoryStore, InjectedStore()]) -> str:
    """Save an important memory about the user."""
    namespace = (user_id, "memories")
    mem_store.put(namespace, f"mem_{hash(memory)}", {"content": memory})
    return f"Saved: {memory}"

@tool  
def recall_memories(query: str, user_id: Annotated[str, InjectedState("user_id")], mem_store: Annotated[InMemoryStore, InjectedStore()]) -> str:
    """Recall memories about the user."""
    namespace = (user_id, "memories")
    memories = [item.value["content"] for item in mem_store.search(namespace, query=query)]
    return "\n".join(memories) if memories else "No relevant memories found."

agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), [save_memory, recall_memories], store=store)
```

---

---

## Human-in-the-Loop

### Basic

**51. What is human-in-the-loop (HITL) in LangGraph?**  
HITL allows a LangGraph workflow to **pause** at a designated point, wait for human input (approval, correction, or additional data), then **resume** from where it stopped. This is essential for high-stakes actions (sending emails, writing to databases, executing code).

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class ApprovalState(TypedDict):
    action: str
    approved: bool
    result: str

def propose_action(state: ApprovalState) -> dict:
    return {"action": "Delete all records from the users table"}

def execute_action(state: ApprovalState) -> dict:
    if state["approved"]:
        return {"result": "Action executed successfully"}
    return {"result": "Action cancelled"}

memory = InMemorySaver()
graph = StateGraph(ApprovalState)
graph.add_node("propose", propose_action)
graph.add_node("execute", execute_action)
graph.add_edge(START, "propose")
graph.add_edge("propose", "execute")  # interrupt_before this!
graph.add_edge("execute", END)

# Interrupt BEFORE the execute node — graph pauses here for human review
app = graph.compile(checkpointer=memory, interrupt_before=["execute"])
```

---

**52. How do you pause and resume a LangGraph workflow for human approval?**

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["execute"])
config = {"configurable": {"thread_id": "approval_thread_1"}}

# Step 1: Run until interrupt
result = app.invoke({"action": "", "approved": False, "result": ""}, config=config)
# Graph pauses before "execute" node

# Step 2: Human reviews the proposed action
state = app.get_state(config)
print(f"Proposed action: {state.values['action']}")
print(f"Next node: {state.next}")  # ('execute',)

# Step 3: Human approves/rejects
human_decision = True  # or False to cancel

# Step 4: Update state and resume
app.update_state(config, {"approved": human_decision})
final_result = app.invoke(None, config=config)  # None = resume from checkpoint
print(final_result["result"])
```

---

**53. How do you implement `interrupt_after` vs `interrupt_before`?**

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()

# interrupt_before: pause BEFORE the node runs (for approval before action)
app_before = graph.compile(checkpointer=memory, interrupt_before=["dangerous_node"])

# interrupt_after: pause AFTER the node runs (to review output before proceeding)
app_after = graph.compile(checkpointer=memory, interrupt_after=["generate_draft"])

# Use case:
# interrupt_before → "Do you want to send this email?" (before send)
# interrupt_after  → "Review this draft before I publish it" (after generate, before publish)
```

---

**54. How do you implement a human feedback node in LangGraph?**

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional

class FeedbackState(TypedDict):
    question: str
    ai_response: str
    human_feedback: Optional[str]
    improved_response: Optional[str]

def generate_response(state: FeedbackState) -> dict:
    from langchain_openai import ChatOpenAI
    response = ChatOpenAI(model="gpt-4o-mini").invoke(state["question"]).content
    return {"ai_response": response}

def incorporate_feedback(state: FeedbackState) -> dict:
    if not state.get("human_feedback"):
        return {}
    from langchain_openai import ChatOpenAI
    improved = ChatOpenAI(model="gpt-4o-mini").invoke(
        f"Original answer: {state['ai_response']}\n"
        f"Human feedback: {state['human_feedback']}\n"
        "Improve the answer based on the feedback."
    ).content
    return {"improved_response": improved}

from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()
graph = StateGraph(FeedbackState)
graph.add_node("generate", generate_response)
graph.add_node("incorporate_feedback", incorporate_feedback)
graph.add_edge(START, "generate")
graph.add_edge("generate", "incorporate_feedback")
graph.add_edge("incorporate_feedback", END)
app = graph.compile(checkpointer=memory, interrupt_after=["generate"])
```

---

### Advanced

**55. How do you implement a full HITL approval workflow?**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Optional, List
from langchain_openai import ChatOpenAI

class ContentWorkflowState(TypedDict):
    topic: str
    draft: Optional[str]
    human_edits: Optional[str]
    published: bool
    revision_history: List[str]

llm = ChatOpenAI(model="gpt-4o-mini")

def write_draft(state: ContentWorkflowState) -> dict:
    draft = llm.invoke(f"Write a 100-word article about: {state['topic']}").content
    return {"draft": draft, "revision_history": [draft]}

def apply_edits(state: ContentWorkflowState) -> dict:
    if state.get("human_edits"):
        revised = llm.invoke(
            f"Apply these edits to the article:\nOriginal: {state['draft']}\nEdits: {state['human_edits']}"
        ).content
        return {"draft": revised, "revision_history": state["revision_history"] + [revised], "human_edits": None}
    return {}

def should_revise(state: ContentWorkflowState) -> str:
    return "apply_edits" if state.get("human_edits") else "publish"

def publish(state: ContentWorkflowState) -> dict:
    print(f"PUBLISHED: {state['draft'][:100]}...")
    return {"published": True}

memory = InMemorySaver()
graph = StateGraph(ContentWorkflowState)
graph.add_node("write", write_draft)
graph.add_node("apply_edits", apply_edits)
graph.add_node("publish", publish)
graph.add_edge(START, "write")
graph.add_conditional_edges("write", lambda s: "apply_edits" if s.get("human_edits") else "check_approval", {})
graph.add_edge("apply_edits", END)
graph.add_edge("publish", END)
# Interrupt after "write" for human review
app = graph.compile(checkpointer=memory, interrupt_after=["write"])
```

---

---

## Multi-Agent Systems

### Basic

**56. What is a multi-agent system in LangGraph?**  
A multi-agent system has multiple AI agents, each specialized for different tasks, coordinated by either a **supervisor** (centralized) or a **swarm** pattern (peer-to-peer). LangGraph enables this with subgraphs, shared state, and inter-agent communication.

```python
# Pattern 1: Supervisor delegates to specialized agents
# Pattern 2: Agents communicate via shared state
# Pattern 3: Network/swarm with peer handoffs

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List

class SupervisorState(TypedDict):
    task: str
    subtasks: List[str]
    results: Annotated[List[str], lambda a, b: a + b]
    final: str
    assigned_to: str  # which agent to call next
```

---

**57. How do you build a supervisor agent in LangGraph?**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List
import json

llm = ChatOpenAI(model="gpt-4o-mini")
workers = ["researcher", "writer", "reviewer"]

class SupervisorState(TypedDict):
    messages: Annotated[List, add_messages]
    next: str  # which worker to call next

def supervisor(state: SupervisorState) -> dict:
    """Supervisor decides which worker to call next, or FINISH."""
    prompt = (
        f"You are a supervisor managing: {', '.join(workers)}.\n"
        f"Current conversation: {state['messages'][-3:]}\n"
        f"Which worker should act next? Reply with one of: {', '.join(workers + ['FINISH'])}. Just the name."
    )
    next_worker = llm.invoke(prompt).content.strip()
    return {"next": next_worker}

def researcher(state: SupervisorState) -> dict:
    response = llm.invoke(f"As a researcher, respond to: {state['messages'][-1].content}").content
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=f"[Researcher] {response}")]}

def writer(state: SupervisorState) -> dict:
    response = llm.invoke(f"As a writer, respond to: {state['messages'][-1].content}").content
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=f"[Writer] {response}")]}

graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", lambda s: s["next"], {
    "researcher": "researcher", "writer": "writer", "FINISH": END
})
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
app = graph.compile()
```

---

**58. How do agents communicate in LangGraph?**  
Agents share the same `State` object. They read from it (inputs) and write partial updates to it (outputs). The `add_messages` reducer is the standard mechanism for agents to append their outputs to a shared conversation history.

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from typing import TypedDict, Annotated, List

class SharedState(TypedDict):
    messages: Annotated[List, add_messages]  # All agents read and write here
    context: str  # Shared context accessible to all agents

def agent_a(state: SharedState) -> dict:
    # Read shared context, write a message
    return {"messages": [AIMessage(content=f"Agent A processed: {state['context']}")]}

def agent_b(state: SharedState) -> dict:
    # Can read Agent A's message from state["messages"]
    last_msg = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"Agent B responding to: {last_msg}")]}
```

---

### Intermediate

**59. How do you implement agent handoffs in LangGraph?**

```python
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from typing import TypedDict, Annotated, List

class HandoffState(TypedDict):
    messages: Annotated[List, add_messages]
    current_agent: str

def create_agent_with_handoff(name: str, specialty: str, handoff_targets: List[str]):
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    def agent_node(state: HandoffState) -> dict:
        prompt = (
            f"You are the {name} agent, specializing in {specialty}.\n"
            f"If the task is outside your specialty, say 'HANDOFF: [agent_name]' to hand off.\n"
            f"Available agents: {', '.join(handoff_targets)}\n"
            f"Task: {state['messages'][-1].content}"
        )
        response = llm.invoke(prompt).content
        
        if response.startswith("HANDOFF:"):
            next_agent = response.split(":")[1].strip()
            return {"messages": [AIMessage(content=response)], "current_agent": next_agent}
        return {"messages": [AIMessage(content=f"[{name}] {response}")], "current_agent": END}
    
    return agent_node
```

---

**60. How do you implement a debate / critique pattern in LangGraph?**

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List

class DebateState(TypedDict):
    topic: str
    arguments: Annotated[List[str], lambda a, b: a + b]
    round: int
    conclusion: str

llm = ChatOpenAI(model="gpt-4o-mini")

def proponent(state: DebateState) -> dict:
    arg = llm.invoke(
        f"Topic: {state['topic']}\nPrevious arguments: {state['arguments'][-2:]}\n"
        "Make a compelling argument IN FAVOR."
    ).content
    return {"arguments": [f"PRO: {arg}"], "round": state["round"] + 1}

def opponent(state: DebateState) -> dict:
    arg = llm.invoke(
        f"Topic: {state['topic']}\nPrevious arguments: {state['arguments'][-2:]}\n"
        "Make a compelling argument AGAINST."
    ).content
    return {"arguments": [f"CON: {arg}"]}

def judge(state: DebateState) -> dict:
    verdict = llm.invoke(
        f"Topic: {state['topic']}\nDebate:\n" + "\n".join(state['arguments']) +
        "\nProvide a balanced conclusion."
    ).content
    return {"conclusion": verdict}

def continue_debate(state: DebateState) -> str:
    return "opponent" if state["round"] < 3 else "judge"

graph = StateGraph(DebateState)
graph.add_node("proponent", proponent)
graph.add_node("opponent", opponent)
graph.add_node("judge", judge)
graph.add_edge(START, "proponent")
graph.add_conditional_edges("proponent", continue_debate, {"opponent": "opponent", "judge": "judge"})
graph.add_edge("opponent", "proponent")
graph.add_edge("judge", END)
app = graph.compile()
```

---

---

## Streaming and Async

### Basic

**61. How do you stream LangGraph outputs?**

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"{city}: Sunny 22°C"

agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini", streaming=True), [get_weather])

# stream() yields state updates after each node
for event in agent.stream({"messages": [("user", "Weather in Tokyo and Paris?")]}):
    for node_name, node_output in event.items():
        print(f"\n{'='*20} Node: {node_name} {'='*20}")
        if "messages" in node_output:
            for msg in node_output["messages"]:
                if hasattr(msg, "content") and msg.content:
                    print(f"Content: {msg.content}")
```

---

**62. What are the different streaming modes in LangGraph?**

```python
# 1. stream("updates") — yields state changes after each node (default)
for chunk in app.stream(inputs, stream_mode="updates"):
    print(chunk)  # {"node_name": {state_changes}}

# 2. stream("values") — yields full state after each step
for chunk in app.stream(inputs, stream_mode="values"):
    print(chunk)  # Full state dict

# 3. stream("messages") — yields LLM tokens as they're generated
for chunk in app.stream(inputs, stream_mode="messages"):
    msg, metadata = chunk
    if hasattr(msg, "content") and msg.content:
        print(msg.content, end="", flush=True)  # token-by-token

# 4. Multiple modes at once
for chunk in app.stream(inputs, stream_mode=["updates", "messages"]):
    mode, data = chunk
    print(f"Mode: {mode}, Data: {data}")
```

---

**63. How do you use async execution in LangGraph?**

```python
import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
async def async_search(query: str) -> str:
    """Async search tool."""
    import asyncio
    await asyncio.sleep(0.1)  # simulate async I/O
    return f"Async results for: {query}"

agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), [async_search])

async def run_agent():
    # Async invoke
    result = await agent.ainvoke({"messages": [("user", "Search for AI news")]})
    
    # Async streaming
    async for chunk in agent.astream({"messages": [("user", "Search for ML news")]}):
        print(chunk)

asyncio.run(run_agent())
```

---

**64. How do you implement token-level streaming in LangGraph?**

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessageChunk

agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini", streaming=True), tools=[])

print("Streaming response: ", end="")
for event, metadata in agent.stream(
    {"messages": [("user", "Write a haiku about Python.")]},
    stream_mode="messages"
):
    if isinstance(event, AIMessageChunk) and event.content:
        print(event.content, end="", flush=True)
print()  # newline at end
```

---

### Advanced

**65. How do you stream intermediate agent steps to a frontend?**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import json, asyncio

app_api = FastAPI()
agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini", streaming=True), tools=[])

@app_api.post("/agent/stream")
async def stream_agent(user_message: str):
    async def event_generator():
        async for event in agent.astream(
            {"messages": [("user", user_message)]},
            stream_mode="messages"
        ):
            msg, metadata = event
            if hasattr(msg, "content") and msg.content:
                data = json.dumps({"type": "token", "content": msg.content, "node": metadata.get("langgraph_node", "")})
                yield f"data: {data}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

---

## LangGraph Deployment and Production

### Basic

**66. What is LangGraph Platform?**  
LangGraph Platform (GA since May 2025) is a managed cloud service for deploying, scaling, and monitoring LangGraph agents. It provides REST APIs, persistence, streaming, and a monitoring dashboard. Used in production by Uber, LinkedIn, and Klarna.

```python
# Local development with LangGraph CLI
# langgraph.json config file
{
    "dependencies": ["./"],
    "graphs": {
        "agent": "./agent.py:app"
    },
    "env": ".env"
}

# Run locally: langgraph dev
# Deploy: langgraph deploy (to LangGraph Platform)
```

---

**67. How do you deploy a LangGraph agent with LangServe?**

```python
from fastapi import FastAPI
from langserve import add_routes
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), [search])
app = FastAPI(title="LangGraph Agent API")
add_routes(app, agent, path="/agent")
# Exposes: POST /agent/invoke, /agent/batch, /agent/stream, GET /agent/playground
```

---

**68. How do you configure LangSmith tracing for LangGraph?**

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "my-langgraph-agent"

# All LangGraph invocations are now automatically traced
# View traces at: https://smith.langchain.com/
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[])
result = agent.invoke({"messages": [("user", "Test")]})  # Automatically traced
```

---

**69. How do you handle configuration and secrets in LangGraph production?**

```python
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, Optional

# Pass configuration per-invocation (not hardcoded in graph)
config: RunnableConfig = {
    "configurable": {
        "thread_id": "session_123",
        "model": "gpt-4o-mini",          # model can vary per request
        "user_id": "user_456",
        "max_iterations": 5
    }
}

# Access config inside nodes
def configurable_node(state, config: RunnableConfig) -> dict:
    model_name = config["configurable"].get("model", "gpt-4o-mini")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=model_name)
    return {"messages": [llm.invoke(state["messages"])]}
```

---

### Advanced

**70. How do you implement graceful shutdown and resumable workflows in LangGraph?**

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
import signal

class GracefulShutdown:
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, *args):
        print("Shutdown signal received. Finishing current checkpoint...")
        self.shutdown = True

shutdown = GracefulShutdown()
DB_URI = "postgresql://user:pass@localhost/langgraph"

with ConnectionPool(conninfo=DB_URI) as pool:
    saver = PostgresSaver(pool)
    app = graph.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "long_running_job_1"}}
    
    # The checkpointer ensures state is saved after each node
    # If interrupted, re-run with the same thread_id to resume
    try:
        result = app.invoke(initial_state, config=config)
    except Exception as e:
        print(f"Interrupted: {e}. Workflow state saved. Run again to resume.")
```

---

---

## LangGraph Debugging and Observability

### Basic

**71. How do you debug a LangGraph graph locally?**

```python
# Option 1: Verbose logging via LangSmith
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Option 2: Print state after each step with stream_mode="values"
for state in app.stream(inputs, stream_mode="values"):
    print(f"--- State ---")
    for key, value in state.items():
        print(f"  {key}: {str(value)[:100]}")

# Option 3: Inspect graph structure
print(app.get_graph().print_ascii())

# Option 4: Get Mermaid diagram
print(app.get_graph().draw_mermaid())
```

---

**72. How do you add logging to LangGraph nodes?**

```python
import logging
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def logged_node(state: dict, config: RunnableConfig) -> dict:
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    logger.info(f"[{thread_id}] Node executing with state keys: {list(state.keys())}")
    
    result = do_work(state)
    
    logger.info(f"[{thread_id}] Node completed. Updated keys: {list(result.keys())}")
    return result
```

---

**73. How do you inspect checkpoint history for debugging?**

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "debug_thread"}}

# Run the graph
app.invoke(initial_state, config=config)

# Inspect full checkpoint history
print(f"\nCheckpoint History:")
for i, snapshot in enumerate(app.get_state_history(config)):
    print(f"\nSnapshot {i}:")
    print(f"  Step: {snapshot.metadata.get('step', 'N/A')}")
    print(f"  Writes: {snapshot.metadata.get('writes', {})}")
    print(f"  State keys: {list(snapshot.values.keys())}")
```

---

**74. How do you use LangSmith for LangGraph monitoring?**

```python
import os
from langsmith import Client

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "prod-agent"

# Run agent — automatically traced
result = app.invoke(initial_state, config={"configurable": {"thread_id": "t1"}})

# Query traces via LangSmith SDK
client = Client()
runs = list(client.list_runs(project_name="prod-agent", limit=10))
for run in runs:
    print(f"Run: {run.name} | Latency: {run.end_time - run.start_time} | Status: {run.status}")
```

---

### Advanced

**75. How do you implement custom metrics and alerts for LangGraph in production?**

```python
from langchain_core.callbacks.base import BaseCallbackHandler
from typing import Any, Dict
import time, logging

logger = logging.getLogger("langgraph.metrics")

class ProductionMetricsCallback(BaseCallbackHandler):
    def __init__(self, alert_threshold_s: float = 30.0):
        self.alert_threshold = alert_threshold_s
        self.node_timings = {}

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        run_id = kwargs.get("run_id", "unknown")
        self.node_timings[str(run_id)] = time.perf_counter()

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        run_id = str(kwargs.get("run_id", "unknown"))
        if run_id in self.node_timings:
            latency = time.perf_counter() - self.node_timings.pop(run_id)
            logger.info(f"Node completed in {latency:.3f}s")
            if latency > self.alert_threshold:
                logger.warning(f"SLOW NODE ALERT: {latency:.1f}s exceeds threshold of {self.alert_threshold}s")

    def on_chain_error(self, error: Exception, **kwargs):
        logger.error(f"NODE ERROR: {type(error).__name__}: {error}", exc_info=True)

metrics_callback = ProductionMetricsCallback(alert_threshold_s=10.0)
result = app.invoke(initial_state, config={"callbacks": [metrics_callback]})
```

---

## Best Practices

**76–100. LangGraph Best Practices and Advanced Patterns**

```python
# 76: Keep State minimal and typed
from typing import TypedDict, Optional
# BAD: large untyped state
# GOOD: small, validated state
class MinimalState(TypedDict):
    messages: list
    task_complete: bool  # clear, boolean flag

# 77: Use Pydantic for state validation
from pydantic import BaseModel, validator
class ValidatedState(BaseModel):
    messages: list = []
    iteration: int = 0
    @validator("iteration")
    def iteration_must_be_non_negative(cls, v):
        if v < 0: raise ValueError("iteration must be >= 0")
        return v

# 78: Set recursion limits to prevent infinite loops
result = app.invoke(state, config={"recursion_limit": 25})

# 79: Always use checkpointers in production
from langgraph.checkpoint.postgres import PostgresSaver
# Never rely on InMemorySaver in production — it doesn't survive restarts

# 80: Test nodes independently before graph assembly
def test_my_node():
    state = {"messages": [], "count": 0}
    result = my_node(state)
    assert "messages" in result or "count" in result

# 81: Use conditional edges sparingly — prefer simple linear flows when possible
# Only add branching where the workflow genuinely needs it

# 82: Handle all possible routing function return values
def safe_router(state) -> str:
    intent = classify(state)
    valid_routes = {"search", "calculate", "answer"}
    return intent if intent in valid_routes else "answer"  # always have a default

# 83: Use thread_id consistently for session continuity
config = {"configurable": {"thread_id": f"user_{user_id}_session_{session_id}"}}

# 84: Stream for interactive UX — don't wait for full completion
for chunk in agent.stream(inputs, stream_mode="messages"):
    yield chunk  # send tokens to frontend as they arrive

# 85: Implement circuit breakers for external tool calls
import time
class ToolCircuitBreaker:
    def __init__(self, max_failures: int = 3, reset_s: int = 60):
        self.failures = 0; self.max = max_failures; self.last_fail = 0; self.reset = reset_s
    def can_call(self) -> bool:
        if self.failures >= self.max and time.time() - self.last_fail < self.reset:
            return False
        if time.time() - self.last_fail > self.reset:
            self.failures = 0
        return True
    def record_failure(self): self.failures += 1; self.last_fail = time.time()
    def record_success(self): self.failures = 0

# 86: Version your graphs for safe rollout
# graph_v1.py, graph_v2.py — deploy v2 as a shadow before switching traffic

# 87: Use subgraphs for code organization and reuse
from langgraph.graph import StateGraph
# Each subgraph is independently testable and reusable across parent graphs

# 88: Instrument with LangSmith for every production run
import os; os.environ["LANGCHAIN_TRACING_V2"] = "true"
# 89% of production agent teams use observability tooling (2025 LangChain survey)

# 89: Use .get_state() to inspect, not to re-run
state = app.get_state(config)  # read-only inspection
# Don't invoke the graph just to check state

# 90: Implement timeouts at the graph level for long-running tasks
import asyncio
async def invoke_with_timeout(app, state, config, timeout_s: float = 60):
    try:
        return await asyncio.wait_for(app.ainvoke(state, config), timeout=timeout_s)
    except asyncio.TimeoutError:
        return {"error": f"Timed out after {timeout_s}s", "state": app.get_state(config).values}

# 91: Separate tool definitions from graph definitions
# tools.py → defines tools
# graph.py → imports tools and builds graph
# main.py → configures and runs graph

# 92: Use namespaced thread IDs for multi-tenant systems
def make_thread_id(org_id: str, user_id: str, conversation_id: str) -> str:
    return f"{org_id}:{user_id}:{conversation_id}"

# 93: Implement idempotent nodes — same input → same output
# Nodes should not have side effects that change on re-run
# If a node sends an email, check "already_sent" in state before sending

# 94: Monitor token usage per thread
from langchain_core.callbacks import BaseCallbackHandler
class TokenTracker(BaseCallbackHandler):
    def __init__(self): self.total_tokens = 0
    def on_llm_end(self, response, **kwargs):
        if hasattr(response, "llm_output") and response.llm_output:
            self.total_tokens += response.llm_output.get("token_usage", {}).get("total_tokens", 0)

# 95: Use interrupt_before for any irreversible action
# Email sending, database writes, API mutations → always interrupt and confirm

# 96: Implement graceful degradation with fallback nodes
def route_with_fallback(state) -> str:
    if state.get("primary_api_available"):
        return "primary_handler"
    return "fallback_handler"

# 97: Cache LLM calls when inputs are deterministic
import langchain; from langchain_community.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

# 98: Document your state schema thoroughly
class WellDocumentedState(TypedDict):
    """State for the customer support agent.
    
    Fields:
        messages: Full conversation history (add_messages reducer appends)
        ticket_id: CRM ticket ID, set on first turn, immutable thereafter
        resolved: True when the agent has solved the user's problem
        escalated: True if agent has escalated to human support
    """
    messages: Annotated[list, add_messages]
    ticket_id: Optional[str]
    resolved: bool
    escalated: bool

# 99: Write integration tests for the full graph, not just nodes
def test_full_agent():
    from langgraph.checkpoint.memory import InMemorySaver
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test_1"}}
    result = app.invoke({"messages": [("user", "Test input")]}, config=config)
    assert "messages" in result
    assert len(result["messages"]) > 0

# 100: Use LangGraph Platform for zero-infra production deployment
# langgraph.json + langgraph deploy → auto-scaling, persistence, monitoring
# https://langchain-ai.github.io/langgraph/cloud/
```

---

## Quick Reference: LangChain Deprecation Guide

| **Old (Deprecated)** | **New (Current)** | **Notes** |
|---|---|---|
| `from langchain.llms import OpenAI` | `from langchain_openai import ChatOpenAI` | Use chat models |
| `LLMChain(llm=llm, prompt=p)` | `prompt \| llm \| parser` | LCEL pipe syntax |
| `SimpleSequentialChain` | `chain1 \| (lambda x: ...) \| chain2` | LCEL composition |
| `SequentialChain` | `RunnablePassthrough.assign(...)` | Pass multiple keys |
| `initialize_agent(tools, llm, ...)` | `create_react_agent` or LangGraph | Use prebuilt helpers |
| `ConversationChain` | `RunnableWithMessageHistory` | Explicit history management |
| `ConversationBufferMemory` | `ChatMessageHistory` | Decoupled from chains |
| `ConversationSummaryMemory` | Manual trimming with `trim_messages` | More control |
| `RetrievalQA` | `create_retrieval_chain` | New chain builder |
| `from langchain.embeddings import OpenAIEmbeddings` | `from langchain_openai import OpenAIEmbeddings` | Provider-specific package |
| `from langchain.vectorstores import FAISS` | `from langchain_community.vectorstores import FAISS` | Community package |
| `chain.run(...)` | `chain.invoke(...)` | LCEL standard interface |
| `InMemoryCache` | `from langchain_community.cache import InMemoryCache` | Community package |
| `chain.save("file.json")` | Store config as code/YAML | No built-in equivalent |

---

*Last updated: June 2026 | LangChain ≥1.0 | LangGraph ≥0.4*
