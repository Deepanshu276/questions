# LangChain & LangGraph Interview Questions for AI/ML Roles

> **Syntax Note:** All LangChain examples use **modern LCEL (LangChain Expression Language)** syntax.  
> Deprecated classes like `LLMChain`, `SequentialChain`, `initialize_agent`, and `from langchain.llms import OpenAI` have been replaced with their current equivalents.  
> Key import changes: `langchain_openai`, `langchain_core`, `langchain_community` packages are now separate installs.

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


---

# Part 3: Production RAG Interview Questions (40 Questions)

> 40 deep-dive questions on production-grade Retrieval-Augmented Generation. Answers follow **industry standards** used at companies like Databricks, Confluent, Stripe, Morgan Stanley, and Bloomberg. Every answer includes architecture decisions, failure modes, and real code.

---

## Table of Contents (RAG)
- [RAG Fundamentals](#rag-fundamentals)
- [Hallucination — Causes, Reality, and Mitigation](#hallucination--causes-reality-and-mitigation)
- [Large-Scale RAG — 1M Documents / 100GB+](#large-scale-rag--1m-documents--100gb)
- [Structured & Tabular RAG — FinTech Accuracy](#structured--tabular-rag--fintech-accuracy)
- [Production Architecture & MLOps for RAG](#production-architecture--mlops-for-rag)
- [Retrieval Quality & Evaluation](#retrieval-quality--evaluation)
- [Advanced Retrieval Patterns](#advanced-retrieval-patterns)
- [RAG Security & Compliance](#rag-security--compliance)

---

## RAG Fundamentals

---

**RAG-1. What is Retrieval-Augmented Generation (RAG) and why do production systems use it instead of fine-tuning?**

RAG grounds LLM responses in external, up-to-date knowledge by retrieving relevant documents at inference time and injecting them into the prompt. Fine-tuning bakes knowledge into weights — it is expensive to update, requires retraining for every data change, and does not inherently cite sources.

> **Interview tip:** The key production argument is **freshness + auditability**. RAG lets you update the knowledge base without touching the model. At Morgan Stanley, the wealth management assistant uses RAG over 100,000+ research documents so advisors can cite the source of every claim.

```python
# Canonical production RAG pipeline (LangChain LCEL)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = PGVector.from_existing_index(
    embedding=embeddings,
    collection_name="prod_docs",
    connection_string="postgresql+psycopg2://..."
)
retriever = vectorstore.as_retriever(
    search_type="mmr",            # Maximal Marginal Relevance — reduces redundancy
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
)

SYSTEM = """You are a factual assistant. Answer ONLY using the provided context.
If the context does not contain enough information to answer, say "I don't have enough information."
Never use prior knowledge outside the context.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source','unknown')} | Page: {d.metadata.get('page','')}]\n{d.page_content}"
        for d in docs
    )

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o", temperature=0)  # temperature=0 for factual tasks
    | StrOutputParser()
)
```

---

**RAG-2. What are the five stages of a production RAG pipeline and what can go wrong at each stage?**

A production RAG pipeline has five stages. Each has distinct failure modes that production teams must monitor.

> **Interview tip:** Interviewers at senior levels want you to walk through each stage and name the metric you'd watch. This answer maps directly to what teams at Databricks and Confluent use in production runbooks.

```
Stage 1: INGESTION
  What happens: Documents are loaded, cleaned, chunked, embedded, and stored.
  Failure mode: Bad chunking destroys semantic coherence. A 2,000-token chunk that
  straddles two topics means both topics retrieve irrelevant context.
  Metric to watch: Chunk overlap ratio, embedding latency P99.

Stage 2: INDEXING
  What happens: Embeddings are stored in a vector DB with metadata.
  Failure mode: Stale index — new documents are not re-indexed. Users get outdated answers.
  Metric to watch: Index lag (time between document publish and index availability).

Stage 3: RETRIEVAL
  What happens: Query is embedded, ANN search returns top-k chunks.
  Failure mode: Semantic mismatch — the query embedding and document embedding use
  different models or vocabularies (e.g., internal jargon not in training data).
  Metric to watch: Recall@k (did ground-truth chunks appear in top-k?).

Stage 4: AUGMENTATION (PROMPT CONSTRUCTION)
  What happens: Retrieved chunks are injected into the LLM prompt.
  Failure mode: Context window overflow; irrelevant chunks dilute the signal
  ("lost in the middle" problem — LLMs underweight information in the middle).
  Metric to watch: Context utilization rate, answer faithfulness score.

Stage 5: GENERATION
  What happens: LLM generates a response conditioned on retrieved context.
  Failure mode: Hallucination when the context is insufficient. LLM "fills in"
  from parametric memory instead of saying "I don't know."
  Metric to watch: Faithfulness (RAGAS), answer relevancy, citation accuracy.
```

```python
# Production monitoring hook — attach to each stage
import time, logging
from langchain_core.callbacks import BaseCallbackHandler

class RAGStageMonitor(BaseCallbackHandler):
    def on_retriever_start(self, query, **kwargs):
        self._t = time.perf_counter()
        logging.info(f"[RETRIEVAL START] query={query[:80]}")

    def on_retriever_end(self, documents, **kwargs):
        latency_ms = (time.perf_counter() - self._t) * 1000
        logging.info(f"[RETRIEVAL END] docs_returned={len(documents)} latency={latency_ms:.1f}ms")
        # Emit to Datadog / Prometheus:
        # metrics.histogram("rag.retrieval.latency_ms", latency_ms)
        # metrics.gauge("rag.retrieval.docs_count", len(documents))
```

---

**RAG-3. How do you design a production-grade RAG system from scratch? Walk through the full architecture.**

A production RAG system has an offline pipeline (ingestion) and an online pipeline (serving). They must be independently scalable and observable.

> **Interview tip:** Draw this architecture in system design interviews. The key insight is that ingestion and serving are separate concerns. Name specific tools at each layer.

```
OFFLINE INGESTION PIPELINE:
  [Document Sources: S3 / GCS / SharePoint / Confluence / Databases]
       │
  [Document Loader: Unstructured.io / LangChain loaders → PDF, DOCX, HTML, CSV]
       │
  [Pre-processing: PII scrubbing (Presidio), language detection, table extraction]
       │
  [Chunking: Semantic chunking (preferred) / Recursive character splitter (fallback)]
       │
  [Embedding Model: text-embedding-3-large (OpenAI) / E5-large / BGE-M3 (self-hosted)]
       │
  [Vector Store + Keyword Index + Metadata Store]
   PGVector/Pinecone   Elasticsearch/BM25   PostgreSQL

ONLINE SERVING PIPELINE:
  [User Query]
       │
  [Query Processing: Query rewriting / HyDE / Query classification]
       │
  [Hybrid Retriever: Dense (vector) + Sparse (BM25) → Reciprocal Rank Fusion]
       │
  [Reranker: Cross-encoder (Cohere Rerank / BGE)]
       │
  [Context Assembly: Dedup, truncate, cite sources]
       │
  [LLM Generation: GPT-4o / Claude at temperature=0 with structured output]
       │
  [Response Validation: Faithfulness check → Guardrails]
       │
  [Response + Citations → User]
```

---

**RAG-4. What chunking strategies exist and which should you use in production?**

Chunking is one of the highest-impact decisions in RAG. The wrong strategy degrades retrieval quality more than almost any other factor.

> **Interview tip:** Most candidates say "fixed-size chunks with overlap." Senior engineers know the five strategies and can articulate when each applies. Semantic chunking and proposition-level chunking are the production standards at leading AI companies.

```python
# Strategy 1: Fixed-size (baseline — avoid in production for prose)
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

# Strategy 2: Semantic chunking (PREFERRED for prose documents)
# Groups sentences by embedding similarity — keeps semantic units together
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85
)

# Strategy 3: Proposition-level chunking (BEST retrieval quality, highest cost)
# Each chunk = one atomic fact. Ask an LLM to decompose documents into propositions.
PROP_PROMPT = """Decompose the following text into atomic propositions.
Each proposition should be a single, self-contained factual statement.
Return as a JSON list of strings.
Text: {text}"""
# Then embed each proposition separately — dense, high-precision retrieval.

# Strategy 4: Parent-child chunking (production standard for long docs)
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma

child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
store = InMemoryStore()  # use Redis in production
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
# Small chunks for precision retrieval, large parent chunks returned for full context

# Strategy 5: Document-structure-aware chunking (PDFs/markdown with headers)
from langchain_text_splitters import MarkdownHeaderTextSplitter
headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# Each chunk retains header metadata → better filtering and context
```

---

**RAG-5. What embedding models should you use in production and how do you choose?**

Embedding model choice affects retrieval quality, cost, latency, and data privacy. There is no universally best model — the right choice depends on your domain, data privacy requirements, and scale.

> **Interview tip:** Cite the MTEB (Massive Text Embedding Benchmark) leaderboard. For domain-specific data (legal, medical, financial), fine-tuning an open-source model often beats generic models.

```python
# Option 1: OpenAI (best general quality, requires API)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # 3072 dims, best quality
    # model="text-embedding-3-small"  # 1536 dims, 5x cheaper, ~95% of quality
)

# Option 2: Self-hosted (data privacy, fixed cost at scale)
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 128}
)

# Option 3: Cohere (strong multilingual support)
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# Production decision matrix:
# General SaaS          → text-embedding-3-large   (best MTEB, API cost tradeoff)
# Financial/legal data  → Fine-tuned BGE            (domain fit, self-hosted)
# GDPR / on-prem        → BGE / E5 local            (no data egress, GPU needed)
# Multilingual          → multilingual-e5-large     (100+ languages)
# High scale, cost-opt  → text-embedding-3-small    (5x cheaper, ~95% quality)
```

---

## Hallucination — Causes, Reality, and Mitigation

---

**RAG-6. Can RAG achieve zero hallucination? If not, why not — and what are the exact production steps to minimize it?**

No. Zero hallucination is not achievable in any production RAG system with a generative LLM. Understanding why — and the precise mitigation layers — is critical for senior roles.

> **Interview tip:** This is a trap question for junior candidates who say "yes, RAG prevents hallucination." The correct answer names three root causes and at least five concrete mitigations. Companies like Glean, Perplexity, and Vectara have published their multi-layer approaches.

**Why zero hallucination is impossible:**

```
Root Cause 1: RETRIEVAL FAILURE
  The most relevant document is not in the index, or the query fails to retrieve it.
  The LLM, having no grounding context, falls back on parametric memory.
  Example: User asks about a policy updated yesterday; the index hasn't re-ingested it.

Root Cause 2: CONTEXT INSUFFICIENT / AMBIGUOUS
  Retrieved chunks exist but don't fully answer the question.
  The LLM "helpfully" extrapolates beyond what the context supports.
  Example: Context says "Revenue was $2.1B in Q3." User asks "Was Q3 better than Q2?"
  LLM may infer yes/no without Q2 data being retrieved.

Root Cause 3: LLM PARAMETRIC MEMORY BLEED
  Even with perfect context, LLMs blend retrieved facts with training-time knowledge.
  Particularly dangerous for numbers, dates, and proper nouns.
  Example: Context says CEO is Alice. LLM "knows" from training it was Bob → blends both.

Root Cause 4: PROMPT INJECTION / CONTEXT POISONING
  Malicious content in the document corpus instructs the LLM to ignore retrieved context.
```

**Production mitigation — the 7-layer approach:**

```python
# LAYER 1: HYBRID SEARCH — catch what pure vector search misses
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25 = BM25Retriever.from_documents(docs, k=10)
dense = vectorstore.as_retriever(search_kwargs={"k": 10})
hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.3, 0.7])

# LAYER 2: RERANKING — score relevance with cross-encoder, not just cosine similarity
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

reranker = CohereRerank(model="rerank-english-v3.0", top_n=4)
retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid)

# LAYER 3: CONFIDENCE GATE — refuse to answer if context isn't relevant enough
def retrieve_with_confidence_gate(query: str, threshold: float = 0.75):
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=6)
    top_score = docs_and_scores[0][1] if docs_and_scores else 0.0
    if top_score < threshold:
        return None, "I don't have reliable information to answer this question."
    return [d for d, _ in docs_and_scores], None

# LAYER 4: STRICT SYSTEM PROMPT — constrain LLM to context only
ANTI_HALLUCINATION_SYSTEM = """You are a precise factual assistant.

STRICT RULES:
1. Answer ONLY using information explicitly stated in the provided context.
2. If the context does not contain enough information, say exactly:
   "The provided documents do not contain enough information to answer this question."
3. Do NOT use any knowledge from your training data.
4. Do NOT make inferences, extrapolations, or educated guesses.
5. Always cite the source document for each factual claim.
6. Numbers, dates, names in your answer MUST appear verbatim in the context.

Context:
{context}"""

# LAYER 5: STRUCTURED OUTPUT WITH CITATIONS
from pydantic import BaseModel, Field
from typing import List

class Citation(BaseModel):
    source_id: str = Field(description="The source document ID or filename")
    quote: str = Field(description="The exact quote from the source supporting this claim")

class GroundedAnswer(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: str = Field(description="high / medium / low")
    unanswered_aspects: List[str] = Field(default_factory=list)

structured_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(GroundedAnswer)

# LAYER 6: FAITHFULNESS CHECK (LLM-as-judge)
FAITHFULNESS_PROMPT = """You are a faithfulness judge. Identify any claims in the answer
NOT supported by the context.

Context: {context}
Answer: {answer}

Return JSON: {{"faithful": true/false, "unsupported_claims": ["claim1", ...]}}"""

async def check_faithfulness(context: str, answer: str, llm) -> dict:
    result = await llm.ainvoke(FAITHFULNESS_PROMPT.format(context=context, answer=answer))
    import json
    return json.loads(result.content)

# LAYER 7: MONITORING — track hallucination rate over time with RAGAS
# Run nightly faithfulness evaluation batch. Alert if score drops below 0.85.
# Log every (query, context, answer) tuple for audit and debugging.

# The honest answer: with all 7 layers, production systems achieve 95-99% faithfulness.
# Zero is not the target — a measurable, monitored, continuously improving rate is.
```

---

**RAG-7. What is the "lost in the middle" problem and how do you fix it?**

LLMs pay disproportionately more attention to content at the beginning and end of the context window than to content in the middle. When you stuff 20 retrieved chunks into a prompt, the most relevant chunk may be in the middle — and the LLM effectively ignores it.

> **Interview tip:** This was formally studied in the paper "Lost in the Middle" (Liu et al., 2023). The fix is reranking + context reordering, not just retrieving more chunks.

```python
from langchain_community.document_transformers import LongContextReorder

reorder = LongContextReorder()

def build_context_with_reorder(docs):
    # After reranking, reorder so top docs are at position 0 and -1 (start and end)
    reordered = reorder.transform_documents(docs)
    return "\n\n".join(d.page_content for d in reordered)

# Fix 2: Limit context to fewer, higher-quality chunks (k=4, not k=20)
# More context ≠ better answers. After reranking, pass only top 4 chunks.

# Fix 3: Use models with better long-context attention
# GPT-4o and Claude 3.5 handle this better than GPT-3.5 or older models.

# Fix 4: Map-reduce for very long document sets — process each chunk independently,
# then synthesize — avoids the problem entirely.
```

---

**RAG-8. What is HyDE (Hypothetical Document Embedding) and when should you use it?**

HyDE generates a hypothetical answer to the query, embeds that answer, and uses it as the retrieval query instead of the original question. It bridges the gap between question embeddings and answer embeddings, which often live in different parts of the embedding space.

> **Interview tip:** HyDE is particularly effective for technical or domain-specific queries where user vocabulary differs from document vocabulary (e.g., plain English query vs legal/medical/financial jargon). Tradeoff: adds one LLM call per retrieval.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

hyde_prompt = ChatPromptTemplate.from_template(
    "Write a short passage (3-5 sentences) that would answer the following question. "
    "Write it as if it were from a technical document.\n\nQuestion: {question}"
)
generate_hypothesis = hyde_prompt | llm | StrOutputParser()

def hyde_retrieve(query: str, vectorstore, k: int = 6):
    hypothesis = generate_hypothesis.invoke({"question": query})
    hypothesis_embedding = embeddings.embed_query(hypothesis)
    docs = vectorstore.similarity_search_by_vector(hypothesis_embedding, k=k)
    return docs

# When to use HyDE:
# ✅ Domain-specific documents (legal, medical, financial)
# ✅ Queries in plain language, documents in technical language
# ✅ When recall@k is low with direct query embedding
# ❌ Real-time / low-latency requirements (adds ~200-500ms)
# ❌ When query vocabulary already matches document vocabulary
```

---

**RAG-9. What is query rewriting and how does it improve RAG accuracy?**

A user's raw query is often ambiguous, uses pronouns referring to prior conversation, or is phrased in a way that doesn't match document vocabulary. Query rewriting transforms the query before retrieval to maximize recall.

> **Interview tip:** Three production patterns — standalone question generation (multi-turn), multi-query expansion (ambiguous queries), and step-back prompting (abstract questions). Name all three.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Pattern 1: Standalone question (multi-turn conversations)
STANDALONE_PROMPT = ChatPromptTemplate.from_template(
    """Given conversation history and a follow-up question, rewrite as a standalone
    question containing all necessary context.
    
    History: {chat_history}
    Follow-up: {question}
    Standalone question:"""
)
standalone_chain = STANDALONE_PROMPT | llm | StrOutputParser()

# Pattern 2: Multi-query expansion
# Generates 3-5 different phrasings — union of results dramatically improves recall
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    llm=llm,
    include_original=True
)

# Pattern 3: Step-back prompting (for abstract / high-level questions)
STEPBACK_PROMPT = ChatPromptTemplate.from_template(
    """Generate a more general "step-back" question that retrieves broader context
    needed to answer the specific question.
    
    Specific question: {question}
    Step-back question:"""
)
# Example:
# Specific: "What was Apple's iPhone 15 revenue in Q4 2023?"
# Step-back: "What were Apple's iPhone revenue trends in 2023?"
# Retrieves broader context that contains the specific answer
```

---

## Large-Scale RAG — 1M Documents / 100GB+

---

**RAG-10. How do you build a RAG system over 1 million documents or 100GB of data?**

At scale, three things break in naive RAG: ingestion speed, index size and query latency, and cost. Each requires a specific architectural response.

> **Interview tip:** This is a senior/staff-level question. The answer has three dimensions: offline ingestion at scale, vector DB selection and sharding, and query-time optimizations. Name specific tools — Ray for parallelism, Pinecone/Weaviate for hosting, HNSW vs IVF-PQ for indexing.

```python
# CHALLENGE 1: INGESTION — processing 100GB in reasonable time
# 1M docs × 100ms/doc = ~28 hours single-threaded
# 1M docs × 16 Ray workers = ~1.75 hours

import ray

@ray.remote
def process_document_batch(doc_paths: list, embedding_model: str) -> list:
    """Process a batch of documents: load → chunk → embed → return vectors"""
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.document_loaders import UnstructuredFileLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    embedder = OpenAIEmbeddings(model=embedding_model)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    results = []
    
    for path in doc_paths:
        try:
            docs = UnstructuredFileLoader(path).load()
            chunks = splitter.split_documents(docs)
            texts = [c.page_content for c in chunks]
            vectors = embedder.embed_documents(texts)
            results.extend(zip(texts, vectors, [c.metadata for c in chunks]))
        except Exception as e:
            print(f"Failed {path}: {e}")  # Don't let one bad doc fail the batch
    return results

def ingest_at_scale(doc_paths: list, batch_size: int = 100):
    batches = [doc_paths[i:i+batch_size] for i in range(0, len(doc_paths), batch_size)]
    futures = [process_document_batch.remote(b, "text-embedding-3-large") for b in batches]
    all_results = ray.get(futures)
    return [item for batch in all_results for item in batch]

# CHALLENGE 2: VECTOR DB SELECTION
# 5M vectors (1M docs × ~5 chunks each, 1536 dims):
# Raw vectors: 5M × 1536 × 4 bytes ≈ 30GB
# Plus HNSW graph + metadata: ~60-100GB total
#
# DB selection guide:
# Pinecone   → 10M-1B vecs  | Managed, serverless, auto-sharding
# Weaviate   → 10M-100M     | Self-host, hybrid search native
# Qdrant     → 1M-100M      | Best OSS performance, Rust-based
# PGVector   → <5M vecs     | Good if already on Postgres
# Milvus     → 100M-10B     | Petabyte scale, complex setup

# CHALLENGE 3: INDEX TYPE — IVF-PQ reduces memory 10-20x
# HNSW (default): ~80GB RAM for 5M vectors — fine for <5M
# IVF-PQ: ~4GB RAM for 5M vectors — required at 100M+ scale
# Tradeoff: IVF-PQ has ~5% lower recall vs HNSW

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff, QuantizationConfig, ScalarQuantizationConfig

client = QdrantClient("localhost", port=6333)
client.create_collection(
    collection_name="large_corpus",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    quantization_config=QuantizationConfig(
        scalar=ScalarQuantizationConfig(type="int8", quantile=0.99, always_ram=True)
    )
)

# CHALLENGE 4: QUERY LATENCY — keep P99 < 200ms at scale
# Technique 1: Pre-filter by metadata before ANN search (reduces search space 99%)
search_result = client.search(
    collection_name="large_corpus",
    query_vector=query_embedding,
    query_filter={"must": [
        {"key": "department", "match": {"value": "finance"}},
        {"key": "year", "range": {"gte": 2023}}
    ]},
    limit=10
)

# Technique 2: Semantic cache (see RAG-17 for full implementation)
# Technique 3: Async batch embedding for ingestion pipelines
```

---

**RAG-11. How do you handle incremental updates to a large document corpus in production?**

In a 1M document corpus, documents are added, updated, and deleted continuously. You cannot re-embed the entire corpus on every change. Production systems use event-driven incremental indexing.

> **Interview tip:** The key concept is **document-level versioning + change detection**. The production pattern is a CDC (change data capture) pipeline feeding an async ingestion queue. Most junior candidates describe batch reingestion — distinguish yourself by describing the event-driven approach.

```python
# Production architecture:
#
# [Document CMS / Database]
#   │ (CDC events: INSERT / UPDATE / DELETE — e.g., Debezium → Kafka)
#   ▼
# [Message Queue — Kafka / SQS / Pub/Sub]
#   │
#   ▼
# [Ingestion Worker — consumes events]
#   INSERT: embed new chunks, upsert to vector DB
#   UPDATE: delete old chunks by doc_id, embed new chunks, upsert
#   DELETE: delete all chunks by doc_id from vector DB

import hashlib

def compute_doc_hash(content: str) -> str:
    """Hash document content — only re-embed if content actually changed."""
    return hashlib.sha256(content.encode()).hexdigest()

class IncrementalIndexer:
    def __init__(self, vectorstore, hash_store):  # hash_store = Redis
        self.vectorstore = vectorstore
        self.hash_store = hash_store

    def upsert_document(self, doc_id: str, content: str, metadata: dict):
        new_hash = compute_doc_hash(content)
        existing_hash = self.hash_store.get(f"doc_hash:{doc_id}")

        if existing_hash and existing_hash.decode() == new_hash:
            return  # Content unchanged — skip re-embedding (saves cost)

        # Delete old vectors for this document
        self.vectorstore.delete(filter={"doc_id": doc_id})

        # Chunk, embed, insert new vectors with doc_id in metadata
        # ... chunking and embedding logic here ...
        self.hash_store.set(f"doc_hash:{doc_id}", new_hash)

    def delete_document(self, doc_id: str):
        self.vectorstore.delete(filter={"doc_id": doc_id})
        self.hash_store.delete(f"doc_hash:{doc_id}")
```

---

**RAG-12. How do you evaluate retrieval quality at scale and what metrics do you track in production?**

Retrieval quality is the most important determinant of RAG answer quality. Yet most teams only measure end-to-end answer quality. Production systems measure retrieval independently.

> **Interview tip:** Name specific metrics: Recall@k, Precision@k, MRR, NDCG. Know how to compute them offline using a golden dataset. Know that retrieval can fail silently — end-to-end answer quality metrics often don't catch retrieval failures.

```python
import numpy as np
from typing import List, Dict

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """What fraction of relevant docs appear in top-k retrieved?"""
    retrieved_top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(retrieved_top_k & relevant) / len(relevant)

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Of the top-k retrieved, what fraction are actually relevant?"""
    retrieved_top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    return sum(1 for d in retrieved_top_k if d in relevant) / k

def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Rank of the first relevant document in the retrieved list."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def evaluate_retriever(retriever, golden_dataset: List[Dict]) -> Dict:
    recall_scores, precision_scores, mrr_scores = [], [], []
    for item in golden_dataset:
        retrieved_docs = retriever.invoke(item["query"])
        retrieved_ids = [d.metadata.get("doc_id") for d in retrieved_docs]
        recall_scores.append(recall_at_k(retrieved_ids, item["relevant_doc_ids"], k=5))
        precision_scores.append(precision_at_k(retrieved_ids, item["relevant_doc_ids"], k=5))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, item["relevant_doc_ids"]))
    return {
        "recall@5": np.mean(recall_scores),
        "precision@5": np.mean(precision_scores),
        "mrr": np.mean(mrr_scores)
    }

# Production benchmark targets:
# Recall@5       > 0.85   (most relevant chunk in top 5)
# Precision@5    > 0.50   (at least half of top 5 are useful)
# MRR            > 0.70   (first relevant chunk is near top)
# Faithfulness   > 0.90   (LLM doesn't hallucinate from context)
# Answer Rel.    > 0.80   (answer addresses the question)
```

---

## Structured & Tabular RAG — FinTech Accuracy

---

**RAG-13. How do you build a production-grade RAG system over tabular data where precision is critical — for example in a FinTech company?**

Standard vector similarity search over prose chunks fails for tabular/structured data. Numbers, dates, and categorical values need exact matching, not semantic similarity. A financial analyst asking "What was EBITDA margin in Q3 2024?" needs the exact number, not a semantically similar paragraph.

> **Interview tip:** This is one of the most important production RAG questions. The correct answer is a **hybrid architecture**: Text-to-SQL for structured queries + vector RAG for unstructured narrative + a classifier/router to decide which path handles each query. Companies like Stripe, Bloomberg, and Palantir use this pattern.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from pydantic import BaseModel
from enum import Enum

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Step 1: Route the query — structured (SQL) vs unstructured (RAG)
class QueryType(str, Enum):
    SQL = "sql"          # Numbers, aggregations, filters, comparisons
    RAG = "rag"          # Definitions, policies, narrative analysis
    HYBRID = "hybrid"    # Needs both (e.g., "explain the Q3 revenue drop")

class QueryRoute(BaseModel):
    query_type: QueryType
    reasoning: str

ROUTER_PROMPT = """Classify this financial query:
- SQL: asks for specific numbers, calculations, time-series, comparisons (revenue, EBITDA, ratios)
- RAG: asks for definitions, policies, qualitative analysis, document content
- HYBRID: needs both structured data AND narrative context

Query: {query}"""

structured_router_llm = llm.with_structured_output(QueryRoute)

# Step 2: SQL path — natural language → SQL → execute → verify
db = SQLDatabase.from_uri(
    "postgresql://user:pass@host/fintech_db",
    include_tables=["revenue", "expenses", "balance_sheet", "transactions"],
    sample_rows_in_table_info=2   # show LLM the schema with sample rows
)
sql_chain = create_sql_query_chain(llm, db)

# Production SQL safety: never allow write operations
SAFE_SQL_PREFIXES = ("SELECT", "WITH")
def execute_safe_sql(query: str, db: SQLDatabase) -> str:
    clean_query = query.strip().upper()
    if not any(clean_query.startswith(p) for p in SAFE_SQL_PREFIXES):
        raise ValueError(f"Non-SELECT query blocked: {query[:100]}")
    return db.run(query)

# Step 3: Full hybrid pipeline
async def financial_rag_query(user_query: str) -> dict:
    route = await structured_router_llm.ainvoke(ROUTER_PROMPT.format(query=user_query))

    if route.query_type == QueryType.SQL:
        sql = await sql_chain.ainvoke({"question": user_query})
        result = execute_safe_sql(sql, db)
        # Format the raw SQL result into a natural language answer
        formatted = await llm.ainvoke(
            f"SQL result: {result}\n\nUser question: {user_query}\n\n"
            f"Provide a clear, precise answer. State the exact numbers."
        )
        return {"answer": formatted.content, "sql": sql, "raw_result": result}

    elif route.query_type == QueryType.RAG:
        docs = retriever.invoke(user_query)
        context = "\n".join(d.page_content for d in docs)
        answer = await llm.ainvoke(f"Context: {context}\n\nQuestion: {user_query}")
        return {"answer": answer.content, "sources": [d.metadata for d in docs]}

    else:  # HYBRID
        import asyncio
        sql_task = asyncio.create_task(
            sql_chain.ainvoke({"question": user_query})
        )
        rag_task = asyncio.create_task(retriever.ainvoke(user_query))
        sql_result, rag_docs = await asyncio.gather(sql_task, rag_task)
        sql_data = execute_safe_sql(sql_result, db)
        rag_context = "\n".join(d.page_content for d in rag_docs)
        combined = await llm.ainvoke(
            f"Structured data:\n{sql_data}\n\nNarrative context:\n{rag_context}\n\n"
            f"Question: {user_query}\n\nProvide a comprehensive answer citing both sources."
        )
        return {"answer": combined.content, "sql": sql_result, "sources": [d.metadata for d in rag_docs]}
```

---

**RAG-14. What are the unique challenges of RAG over financial tables and how do you handle each?**

Financial tables have four unique properties that break standard RAG: hierarchical structure (header rows, merged cells), implicit context (column headers matter as much as cell values), temporal indexing (same metric at different time periods), and precision requirements (a 0.1% error in a financial ratio matters).

> **Interview tip:** Candidates who've worked on FinTech RAG will mention the table serialization problem. How you convert a table to text determines retrieval quality more than any other single factor for tabular data.

```python
import pandas as pd
from langchain_core.documents import Document

# CHALLENGE 1: TABLE SERIALIZATION
# Naive: df.to_string() → loses structure, LLM struggles to parse
# Production: row-as-sentence serialization that preserves column semantics

def serialize_financial_table(df: pd.DataFrame, table_name: str, period: str) -> str:
    lines = [f"TABLE: {table_name} | PERIOD: {period}"]
    lines.append(f"Columns: {', '.join(df.columns.tolist())}")
    lines.append("")
    for _, row in df.iterrows():
        parts = []
        for col, val in row.items():
            if pd.notna(val):
                parts.append(f"{col}: {val}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)

# Output example:
# TABLE: Income Statement | PERIOD: Q3 2024
# Metric: Revenue | Q3_2024: $2.1B | Q3_2023: $1.8B | YoY_Change: +16.7%
# Metric: EBITDA | Q3_2024: $420M | Q3_2023: $360M | YoY_Change: +16.7%

# CHALLENGE 2: METADATA TAGGING — enable precision filtering
def create_financial_chunks(df: pd.DataFrame, metadata: dict) -> list[Document]:
    """One Document per row for maximum retrieval precision."""
    docs = []
    for _, row in df.iterrows():
        content = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
        doc_metadata = {
            **metadata,
            "metric": row.get("Metric", ""),
            "period": row.get("Period", ""),
            "value": str(row.get("Value", "")),
        }
        docs.append(Document(page_content=content, metadata=doc_metadata))
    return docs

# CHALLENGE 3: NUMERIC PRECISION — validate numbers with structured output
from pydantic import BaseModel, validator

class FinancialAnswer(BaseModel):
    metric_name: str
    value: float
    unit: str           # "USD millions", "percent", "ratio"
    period: str         # "Q3 2024"
    source_table: str
    exact_quote: str    # The exact string from the source — forces grounding

# CHALLENGE 4: CALCULATION QUERIES — use code execution, not LLM arithmetic
# Never trust LLMs to do arithmetic on financial data.
from langchain_core.tools import tool

@tool
def calculate_financial_ratio(numerator: float, denominator: float, ratio_name: str) -> str:
    """Safely calculate a financial ratio using code, not LLM inference."""
    if denominator == 0:
        return f"Cannot compute {ratio_name}: denominator is zero"
    result = numerator / denominator
    return f"{ratio_name} = {numerator} / {denominator} = {result:.4f}"

# Example: "What is the P/E ratio?"
# → SQL retrieves: price=$150, EPS=$5
# → Tool computes: 150 / 5 = 30.0 (code, not LLM)
# → LLM formats: "The P/E ratio is 30.0x (price $150 / EPS $5.00)"
```

---

**RAG-15. How do you build a RAG system over mixed data sources — PDFs, databases, spreadsheets, and APIs — in a FinTech context?**

Enterprise FinTech data lives in many places: regulatory filings as PDFs, transaction data in Postgres, models in Excel, and market data from APIs. A production system must federate these sources transparently.

> **Interview tip:** The key pattern is the **retrieval federation layer** — a unified interface that dispatches queries to the right data source. This is architecturally similar to a data mesh at the retrieval layer.

```python
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List

class DataSourceRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, metadata_filter: dict = None) -> List[Document]:
        pass
    
    @abstractmethod
    def handles(self, query: str) -> bool:
        """Return True if this retriever is appropriate for this query."""
        pass

class PDFDocumentRetriever(DataSourceRetriever):
    """For regulatory filings, research reports, policy documents."""
    def __init__(self, vectorstore):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    async def retrieve(self, query, metadata_filter=None):
        return await self.retriever.ainvoke(query)
    
    def handles(self, query):
        keywords = ["policy", "regulation", "filing", "report", "clause", "section"]
        return any(k in query.lower() for k in keywords)

class SQLDatabaseRetriever(DataSourceRetriever):
    """For transaction data, account balances, time-series financial data."""
    def __init__(self, db, llm):
        self.chain = create_sql_query_chain(llm, db)
        self.db = db
    
    async def retrieve(self, query, metadata_filter=None):
        sql = await self.chain.ainvoke({"question": query})
        result = self.db.run(sql)
        return [Document(page_content=result, metadata={"source": "sql", "query": sql})]
    
    def handles(self, query):
        keywords = ["revenue", "balance", "transaction", "amount", "total", "average", "count"]
        return any(k in query.lower() for k in keywords)

class FederatedRetriever:
    """Routes queries to the appropriate data source(s) and merges results."""
    def __init__(self, retrievers: List[DataSourceRetriever]):
        self.retrievers = retrievers
    
    async def retrieve(self, query: str) -> List[Document]:
        import asyncio
        capable = [r for r in self.retrievers if r.handles(query)]
        if not capable:
            capable = self.retrievers  # fallback: try all
        
        results = await asyncio.gather(*[r.retrieve(query) for r in capable], return_exceptions=True)
        all_docs = []
        for result in results:
            if isinstance(result, list):
                all_docs.extend(result)
        return all_docs
```

---

## Production Architecture & MLOps for RAG

---

**RAG-16. How do you implement a production-grade RAG system with proper observability and monitoring?**

A production RAG system without observability is flying blind. Production monitoring covers four planes: infrastructure (latency, uptime), retrieval quality (recall, precision), generation quality (faithfulness, relevancy), and business outcomes (user satisfaction, task completion rate).

> **Interview tip:** LangSmith is the most common answer, but senior engineers go further. They describe custom dashboards, alerting thresholds, and the distinction between online metrics (available in real-time) and offline metrics (computed in batch over a golden dataset).

```python
import time, uuid, logging
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

@dataclass
class RAGTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    query: str = ""
    rewritten_query: Optional[str] = None
    retrieved_doc_ids: List[str] = field(default_factory=list)
    retrieved_doc_scores: List[float] = field(default_factory=list)
    context_length_tokens: int = 0
    answer: str = ""
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    user_feedback: Optional[str] = None   # "thumbs_up" / "thumbs_down"
    error: Optional[str] = None

class ObservableRAGPipeline:
    def __init__(self, retriever, llm, tracer_backend):
        self.retriever = retriever
        self.llm = llm
        self.tracer = tracer_backend

    async def query(self, user_query: str) -> dict:
        trace = RAGTrace(query=user_query)
        t_start = time.perf_counter()
        try:
            t_retr = time.perf_counter()
            docs = await self.retriever.ainvoke(user_query)
            trace.retrieval_latency_ms = (time.perf_counter() - t_retr) * 1000
            trace.retrieved_doc_ids = [d.metadata.get("doc_id", "") for d in docs]

            context = "\n\n".join(d.page_content for d in docs)
            t_gen = time.perf_counter()
            response = await self.llm.ainvoke(
                f"Context:\n{context}\n\nQuestion: {user_query}"
            )
            trace.generation_latency_ms = (time.perf_counter() - t_gen) * 1000
            trace.answer = response.content

        except Exception as e:
            trace.error = str(e)
            raise
        finally:
            trace.total_latency_ms = (time.perf_counter() - t_start) * 1000
            await self.tracer.log(trace)
            # Emit to Prometheus/Datadog:
            # metrics.histogram("rag.latency.total_ms", trace.total_latency_ms)
            # metrics.histogram("rag.latency.retrieval_ms", trace.retrieval_latency_ms)

        return {"answer": trace.answer, "trace_id": trace.trace_id}

# Production alerting thresholds:
ALERT_THRESHOLDS = {
    "p99_latency_ms": 3000,         # Alert if P99 > 3 seconds
    "faithfulness_score": 0.85,     # Alert if daily avg faithfulness < 0.85
    "error_rate": 0.01,             # Alert if error rate > 1%
    "retrieval_recall_at_5": 0.80   # Alert if recall drops below 0.80
}
```

---

**RAG-17. How do you implement semantic caching in a production RAG system?**

Semantic caching stores results of previous RAG queries and returns cached answers when a new query is semantically similar. This reduces latency from ~2s to ~50ms and cuts LLM costs by 30–60% in production.

> **Interview tip:** Semantic caching is different from exact-match caching. The threshold is critical — too tight misses paraphrases, too loose returns wrong cached answers. For FinTech, use threshold ≥ 0.95. For general assistants, 0.92 is common.

```python
import json, hashlib
import numpy as np
from redis import Redis
from langchain_openai import OpenAIEmbeddings
from typing import Optional
from datetime import datetime

class SemanticCache:
    def __init__(
        self,
        redis_client: Redis,
        embedder: OpenAIEmbeddings,
        similarity_threshold: float = 0.95,  # Very tight for FinTech
        ttl_seconds: int = 3600
    ):
        self.redis = redis_client
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds

    def _cosine_similarity(self, a: list, b: list) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def get(self, query: str) -> Optional[str]:
        query_embedding = self.embedder.embed_query(query)
        cache_keys = self.redis.keys("semantic_cache:*")
        
        for key in cache_keys:
            entry = json.loads(self.redis.get(key))
            similarity = self._cosine_similarity(query_embedding, entry["embedding"])
            if similarity >= self.threshold:
                return entry["answer"]
        return None

    async def set(self, query: str, answer: str):
        query_embedding = self.embedder.embed_query(query)
        cache_key = f"semantic_cache:{hashlib.sha256(query.encode()).hexdigest()}"
        entry = {
            "query": query,
            "embedding": query_embedding,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.redis.setex(cache_key, self.ttl, json.dumps(entry))

async def rag_with_semantic_cache(query: str, cache: SemanticCache, rag_chain) -> str:
    cached_answer = await cache.get(query)
    if cached_answer:
        return cached_answer   # ~50ms vs ~2000ms
    answer = await rag_chain.ainvoke(query)
    await cache.set(query, answer)
    return answer
```

---

**RAG-18. How do you version and manage your RAG knowledge base in production?**

A RAG knowledge base is a live artifact. Documents get updated, embedding models get upgraded, and chunking strategies change. Without versioning, you cannot roll back a bad ingestion, A/B test retrieval strategies, or audit which document version was used for a past answer.

> **Interview tip:** Treat the vector index like a model artifact — version it, test before promoting, support rollback. Use blue/green deployment for major index updates. Most candidates miss this entirely.

```python
from dataclasses import dataclass

@dataclass
class IndexVersion:
    version_id: str
    embedding_model: str
    chunking_strategy: str
    doc_count: int
    vector_count: int
    created_at: str
    status: str              # "staging" | "active" | "deprecated"
    eval_scores: dict        # {"recall@5": 0.87, "faithfulness": 0.91}

class KnowledgeBaseVersionManager:
    def __init__(self, metadata_store):
        self.metadata = metadata_store

    def promote_to_active(self, version_id: str, min_recall: float = 0.85):
        """Promote a staging index only if it meets quality thresholds."""
        version = self.metadata.get_version(version_id)
        
        if version.status != "staging":
            raise ValueError(f"Can only promote staging versions, got: {version.status}")
        
        recall = version.eval_scores.get("recall@5", 0)
        if recall < min_recall:
            raise ValueError(
                f"Version {version_id} has recall@5={recall:.3f} < threshold {min_recall}. "
                f"Deployment blocked."
            )
        
        # Deprecate current active
        current_active = self.metadata.get_active_version()
        if current_active:
            self.metadata.update_status(current_active.version_id, "deprecated")
        
        self.metadata.update_status(version_id, "active")

    def rollback(self):
        """Roll back to the most recently deprecated version."""
        deprecated = self.metadata.get_most_recent_deprecated()
        if not deprecated:
            raise RuntimeError("No deprecated version available for rollback")
        current_active = self.metadata.get_active_version()
        self.metadata.update_status(current_active.version_id, "deprecated")
        self.metadata.update_status(deprecated.version_id, "active")
```

---

**RAG-19. How do you use RAGAS to evaluate a RAG pipeline in production?**

RAGAS is the standard evaluation framework for RAG. It measures four dimensions: faithfulness, answer relevancy, context precision, and context recall — all using an LLM as judge, making it scalable to thousands of queries.

> **Interview tip:** Know all four metrics and what a bad score tells you about which component to fix. Most candidates know faithfulness but not context precision vs context recall — distinguish yourself.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,       # Does the answer contain only info from the context?
    answer_relevancy,   # Does the answer address the question asked?
    context_precision,  # Are the retrieved chunks actually useful?
    context_recall,     # Did retrieval find all relevant information?
)
from datasets import Dataset

eval_data = {
    "question": ["What is the company's revenue growth target?"],
    "answer": ["The company targets 15% YoY revenue growth."],
    "contexts": [["[chunk] 2024 Strategic Plan... 15% annual revenue growth target..."]],
    "ground_truth": ["The company's policy is 15% annual revenue growth."]
}

dataset = Dataset.from_dict(eval_data)
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

# Interpreting results:
# faithfulness < 0.85     → LLM is hallucinating; strengthen system prompt or add guardrails
# answer_relevancy < 0.80 → LLM is going off-topic; check prompt or query routing
# context_precision < 0.70 → Too many irrelevant chunks; improve reranking
# context_recall < 0.80   → Missing relevant docs; improve retrieval (hybrid search)
```

---

**RAG-20. How do you build a golden dataset for RAG evaluation?**

A golden dataset is a curated set of (question, ground_truth_answer, relevant_doc_ids) triples. It is the most important evaluation artifact for a RAG system, but most teams either don't have one or let it go stale.

> **Interview tip:** Three creation strategies: human annotation (expensive, highest quality), synthetic generation with LLMs (scalable, needs validation), and production query mining with expert review (best ROI). Name all three.

```python
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import json

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

QUESTION_GEN_PROMPT = """Given the following document chunk, generate {n_questions} 
diverse questions answerable using ONLY this text.
Include factual, comparative, and "why/how" questions.
Return as a JSON list of strings.

Document:
{chunk}

Questions:"""

async def generate_eval_questions(doc: Document, n: int = 3):
    response = await llm.ainvoke(
        QUESTION_GEN_PROMPT.format(chunk=doc.page_content, n_questions=n)
    )
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return []

# Keeping the golden dataset fresh:
# - Add 20-50 new questions per month from real production queries
# - Remove questions covering deprecated features/policies
# - Re-validate against updated documents when source docs change
# - Track coverage: what % of your golden dataset covers each major topic?

def check_dataset_coverage(golden_dataset: list, topic_categories: list) -> dict:
    coverage = {topic: 0 for topic in topic_categories}
    for item in golden_dataset:
        for topic in topic_categories:
            if topic.lower() in item["question"].lower():
                coverage[topic] += 1
    return {topic: count / len(golden_dataset) for topic, count in coverage.items()}
```

---

## Advanced Retrieval Patterns

---

**RAG-21. What is Corrective RAG (CRAG) and when should you use it?**

Corrective RAG adds a self-correction step after retrieval: it evaluates retrieved documents for relevance, and if they are insufficient, either performs web search or returns a "cannot answer" signal. It dramatically reduces hallucination from failed retrieval.

> **Interview tip:** CRAG is commonly implemented as a LangGraph workflow with conditional edges. It is the production pattern for high-stakes applications where answering from insufficient context is worse than saying "I don't know."

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from langchain_core.documents import Document
import json

class CRAGState(TypedDict):
    query: str
    documents: List[Document]
    document_grade: str      # "relevant" | "irrelevant"
    transformed_query: str
    answer: str
    source: str              # "knowledge_base" | "web_search"

def retrieve(state: CRAGState) -> CRAGState:
    docs = retriever.invoke(state["query"])
    return {**state, "documents": docs}

GRADER_PROMPT = """Assess if this document contains information to answer the question.
Return JSON: {{"grade": "relevant" | "irrelevant"}}

Question: {question}
Document: {document}"""

def grade_documents(state: CRAGState) -> CRAGState:
    grades = []
    for doc in state["documents"]:
        result = llm.invoke(GRADER_PROMPT.format(
            question=state["query"], document=doc.page_content
        ))
        grade_data = json.loads(result.content)
        grades.append(grade_data["grade"])
    overall = "relevant" if grades.count("relevant") >= len(grades) / 2 else "irrelevant"
    return {**state, "document_grade": overall}

def rewrite_query(state: CRAGState) -> CRAGState:
    rewritten = llm.invoke(
        f"Rewrite for a web search engine:\n{state['query']}"
    )
    return {**state, "transformed_query": rewritten.content}

def web_search(state: CRAGState) -> CRAGState:
    from langchain_community.tools.tavily_search import TavilySearchResults
    results = TavilySearchResults(max_results=3).invoke(state["transformed_query"])
    web_docs = [Document(page_content=r["content"], metadata={"source": r["url"]})
                for r in results]
    return {**state, "documents": web_docs, "source": "web_search"}

graph = StateGraph(CRAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_documents", grade_documents)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("generate", generate_answer)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    lambda s: "generate" if s["document_grade"] == "relevant" else "rewrite_query"
)
graph.add_edge("rewrite_query", "web_search")
graph.add_edge("web_search", "generate")
graph.add_edge("generate", END)

crag_app = graph.compile()
```

---

**RAG-22. What is Agentic RAG and how does it handle multi-hop questions?**

Agentic RAG treats retrieval as a tool that an LLM agent can call iteratively. For multi-hop questions (e.g., "What is the revenue of the company that acquired Startup X in 2023?"), the agent retrieves Step 1 ("who acquired Startup X?"), then uses that answer to retrieve Step 2 ("what is that company's revenue?"). Standard RAG fails at multi-hop because all retrieval happens in a single pass.

> **Interview tip:** Implemented as a ReAct agent with retrieval as a tool. Key production concern: loop detection — agents can get stuck in cycles. Always set a recursion limit. Name both the pattern and the risk.

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for specific facts, documents, or data.
    Break complex questions into multiple targeted searches."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
        for d in docs
    )

@tool
def search_financial_database(description: str) -> str:
    """Query the financial database in plain English.
    Use for precise numerical data: revenue, expenses, ratios, time-series."""
    sql = sql_chain.invoke({"question": description})
    return db.run(sql)

agentic_rag = create_react_agent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=[search_knowledge_base, search_financial_database],
    state_modifier="""You are a precise research assistant.
    For complex questions, break them into sub-questions and retrieve step by step.
    Always cite your sources. Never answer from memory — always use the provided tools.
    If you cannot find information after 3 searches, state that clearly."""
)

# Multi-hop example:
# Q: "What is the D/E ratio of the company that acquired FinCorp in 2023?"
# Step 1: search_knowledge_base("acquisition of FinCorp 2023") → "AcquiCo acquired FinCorp"
# Step 2: search_financial_database("AcquiCo debt-to-equity ratio") → "D/E = 1.4"
# Answer: "AcquiCo's D/E ratio is 1.4 [Source: 2023 Annual Report, Financial DB]"
```

---

## RAG Security & Compliance

---

**RAG-23. What are the security risks in a production RAG system and how do you mitigate them?**

RAG introduces unique security risks: prompt injection through document content, data leakage across tenants, and retrieval of unauthorized documents.

> **Interview tip:** Three categories: prompt injection (through documents), authorization bypass (multi-tenant access control), and PII leakage (sensitive data in retrieved chunks). Name all three with concrete mitigations.

```python
import re

# RISK 1: PROMPT INJECTION VIA DOCUMENT CONTENT
INJECTION_PATTERNS = [
    r"ignore (previous|prior|all) instructions",
    r"system prompt",
    r"you are now",
    r"disregard your (guidelines|rules|instructions)",
]

def sanitize_retrieved_content(content: str) -> str:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            import logging
            logging.warning(f"Potential injection in document: {content[:100]}")
            content = re.sub(pattern, "[FILTERED CONTENT]", content, flags=re.IGNORECASE)
    return content

def wrap_context_safely(docs: list) -> str:
    """XML delimiting separates instructions from data — reduces injection risk."""
    sanitized = [sanitize_retrieved_content(d.page_content) for d in docs]
    return f"<retrieved_context>\n{chr(10).join(sanitized)}\n</retrieved_context>"

# RISK 2: MULTI-TENANT ACCESS CONTROL
# Filter must be applied at vector DB level — not application layer
def get_authorized_retriever(user_id: str, user_permissions: list):
    authorized_filter = {
        "must": [
            {"key": "org_id", "match": {"value": get_user_org(user_id)}},
            {"key": "access_level", "match": {"any": user_permissions}}
        ]
    }
    return vectorstore.as_retriever(
        search_kwargs={"k": 6, "filter": authorized_filter}
    )
    # Critical: DB-layer filtering cannot be bypassed; application-layer filtering can

# RISK 3: PII IN RETRIEVED CHUNKS — scrub at ingestion time
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def scrub_pii_from_chunk(text: str) -> str:
    results = analyzer.analyze(text=text, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=results).text
    # "John Smith's SSN is 123-45-6789" → "<PERSON>'s SSN is <US_SSN>"
```

---

**RAG-24. How do you implement GDPR right-to-erasure (right to be forgotten) in a RAG system?**

GDPR's right to erasure means that when a user requests deletion of their data, all copies must be deleted — including any vector embeddings derived from their documents. This is the "right to be forgotten" problem for RAG: embeddings are derived data and must be purged too.

> **Interview tip:** The production pipeline is: delete source document → find all embeddings by doc_id → delete embeddings → invalidate cache → log the deletion for audit. Always audit even failed deletions.

```python
from datetime import datetime, timedelta

class GDPRCompliantRAG:
    def __init__(self, vectorstore, document_store, cache, audit_log):
        self.vectorstore = vectorstore
        self.document_store = document_store
        self.cache = cache
        self.audit_log = audit_log

    async def delete_document(self, doc_id: str, requester_id: str, reason: str):
        deletion_event = {
            "doc_id": doc_id, "requester_id": requester_id, "reason": reason,
            "timestamp": datetime.utcnow().isoformat(), "steps": []
        }
        try:
            self.document_store.delete(doc_id)
            deletion_event["steps"].append("document_store: deleted")
            
            self.vectorstore.delete(filter={"doc_id": doc_id})
            deletion_event["steps"].append("vector_store: embeddings deleted")
            
            self.cache.delete_by_pattern(f"*{doc_id}*")
            deletion_event["steps"].append("cache: invalidated")
            
            deletion_event["status"] = "completed"
        except Exception as e:
            deletion_event["status"] = "failed"
            deletion_event["error"] = str(e)
            raise
        finally:
            self.audit_log.write(deletion_event)  # Always audit, even failures
```

---

**RAG-25. How do you handle contradictory information across retrieved documents?**

In a large corpus, the same question may have conflicting answers in different documents (e.g., a policy updated in Q2 contradicts the Q1 version). Naive RAG stuffs both into context and lets the LLM pick — which often produces confused answers.

> **Interview tip:** Production resolution hierarchy: (1) most recent version wins for versioned docs, (2) primary source beats secondary source, (3) if unresolvable, disclose the conflict to the user rather than silently averaging. Name all three.

```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json

class ConflictDetectionResult(BaseModel):
    has_conflict: bool
    conflicting_claims: List[str]
    resolution: str  # "use_most_recent" | "use_authoritative" | "present_both" | "escalate"

CONFLICT_DETECTOR_PROMPT = """Analyze these documents for contradictions regarding the question.
Documents: {documents}
Question: {question}
Return JSON: {{"has_conflict": true/false, "conflicting_claims": [...], 
"resolution": "use_most_recent|present_both|escalate"}}"""

async def retrieve_with_conflict_resolution(query: str) -> dict:
    docs = await retriever.ainvoke(query)
    
    # Sort by recency — most recent is most authoritative for versioned policies
    docs_sorted = sorted(
        docs,
        key=lambda d: datetime.fromisoformat(d.metadata.get("last_updated", "2000-01-01")),
        reverse=True
    )
    
    docs_summary = "\n---\n".join(
        f"[{d.metadata.get('source','?')} | {d.metadata.get('last_updated','?')}]\n{d.page_content[:400]}"
        for d in docs_sorted
    )
    
    conflict_result = await structured_llm.ainvoke(
        CONFLICT_DETECTOR_PROMPT.format(documents=docs_summary, question=query)
    )
    
    if conflict_result.has_conflict:
        if conflict_result.resolution == "use_most_recent":
            context = docs_sorted[0].page_content
            disclaimer = f"\n\n⚠️ Multiple versions exist. Using most recent ({docs_sorted[0].metadata.get('last_updated')})."
        elif conflict_result.resolution == "escalate":
            return {
                "answer": "Conflicting information found. Please consult a subject matter expert.",
                "conflict": True
            }
        else:
            context = docs_summary
            disclaimer = "\n\n⚠️ Documents contain conflicting information. Both perspectives are presented."
    else:
        context = "\n\n".join(d.page_content for d in docs_sorted[:4])
        disclaimer = ""
    
    answer = await llm.ainvoke(f"Context: {context}\n\nQuestion: {query}")
    return {"answer": answer.content + disclaimer, "conflict": conflict_result.has_conflict}
```

---

**RAG-26. What is the difference between Naive RAG, Advanced RAG, and Modular RAG?**

These are the three generations of RAG architecture. Knowing the distinctions signals depth of experience and keeps interviewers from using outdated patterns as baselines.

> **Interview tip:** This taxonomy comes from the "RAG Survey" paper (Gao et al., 2023). Production systems at mature AI companies operate at the Modular RAG level. Cite specific components that distinguish each generation.

```
NAIVE RAG (2020-2022):
  Indexing:  Fixed-size chunking → embedding → vector store
  Retrieval: Single-pass similarity search, top-k
  Generation: Stuff all chunks into prompt → LLM generates
  Problems:  Low retrieval precision, context overflow, no self-correction

ADVANCED RAG (2023):
  Adds pre-retrieval:  Query rewriting, HyDE, step-back prompting
  Adds post-retrieval: Cross-encoder reranking, contextual compression
  Better chunking:     Semantic, parent-child, proposition-level
  Better indexing:     Metadata enrichment, hierarchical indices
  Problems:  Linear pipeline, no feedback loops, single-source

MODULAR RAG (2024+):
  Adaptive retrieval:  Agent decides when/what to retrieve
  Iterative retrieval: Retrieve → read → retrieve again if needed
  Self-correction:     CRAG, Self-RAG patterns
  Multi-source:        Vector DB + SQL + API + web
  Graph-based:         RAG over knowledge graphs (GraphRAG)
  Production examples: Perplexity AI, Glean, Notion AI
  Implementation:      LangGraph workflows with conditional routing
```

---

**RAG-27. How do you handle multi-modal RAG — combining text, tables, and images?**

Enterprise documents (financial reports, scientific papers, product manuals) contain text, tables, and figures. Standard RAG only handles text. Multi-modal RAG retrieves and reasons over all content types.

> **Interview tip:** Two production architectures: (1) extract-and-describe (use vision LLMs to describe images/tables as text, then embed descriptions), and (2) multi-modal embeddings (embed images and text in the same vector space). Architecture (1) is the production standard because it is more controllable and auditable.

```python
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)

def extract_table_as_text(table_image_bytes: bytes, context: str = "") -> str:
    """Use vision LLM to convert a table image to structured text."""
    b64 = base64.b64encode(table_image_bytes).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": f"""Extract all data from this table as structured text.
Use format: [Row N] Col1: Value | Col2: Value | ...
Preserve all numbers exactly. Do not round or approximate.
Context: {context}"""},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    ])
    return vision_llm.invoke([message]).content

def extract_figure_as_text(figure_bytes: bytes, surrounding_text: str = "") -> str:
    """Describe a figure/chart for text-based retrieval."""
    b64 = base64.b64encode(figure_bytes).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": f"""Describe this figure for a search index. Include:
1. Chart type (bar, line, pie, scatter, etc.)
2. Key data points and their values (exact numbers)
3. Trends or patterns shown
4. Units, time periods, title, axis labels
Context: {surrounding_text[:500]}"""},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    ])
    return vision_llm.invoke([message]).content

# At ingestion: each table/figure becomes a Document with content_type metadata
def process_document_multimodal(pdf_path: str) -> list:
    docs = []
    # Extract tables → serialize as text Documents
    docs.append(Document(
        page_content=extract_table_as_text(table_bytes, context=pdf_path),
        metadata={"source": pdf_path, "content_type": "table", "page": page_num}
    ))
    # Extract figures → describe as text Documents
    docs.append(Document(
        page_content=extract_figure_as_text(figure_bytes, nearby_text),
        metadata={"source": pdf_path, "content_type": "figure", "page": page_num}
    ))
    return docs
```

---

**RAG-28. How do you implement streaming responses in a production RAG system?**

Users expect progressive token streaming rather than waiting 3-5 seconds for a complete answer. Retrieval cannot stream (you need all documents before assembling context), but generation can.

> **Interview tip:** The production pattern uses `astream()` after blocking retrieval. For web frontends, stream via Server-Sent Events (SSE). Mention that Nginx buffering must be disabled for SSE to work correctly.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

app = FastAPI()
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

async def rag_stream_generator(query: str):
    # Step 1: Retrieve (blocking — must complete before generation starts)
    docs = await retriever.ainvoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    
    # Step 2: Emit retrieval metadata as first event
    yield f"data: {json.dumps({'type': 'metadata', 'sources': [d.metadata.get('source','') for d in docs]})}\n\n"
    
    # Step 3: Stream generation token by token
    prompt = ChatPromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    async for chunk in (prompt | llm).astream({"context": context, "question": query}):
        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
    
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

@app.get("/rag/stream")
async def stream_rag(query: str):
    return StreamingResponse(
        rag_stream_generator(query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

---

**RAG-29. How do you implement knowledge graph RAG (GraphRAG) and when is it better than vector RAG?**

GraphRAG stores knowledge as a graph (entities as nodes, relationships as edges) and retrieves by traversing relationships. It excels at multi-hop reasoning and global synthesis across many documents.

> **Interview tip:** Microsoft's GraphRAG paper (2024) showed it significantly outperforms vector RAG on questions requiring synthesis across many documents ("global questions"). Standard vector RAG wins on local, specific lookups. Use both in a hybrid router.

```python
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
graph_db = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

transformer = LLMGraphTransformer(llm=llm)

def ingest_to_graph(documents: list):
    graph_docs = transformer.convert_to_graph_documents(documents)
    graph_db.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)

def graph_rag_query(query: str) -> str:
    entities_response = llm.invoke(
        f"Extract main entities (people, companies, concepts) from: {query}\nReturn comma-separated list."
    )
    entities = [e.strip() for e in entities_response.content.split(",")]
    
    cypher = """
    MATCH (n) WHERE n.id IN $entities OR n.name IN $entities
    MATCH (n)-[r*1..2]-(m)
    RETURN n.id, type(r), m.id, n.description, m.description LIMIT 50"""
    
    results = graph_db.query(cypher, params={"entities": entities})
    context = "\n".join(f"{r['n.id']} --[{r['type(r)']}]--> {r['m.id']}" for r in results)
    return llm.invoke(f"Graph relationships:\n{context}\n\nQuestion: {query}").content

# When to use GraphRAG vs Vector RAG:
# Specific fact lookup    → Vector RAG   ("What is X?")
# Multi-hop reasoning     → GraphRAG     ("A's partner's revenue?")
# Cross-document synthesis → GraphRAG   ("Key themes across all filings?")
# Relationship mapping    → GraphRAG     ("How does A relate to B?")
# Recent/exact quote      → Vector RAG   ("Quote from Section 4.2")
```

---

**RAG-30. How do you do RAG over code repositories?**

Code has different structure from prose — functions call other functions, and relevant context for a code question may span multiple files. Naive text chunking destroys the semantic units of code.

> **Interview tip:** Production pattern: AST-based chunking at function/class boundaries, with call graph metadata for context expansion. Tree-sitter or Python's `ast` module extracts structure. Each function becomes a Document with signature, docstring, file, and call graph.

```python
import ast
from langchain_core.documents import Document

def extract_python_functions(source_code: str, file_path: str) -> list[Document]:
    docs = []
    tree = ast.parse(source_code)
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            func_source = ast.get_source_segment(source_code, node)
            if not func_source:
                continue
            
            docstring = ast.get_docstring(node) or ""
            calls = [
                n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            ]
            
            content = f"""File: {file_path}
Name: {node.name}
Docstring: {docstring}

Code:
{func_source}"""
            
            docs.append(Document(
                page_content=content,
                metadata={"file": file_path, "name": node.name, "calls": calls, "line": node.lineno}
            ))
    return docs

def retrieve_with_call_graph(query: str, vectorstore, top_k: int = 3) -> list:
    """Retrieve top-k functions, then expand with their callees for richer context."""
    primary_docs = vectorstore.similarity_search(query, k=top_k)
    all_names = {d.metadata["name"] for d in primary_docs}
    expanded = list(primary_docs)
    
    for doc in primary_docs:
        for callee_name in doc.metadata.get("calls", []):
            if callee_name not in all_names:
                callee_docs = vectorstore.similarity_search(
                    f"function {callee_name}", k=1, filter={"name": callee_name}
                )
                expanded.extend(callee_docs)
                all_names.add(callee_name)
    return expanded
```

---

**RAG-31. What is the difference between dense retrieval, sparse retrieval, and hybrid retrieval?**

Dense retrieval embeds text as continuous vectors and uses ANN search. Sparse retrieval uses term frequency statistics (BM25/TF-IDF). Hybrid combines both with Reciprocal Rank Fusion.

> **Interview tip:** Dense wins on semantic similarity ("affordable" matches "budget-friendly"); sparse wins on keyword precision (exact product codes, names, acronyms). Production systems almost always use hybrid. Reciprocal Rank Fusion (RRF) is the standard merge strategy.

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

dense = vectorstore.as_retriever(search_kwargs={"k": 10})
bm25 = BM25Retriever.from_documents(all_docs, k=10)

# Hybrid with Reciprocal Rank Fusion
# Weights tuned on golden dataset: 0.3 sparse + 0.7 dense is a common starting point
hybrid = EnsembleRetriever(
    retrievers=[bm25, dense],
    weights=[0.3, 0.7]
)

# When sparse outperforms dense:
# - Exact codes: "SKU-ABC-123", "ISIN: US0378331005" → BM25 finds exact match
# - Named entities in specialized domains: "SOFR", "FRB-NY" → BM25 wins
# - Short queries: BM25 is more reliable for 1-3 word queries

# When dense outperforms sparse:
# - Synonyms: "car" matches "automobile", "vehicle"
# - Paraphrasing: "get rid of" matches "remove", "eliminate"
# - Conceptual: "what causes inflation?" → no exact keyword match needed
```

---

**RAG-32. How do you optimize RAG costs at high volume?**

At scale, RAG costs come from four sources: ingestion embedding, query embedding, LLM generation (largest), and vector DB hosting. Production systems optimize each.

> **Interview tip:** Most candidates only address LLM cost. Senior engineers address all four. A well-optimized RAG system can reduce costs by 60-80% vs a naive implementation without sacrificing quality.

```python
# COST 1: Ingestion embeddings — use smaller model (5x cheaper, ~95% quality)
# text-embedding-3-small: $0.02/1M tokens vs text-embedding-3-large: $0.13/1M tokens

# COST 2: Query embeddings — semantic cache (30-60% cache hit rate in production)
# See RAG-17 for full implementation

# COST 3: LLM generation — route to cheaper model when confidence is high
async def cost_optimized_generation(query: str, context: str, confidence: float) -> str:
    is_simple = len(query.split()) < 15 and "?" in query
    model = "gpt-4o-mini" if (confidence > 0.90 and is_simple) else "gpt-4o"
    # gpt-4o-mini is ~16x cheaper than gpt-4o
    return (await ChatOpenAI(model=model, temperature=0).ainvoke(
        f"Context: {context}\n\nQuestion: {query}"
    )).content

# COST 4: Vector DB — use PGVector for <5M vectors (free with existing Postgres)
# vs $70-200/month for managed Pinecone at same scale

# COST 5: Unnecessary retrieval — skip vector search for non-questions
RETRIEVAL_NEEDED_PROMPT = """Does this message require searching a knowledge base?
YES: questions or requests needing specific information
NO: greetings, acknowledgments, simple confirmations
Message: {message}
Answer (YES/NO):"""

async def should_retrieve(message: str, llm) -> bool:
    result = await llm.ainvoke(RETRIEVAL_NEEDED_PROMPT.format(message=message))
    return result.content.strip().upper() == "YES"
# This alone saves ~20% of vector DB queries in conversational RAG
```

---

**RAG-33. How do you implement RAG with real-time data sources?**

Embedding-based RAG is inherently stale — the vector index reflects the state of the world at ingestion time. Real-time RAG requires a tiered architecture that combines the static knowledge base with live data.

> **Interview tip:** The production pattern is a **tiered freshness architecture**: static knowledge base (updated daily/weekly) + near-real-time layer (Elasticsearch over CDC stream, sub-minute lag) + real-time API calls (live prices, balances). A freshness classifier routes each query to the right tier.

```python
from enum import Enum
from datetime import timedelta

class DataFreshness(str, Enum):
    STATIC = "static"       # Days/weeks OK (policies, documentation)
    RECENT = "recent"       # Hours OK (daily reports, summaries)
    REALTIME = "realtime"   # Must be current (prices, balances, status)

class TieredRAGRetriever:
    def __init__(self, vector_store, streaming_index, live_api_tools, llm):
        self.vector_store = vector_store
        self.streaming_index = streaming_index  # Elasticsearch over Kafka stream
        self.live_api_tools = live_api_tools    # Bloomberg, internal APIs
        self.llm = llm

    def classify_freshness(self, query: str) -> DataFreshness:
        realtime_kw = ["current", "now", "today", "live", "latest price", "right now"]
        recent_kw = ["this week", "recent", "last few days", "yesterday"]
        q = query.lower()
        if any(k in q for k in realtime_kw):
            return DataFreshness.REALTIME
        elif any(k in q for k in recent_kw):
            return DataFreshness.RECENT
        return DataFreshness.STATIC

    async def retrieve(self, query: str) -> list:
        freshness = self.classify_freshness(query)
        if freshness == DataFreshness.STATIC:
            return await self.vector_store.as_retriever().ainvoke(query)
        elif freshness == DataFreshness.RECENT:
            return await self.streaming_index.search(query, time_window=timedelta(hours=24))
        else:
            from langgraph.prebuilt import create_react_agent
            agent = create_react_agent(self.llm, self.live_api_tools)
            result = await agent.ainvoke({"messages": [("user", query)]})
            from langchain_core.documents import Document
            return [Document(
                page_content=result["messages"][-1].content,
                metadata={"source": "live_api", "freshness": "realtime"}
            )]
```

---

**RAG-34. How do you test a RAG system end-to-end before deploying to production?**

RAG systems have three testable layers: unit tests (individual components), integration tests (full pipeline with known inputs/outputs), and evaluation tests (RAGAS quality gates on golden dataset).

> **Interview tip:** Most candidates describe only integration testing. Senior engineers distinguish all three layers and know that quality gates in CI/CD block deployments when RAGAS scores drop — the same principle as code coverage thresholds.

```python
import pytest
from unittest.mock import Mock
from langchain_core.documents import Document

# LAYER 1: UNIT TESTS
class TestChunking:
    def test_chunk_metadata_preserved(self):
        doc = Document(
            page_content="Long text...",
            metadata={"source": "s3://bucket/file.pdf", "page": 3}
        )
        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            assert chunk.metadata.get("source") == "s3://bucket/file.pdf"

# LAYER 2: INTEGRATION TESTS
class TestRAGPipeline:
    @pytest.fixture
    def golden_qa_pairs(self):
        return [{
            "question": "What is the company's revenue growth target?",
            "expected_keywords": ["15%", "revenue"],
            "expected_source": "strategic_plan_2024.pdf"
        }]

    def test_retrieval_returns_correct_source(self, golden_qa_pairs):
        for qa in golden_qa_pairs:
            docs = retriever.invoke(qa["question"])
            sources = [d.metadata.get("source", "") for d in docs]
            assert qa["expected_source"] in sources

    def test_answer_contains_expected_content(self, golden_qa_pairs):
        for qa in golden_qa_pairs:
            answer = rag_chain.invoke(qa["question"])
            for keyword in qa["expected_keywords"]:
                assert keyword.lower() in answer.lower()

# LAYER 3: QUALITY GATES IN CI/CD
def test_ragas_quality_gates():
    """Fail deployment if quality drops below thresholds."""
    results = evaluate(golden_dataset, metrics=[faithfulness, answer_relevancy])
    assert results["faithfulness"] >= 0.88, \
        f"Faithfulness {results['faithfulness']:.3f} below threshold — deployment blocked"
    assert results["answer_relevancy"] >= 0.82, \
        f"Answer relevancy {results['answer_relevancy']:.3f} below threshold — deployment blocked"
```

---

**RAG-35. How do you handle context window limits when documents are very long?**

When retrieved chunks exceed the LLM's context limit, you must decide what to include. Naive truncation loses critical information.

> **Interview tip:** Four strategies: map-reduce (production standard, parallelizable), refine (iterative, good for building up answers), recursive summarization (compression), selective inclusion (reranker picks best subset). Name each and state when to use them.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

MAP_PROMPT = ChatPromptTemplate.from_template(
    "Extract information relevant to: {question}\n\nText: {text}\n\nRelevant information:"
)

REDUCE_PROMPT = ChatPromptTemplate.from_template(
    "Synthesize a comprehensive answer to: {question}\n\nExtracted information:\n{text}\n\nAnswer:"
)

async def map_reduce_rag(query: str, docs: list) -> str:
    import asyncio
    # MAP: process each chunk independently (parallelizable — all run concurrently)
    map_chain = MAP_PROMPT | llm
    map_tasks = [map_chain.ainvoke({"question": query, "text": doc.page_content}) for doc in docs]
    mapped_results = await asyncio.gather(*map_tasks)
    
    relevant = [r.content for r in mapped_results if "no relevant" not in r.content.lower()]
    
    # REDUCE: synthesize all relevant excerpts into one answer
    reduce_chain = REDUCE_PROMPT | llm
    final = await reduce_chain.ainvoke({"question": query, "text": "\n\n---\n\n".join(relevant)})
    return final.content

# When to use each:
# Map-reduce:       Q&A + summarization over many docs. Parallelizable. ← production default
# Refine:           Answer builds progressively (timelines, step-by-step analysis)
# Recursive:        Pure compression (500-page doc → 5-page summary)
# Selective (top-k): Context window is generous but you want precision → rerank + top-4
```

---

**RAG-36. How do you handle multilingual RAG in production?**

Most embedding models are trained predominantly on English. For low-resource languages, vector similarity degrades significantly. A query in Hindi may not retrieve semantically equivalent Hindi documents.

> **Interview tip:** Four strategies in priority order: multilingual embedding models (LaBSE, multilingual-e5-large), language-specific indices with routing, cross-lingual translation at query time, and machine translation at ingestion. Each has latency, cost, and quality tradeoffs.

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langdetect import detect

# Strategy 1: Multilingual embedding model (preferred)
multilingual_embedder = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",  # 100+ languages
    encode_kwargs={"normalize_embeddings": True}
)

# Strategy 2: Language detection + per-language index routing
def multilingual_retriever(query: str, retrievers_by_lang: dict):
    try:
        lang = detect(query)  # "en", "hi", "ar", "fr", etc.
    except:
        lang = "en"  # fallback
    
    if lang in retrievers_by_lang:
        return retrievers_by_lang[lang].invoke(query)
    return retrievers_by_lang["multilingual"].invoke(query)

# Strategy 3: Cross-lingual query translation (simplest, good for low-frequency languages)
async def translate_and_retrieve(query: str, source_lang: str) -> list:
    if source_lang != "en":
        english_query = (await llm.ainvoke(
            f"Translate to English, preserving technical terminology:\n{query}"
        )).content
    else:
        english_query = query
    return await retriever.ainvoke(english_query)
```

---

**RAG-37. What is the "needle in a haystack" problem in RAG and how is it solved?**

A single critical piece of information buried in a large corpus — vector ANN search degrades in precision at scale and may miss the exact best match. For truly unique identifiers (contract IDs, ISINs, invoice numbers), similarity search is the wrong tool entirely.

> **Interview tip:** Solutions in priority order: pre-filter by metadata (reduces search space 99%), exact-match retrieval for unique identifiers (O(1) lookup, not O(log n) ANN), and tiered search (exact first, ANN fallback).

```python
import re

def smart_retriever(query: str, metadata_filter: dict, vectorstore, document_store):
    # Stage 1: Try exact ID match — always faster and more accurate for unique identifiers
    id_patterns = [
        r"\b[A-Z]{2,5}-\d{3,}-\d{4}\b",  # Contract IDs: "MSFT-001-2024"
        r"\b[A-Z]{2}\d{10}\b",             # ISIN codes: "US0378331005"
        r"\bINV-\d{6}\b"                   # Invoice numbers
    ]
    for pattern in id_patterns:
        match = re.search(pattern, query)
        if match:
            doc = document_store.get(match.group())  # O(1) exact lookup
            if doc:
                return [doc]
    
    # Stage 2: Pre-filter by metadata, then ANN search on reduced set
    return vectorstore.similarity_search(
        query=query,
        k=6,
        filter=metadata_filter    # Reduces from 5M to ~5K vectors before ANN
    )
```

---

**RAG-38. How do you implement self-RAG — where the model decides whether to retrieve?**

Self-RAG gives the model agency over retrieval: it decides whether retrieval is needed, evaluates retrieved documents, critiques its own answer, and can regenerate if unsatisfied. Compared to always-on RAG, it reduces unnecessary API calls and improves answer quality for queries that don't need retrieval.

> **Interview tip:** Implemented as a LangGraph with self-critique loops. Key production concern: setting MAX_ITERATIONS to prevent infinite loops. Name the reflection tokens concept from the original Self-RAG paper (Asai et al., 2023).

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, List

class SelfRAGState(TypedDict):
    query: str
    documents: Optional[List]
    answer: Optional[str]
    needs_retrieval: bool
    answer_is_supported: bool
    answer_is_useful: bool
    iteration: int

MAX_ITERATIONS = 3

RETRIEVAL_DECISION_PROMPT = """Does answering this question require retrieving external documents?
YES: questions needing specific facts, recent data, or domain knowledge.
NO: general reasoning or knowledge questions.
Return JSON: {{"needs_retrieval": true/false}}
Question: {question}"""

def decide_retrieval(state: SelfRAGState) -> SelfRAGState:
    import json
    result = llm.invoke(RETRIEVAL_DECISION_PROMPT.format(question=state["query"]))
    data = json.loads(result.content)
    return {**state, "needs_retrieval": data["needs_retrieval"]}

CRITIQUE_PROMPT = """Evaluate this answer:
1. Is it fully supported by the context? 2. Does it fully address the question?
Context: {context}  Question: {question}  Answer: {answer}
Return JSON: {{"supported": true/false, "useful": true/false}}"""

def critique_answer(state: SelfRAGState) -> SelfRAGState:
    import json
    context = "\n".join(d.page_content for d in state["documents"])
    result = llm.invoke(CRITIQUE_PROMPT.format(
        context=context, question=state["query"], answer=state["answer"]
    ))
    data = json.loads(result.content)
    return {**state, "answer_is_supported": data["supported"], "answer_is_useful": data["useful"],
            "iteration": state["iteration"] + 1}

def should_stop(state: SelfRAGState) -> str:
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"  # Prevent infinite loops
    if state["answer_is_supported"] and state["answer_is_useful"]:
        return "end"
    return "retry"
```

---

**RAG-39. How do you implement document-level access control (multi-tenant RAG) at scale?**

In enterprise SaaS RAG (HR chatbots, legal assistants, financial advisors), different users can access different document subsets. This is Row-Level Security at the document level.

> **Interview tip:** The production rule: access control filter must be applied at the vector DB layer, never in application code. Application-layer filtering can be bypassed; DB-layer filtering cannot. Every document must be tagged at ingestion with org_id, access_level, and allowed_roles.

```python
def ingest_with_access_control(
    documents: list,
    org_id: str,
    department: str,
    classification: str,        # "public", "internal", "confidential", "restricted"
    allowed_roles: list         # ["analyst", "manager", "executive"]
):
    for doc in documents:
        doc.metadata.update({
            "org_id": org_id,
            "department": department,
            "classification": classification,
            "allowed_roles": allowed_roles,
        })
    vectorstore.add_documents(documents)

def retrieve_authorized(query: str, user: dict) -> list:
    # user = {"org_id": "...", "departments": [...], "roles": [...]}
    filter_conditions = {
        "must": [{"key": "org_id", "match": {"value": user["org_id"]}}],
        "should": [
            {"key": "department", "match": {"any": user["departments"]}},
            {"key": "classification", "match": {"value": "public"}}
        ]
    }
    return vectorstore.similarity_search(query=query, k=6, filter=filter_conditions)
    # The filter is enforced by Qdrant/Pinecone at the DB layer — not bypassable
```

---

**RAG-40. What is the most important difference between a demo RAG system and a production RAG system?**

This is a synthesis question that tests end-to-end understanding. The answer should identify at least 5 dimensions where production RAG differs substantively from a weekend demo.

> **Interview tip:** Senior engineers know that demo RAG achieves 70-80% quality easily. Getting from 80% to 95%+ requires obsessing over every component. Name the specific technical choices at each layer.

```
DIMENSION 1: CHUNKING
  Demo:       Fixed-size splitter, chunk_size=1000
  Production: Semantic + parent-child + structure-aware chunking
              Impact: 15-20% improvement in Recall@5

DIMENSION 2: RETRIEVAL
  Demo:       Single vector similarity search
  Production: Hybrid (dense + BM25) + cross-encoder reranking + metadata filters
              Impact: 20-30% improvement in Precision@k and Recall@k

DIMENSION 3: HALLUCINATION CONTROL
  Demo:       Generic "answer from context" prompt
  Production: Confidence gate + structured output + faithfulness judge + guardrails
              Impact: Faithfulness from ~75% to 95%+

DIMENSION 4: EVALUATION
  Demo:       Manual spot-checking
  Production: Golden dataset + RAGAS + CI/CD quality gates that block deployments
              "You cannot improve what you cannot measure"

DIMENSION 5: OBSERVABILITY
  Demo:       print() statements
  Production: Full traces (query, docs, answer, latency, scores) + dashboards + alerts
              Essential for debugging regressions in production

DIMENSION 6: SECURITY
  Demo:       All documents accessible to all users
  Production: DB-layer access control, PII scrubbing, prompt injection detection,
              GDPR right-to-erasure pipeline

DIMENSION 7: FRESHNESS & VERSIONING
  Demo:       One-time ingestion, no rollback
  Production: Event-driven incremental indexing, index versioning, blue/green deployment

DIMENSION 8: COST
  Demo:       Always GPT-4o + text-embedding-3-large + no cache
  Production: Semantic cache + model routing + smaller embeddings
              Result: 60-80% cost reduction without quality loss

DIMENSION 9: SCALE
  Demo:       FAISS in memory
  Production: Qdrant/Pinecone with IVF-PQ, Ray-parallelized ingestion, horizontal scale

DIMENSION 10: FAILURE HANDLING
  Demo:       Crashes or hallucinate on bad input
  Production: "I don't know" > hallucination, circuit breakers, graceful degradation
```

---

*Last updated: June 2026 | LangChain ≥1.0 | LangGraph ≥0.4 | RAGAS ≥0.2*
