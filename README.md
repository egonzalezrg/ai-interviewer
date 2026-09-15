# AI Interviewer

A real-time AI interviewer built in Python. It runs a structured voice interview, can answer questions about a fictional company using RAG, saves interview answers, and uses a synced avatar for the interviewer.


## What it does

* Runs a live voice interview through LiveKit
* Uses Deepgram for speech-to-text
* Uses GPT-4o with LangGraph to manage the interview flow
* Retrieves company information from a PDF with Chroma and OpenAI embeddings
* Uses tool calls to look up company information and save candidate answers
* Uses Cartesia for text-to-speech
* Uses Simli for the real-time avatar
* Keeps tool calls and raw RAG output internal so they are not spoken to the user

## Tech

Python, LiveKit, LangGraph, LangChain, OpenAI, Deepgram, Cartesia, Simli, Chroma, PyPDF, uv

## How it works

The candidate speaks to the interviewer through LiveKit.

Deepgram transcribes the audio and sends the text into the LangGraph workflow. The graph uses GPT-4o to continue the interview and decide when a tool is needed.

If the candidate asks something about the company, the agent searches a Chroma vector store built from `company\_profile.pdf`. If the candidate gives an interview answer, the agent can save it to `interview\_answers.txt`.

The final response is turned into speech with Cartesia and sent through the Simli avatar.

```text
Candidate
   |
   v
Deepgram STT
   |
   v
LangGraph + GPT-4o
   |
   +--> Company info tool --> Chroma --> company\_profile.pdf
   |
   +--> Record answer tool --> interview\_answers.txt
   |
   v
Cartesia TTS
   |
   v
Simli Avatar
   |
   v
LiveKit
```

## Project files

`agent.py`  
Sets up the LiveKit session, STT, TTS, turn detection, avatar, and connects the agent to the LangGraph workflow.

`graph.py`  
Contains the interview flow, prompts, RAG setup, tools, and response streaming logic.

`company\_profile.pdf`  
Fictional company information used by the RAG system.

`pyproject.toml` and `uv.lock`  
Project dependencies.

## Setup

### 1\. Clone the repo

```bash
git clone <YOUR-REPOSITORY-URL>
cd realtime-interview-agent-livekit
```

### 2\. Install dependencies

```bash
uv sync
```

### 3\. Set up environment variables

Make a copy of .env.example and rename the copy to .env.local.



Then open .env.local and fill in your API keys and configuration values.



Do not commit .env.local.

### 4\. Run the agent

```bash
uv run agent.py dev
```

Then connect to the agent from the LiveKit Agent Console.

## One issue I had to solve

At first, LangGraph was sending intermediate tool and RAG messages into the voice pipeline. That meant the interviewer could start reading internal retrieval output instead of only saying the final response.

I changed the workflow to use custom streaming so tool calls stay inside LangGraph and only completed assistant responses are sent to TTS.

I also added a short delay before the first greeting because the avatar audio path was sometimes not ready fast enough and the start of the first sentence would get cut off.

## Notes

* `company\_profile.pdf` is fictional and only used for testing the RAG flow.
* `interview\_answers.txt` and the local Chroma database are generated files and are ignored by Git.
* API keys are stored locally in `.env.local`.

