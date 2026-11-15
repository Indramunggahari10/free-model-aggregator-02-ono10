# free-model-aggregator-02

One server, two API wires, and a real free model behind every answer.

Point any OpenAI or Anthropic client at it and every completion it answers comes from a
live free OpenRouter model, dialled through the gateway on the key pushed to the competition.
It ranks the free catalog by the brief's quality ratings, so each prompt is served by the
best free model that meets it — JSON-mode prompts by the best JSON-capable model, tools
prompts by the best tools-capable model, plain prompts by the best model overall — and it
fails over down the ranking when a provider refuses, reporting the refusals it recovered
from in the `x-makefun-upstream-failures` header.

## Run

```sh
node main.ts            # serves the trial: registers with $MAKEFUN_ENV_URL, exits 0 when done
node main.ts --list     # prints the catalog document it registers, byte for byte
```

Node 22, standard library only, nothing to install.

## What it serves

- `POST /v1/chat/completions` — OpenAI Chat Completions, JSON and server-sent events
- `POST /v1/messages` — Anthropic Messages, JSON and server-sent events
- `GET /v1/models` — the model list in the wire the auth header names
  (`Authorization: Bearer` gets the OpenAI list, `x-api-key` the Anthropic one)

Every answer names, in its `model` field, the free model that really dialled it. The
catalog holds exactly the fourteen free OpenRouter models the brief lists.

## Verify locally

```sh
# catalog it registers (needs the brief):
MAKEFUN_PROMPT_FILE=<kit>/practice/trial-0/input/prompt.txt node main.ts --list

# full practice trial (needs an OpenRouter key pushed to the competition, or
# OPENROUTER_API_KEY in the environment for the local gateway):
node <kit>/cli/makefun.mjs grade-local --kit <kit> --bot . --driver process
```


Candidates are ranked by the brief's `rating_milli`, best first, among the models whose
capabilities meet the request (JSON mode, tools, vision, read off the request body). A
429 or 5xx fails over to the next model down, up to three attempts a prompt, and a model
that refused is cooled down for the rest of the trial. A 400 or 401 is not retried.
Streaming answers are collected in full before the client sees a byte, so a refusal
mid-stream can still fail over cleanly. No wall clock or unseeded randomness reaches
any request, so a replay of a trial makes the same calls and leaves the same export.
