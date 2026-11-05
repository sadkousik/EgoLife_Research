"""Qwen2.5-7B-Instruct backend and the question-decomposition prompt.

Used by both pipelines:
  * scripts/build_memory.py — loads the local model (``tokenizer`` / ``model``)
    for batched Phase-1 FIO extraction.
  * scripts/qa.py           — ``call_qwen`` decomposes each question into a JSON
    query (entity, action, temporal relation, query_intent, ...) using
    ``system_message`` below, and writes LLM-fallback answers.

Backends (``QWEN_BACKEND``):
  local   Hugging Face Transformers, bf16, ``device_map="auto"``. The model is
          loaded on first use, at the pinned ``MODEL_REVISION``.
  openai  Any OpenAI-compatible server (e.g. vLLM) at ``OPENAI_BASE_URL``
          serving ``OPENAI_MODEL``. The server must already be running.

Environment: QWEN_BACKEND, QWEN_MODEL_ID, QWEN_REVISION, OPENAI_BASE_URL,
OPENAI_MODEL, OPENAI_API_KEY, SEED (passed to the OpenAI backend).

``call_qwen1`` is the old Together-API path; its client is commented out, so it
is not usable as is.
"""

#--------------using Qwen extract key information from question------------------#
system_message = '''You are an information extraction engine for egocentric video question answering.

Your task is to convert the given natural-language question into a JSON object
that STRICTLY follows the schema below.

IMPORTANT RULES:
1. Output ONLY valid JSON. No explanations, no comments, no markdown.
2. Use ONLY the allowed values listed in the schema.
3. If a value is unknown or not explicitly stated, use null.
4. Do NOT invent new fields or values.
5. Choose the entity the question is ASKING ABOUT, not merely mentioning.
6. The "action" field: set ONLY if the question explicitly states a concrete action.
7. For contextual questions ("Now X does Y, what/who...?"), extract entity and action
   from the ACTUAL QUESTION part after the comma — not from the context prefix.
8. "whose" and "who" both map to query_intent "who".

SCHEMA:
{
  "entity":           string | null,
  "entity_relation":  "reference" | "ownership" | "possession",
  "action":           string | null,
  "object_relation":  "contained_object" | "on_object" | null,
  "related_person":   string | null,
  "temporal_relation":"first_time" | "last_time" | "before" | "after" |
                       "habitual" | "yesterday" | "today" | null,
  "query_intent":     "who" | "where" | "when" | "what" | "why" | "how" | null,
  "speech_required":  true | false
}

FIELD GUIDANCE:

temporal_relation:
  "first_time"  — "first", "originally", "first to"
  "last_time"   — "last", "most recent", "latest"
  "before"      — "before", "previously", "used to"
  "after"       — "after", "since", "following"
  "habitual"    — "usually", "always", "often", "typically", "never", "rarely",
                   "every time", "tend to" → USE FOR HabitInsight questions
  null          — single specific event, no temporal qualifier

query_intent:
  "who"   — who / whose / which person
  "what"  — what / which object / app / tool / food / activity
  "where" — where / which place / location
  "when"  — when / how long ago / what time
  "why"   — why / what is the reason / because of what
  "how"   — how / in what way / by what method / how many / how much
  null    — yes/no questions

speech_required:
  Set true when the answer is likely spoken by a person (plan, preference, reason,
  decision, opinion). Covers:
    - "why" questions (reason stated verbally)
    - "what does X not like / love / prefer" (stated preference)
    - "what are we planning / deciding to do" (plan)
    - "what did X say / mention / suggest" (direct speech)
    - "what drinks / food / items" when asking about group decisions
  Set false for all other questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES — one per question type pattern
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Question: Who used the screwdriver first?
{
  "entity": "screwdriver",
  "entity_relation": "reference",
  "action": "used",
  "object_relation": null,
  "related_person": null,
  "temporal_relation": "first_time",
  "query_intent": "who",
  "speech_required": false
}

Question: Whose phone is on the table?
{
  "entity": "phone",
  "entity_relation": "ownership",
  "action": null,
  "object_relation": "on_object",
  "related_person": null,
  "temporal_relation": null,
  "query_intent": "who",
  "speech_required": false
}

Question: When was the last time we talked about coffee?
{
  "entity": "coffee",
  "entity_relation": "reference",
  "action": "talked about",
  "object_relation": null,
  "related_person": null,
  "temporal_relation": "last_time",
  "query_intent": "when",
  "speech_required": false
}

Question: Who usually chats with me the most when I am cleaning in the kitchen?
{
  "entity": "person",
  "entity_relation": "reference",
  "action": "chat",
  "object_relation": null,
  "related_person": null,
  "temporal_relation": "habitual",
  "query_intent": "who",
  "speech_required": false
}

Question: What does Tasha not like to eat?
{
  "entity": "food",
  "entity_relation": "reference",
  "action": null,
  "object_relation": null,
  "related_person": "Tasha",
  "temporal_relation": "habitual",
  "query_intent": "what",
  "speech_required": true
}

Question: Why did the clerk not agree when we asked them to cut the pineapple?
{
  "entity": "clerk",
  "entity_relation": "reference",
  "action": "cut",
  "object_relation": null,
  "related_person": null,
  "temporal_relation": null,
  "query_intent": "why",
  "speech_required": true
}

Question: To shop faster, how did we divide into groups?
{
  "entity": "groups",
  "entity_relation": "reference",
  "action": "divide",
  "object_relation": null,
  "related_person": null,
  "temporal_relation": null,
  "query_intent": "how",
  "speech_required": true
}

Question: Who just helped Alice unpack and arrange the flowers?
{
  "entity": "person",
  "entity_relation": "reference",
  "action": "helped",
  "object_relation": null,
  "related_person": "Alice",
  "temporal_relation": "last_time",
  "query_intent": "who",
  "speech_required": false
}

Question: What other drinks are needed to mix with rum?
{
  "entity": "drinks",
  "entity_relation": "reference",
  "action": "mix",
  "object_relation": null,
  "related_person": null,
  "temporal_relation": null,
  "query_intent": "what",
  "speech_required": true
}

Question: Katrina is now suggesting dancing, what other activities has she suggested before?
{
  "entity": "activities",
  "entity_relation": "reference",
  "action": "suggested",
  "object_relation": null,
  "related_person": "Katrina",
  "temporal_relation": "before",
  "query_intent": "what",
  "speech_required": true
}

Question: What app did I order the takeout from?
{
  "entity": "app",
  "entity_relation": "reference",
  "action": "order",
  "object_relation": null,
  "related_person": null,
  "temporal_relation": null,
  "query_intent": "what",
  "speech_required": false
}
'''
#together ai
import os

# from together import Together
# # Initialize client once (outside function ideally)
# client = Together(
#     api_key=os.environ.get("TOGETHER_API_KEY")
# )

import json 

##added by me for qwen
from typing import Optional
# Select with QWEN_BACKEND=local|openai. The OpenAI-compatible mode works with
# vLLM (or another server implementing /v1/chat/completions).
QWEN_BACKEND = os.environ.get("QWEN_BACKEND", "local").strip().lower()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8001/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
# Pinned Hugging Face snapshot, so a later upload to the hub can't change results.
# This is the snapshot the reported runs used; set QWEN_REVISION=main to follow the hub.
MODEL_REVISION = os.environ.get("QWEN_REVISION", "a09a35458c702b33eeacc393d103063234e8bc28")

tokenizer = None
model = None


def _load_local_model() -> None:
    """Load the Transformers model on first local inference."""
    global tokenizer, model
    if model is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    print("Model device:", next(model.parameters()).device, flush=True)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id


def call_qwen(
    prompt: str,
    system_message: str = "You are an effective first-person assistant.",
    temperature: float = 0.6,#0.9 #best 0.4
    top_p: float = 0.8,  # =value is default value and float is type
    max_tokens: int = 2200
) -> Optional[str]:
    """
    Generate with the selected local or OpenAI-compatible backend.
    temperature <= 0 means greedy (deterministic) decoding.
    """
    greedy = temperature <= 0
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    try:
        if QWEN_BACKEND in {"openai", "openai_compatible", "vllm"}:
            from openai import OpenAI

            print(f"Using OpenAI-compatible Qwen endpoint: {OPENAI_BASE_URL} ({OPENAI_MODEL})")
            client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.0 if greedy else temperature,
                top_p=1.0 if greedy else top_p,
                max_tokens=max_tokens,
                seed=int(os.environ.get("SEED", 1234)),
            )
            content = response.choices[0].message.content
            return content.strip() if content else None

        if QWEN_BACKEND != "local":
            raise ValueError(
                f"Unsupported QWEN_BACKEND={QWEN_BACKEND!r}; use 'local' or 'openai'."
            )

        print("Using local Qwen model for generation.")
        _load_local_model()
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
        sampling = ({"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
                    if greedy else {"do_sample": True, "temperature": temperature, "top_p": top_p})
        outputs = model.generate(
            **inputs,
            **sampling,
            max_new_tokens=max_tokens,
            pad_token_id=model.config.pad_token_id
        )
        # Decode only newly generated tokens to avoid matching prompt text.
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Qwen generation failed ({QWEN_BACKEND} backend): {e}")
        return None

#pretraines qwen2.5 end

#together ai
def call_qwen1(
    prompt: str,
    system_message: str = "You are an effective first-person assistant.",
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_tokens: int = 2200
) -> Optional[str]:
    """
    Generate a response using Together API with Qwen2.5-7B model.
    """
    print("Using Together API Qwen model for generation.")

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="debadityaroy55_53c0/Qwen/Qwen2.5-7B-Instruct-e4857cc2",
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Together API Qwen generation failed: {e}")
        return None
    

#together ai_end


if __name__ == "__main__":
    sample_questions = []
    with open("/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/filtered_day1_f.json", "r") as f:
        data = json.load(f)
        for item in data:
            if item.get("type") == "EntityLog":
                q = item.get("question")
                if q is not None:
                    sample_questions.append(q)

    for q in sample_questions:
        print(f"Question: {q}")
        # Using Qwen to get structured JSON
        response = call_qwen(prompt=q, system_message=system_message)
        print(f"Qwen Response:\n{response}\n")
