# prompt_fio_v2.py
# ─────────────────────────────────────────────────────────────────────────────
# FIO extraction prompt — v2
#
# Changes vs original prompt_for_fio.py:
#   1. "persons" replaces "person" — extract ALL named persons, not just primary
#   2. "speech_content" added — verbatim quoted speech (critical for HabitInsight
#      SAID questions and TaskMaster need_audio questions)
#   3. "source_location" / "target_location" made explicit (were implicit)
#   4. "tool" separated from "object" with clear examples
#   5. "attributes" explicitly scoped (color, quantity, state, size)
#   6. "location" field removed — replaced by source/target pair
#
# DO NOT MODIFY prompt_for_fio.py — this is a standalone replacement.
# Import as: from prompt_fio_v2 import FIO_EXTRACTION_PROMPT
# ─────────────────────────────────────────────────────────────────────────────

FIO_EXTRACTION_PROMPT = '''You are a structured information extraction engine for egocentric video captions.

Your task: extract ALL distinct human interactions from the caption and return them
as a JSON array. Each element is one Functional Interaction Object (FIO).

OUTPUT FORMAT: A JSON array only. No explanation. No markdown fences.

SCHEMA FOR EACH FIO:
{
  "persons":         string,   // ALL named persons present or mentioned, comma-separated.
                               // e.g. "Alice, Tasha, Katrina". Use "" if no named person.
                               // IMPORTANT: list every named person visible or speaking, not just the actor.
  "action":          string,   // Primary verb phrase. e.g. "helped assemble", "said she dislikes", "picked up"
  "object":          string,   // Primary physical item interacted with. e.g. "screwdriver", "puzzle", "pot"
                               // Use null if no physical object.
  "tool":            string,   // Instrument USED to perform the action, if different from object.
                               // e.g. scissors (tool) to cut box (object). Use null if none.
  "app":             string,   // Digital application if any. e.g. "Meituan", "WeChat", "timer". null if none.
  "source_location": string,   // Where the object or person WAS BEFORE the action.
                               // e.g. "second floor shelf", "on the table". null if not stated.
  "target_location": string,   // Where the object or person MOVED TO after the action.
                               // e.g. "kitchen counter", "living room sofa". null if not stated.
  "speech_content":  string,   // EXACT verbatim speech if any person speaks.
                               // Format: "PersonName said: <exact words>".
                               // e.g. "Tasha said: I don't want to eat hamburgers".
                               // e.g. "Shure said: coffee — without coffee I would die".
                               // Extract ALL quoted speech from the caption.
                               // Use "" if no speech is present.
  "attributes":      object    // Extra descriptive properties as key-value pairs.
                               // Use for: color, quantity, state, size, material, brand.
                               // e.g. {"color": "black", "quantity": "3", "state": "open"}
                               // Use {} if none.
}

RULES:
1. Output ONLY a valid JSON array. No text before or after.
2. Create ONE FIO per distinct interaction. A caption may have 1-5 interactions.
3. "persons" MUST list ALL named persons in the scene — not just who is acting.
   e.g. If Alice and Tasha are both present, both appear in "persons".
4. "speech_content" MUST include exact words when a person speaks.
   This is the most commonly missed field — always check for quoted speech.
5. "object" and "tool" are different:
   - Shure used scissors (tool) to cut the box (object) → tool="scissors", object="box"
   - Jake picked up the screwdriver (object) → object="screwdriver", tool=null
6. Use null for missing string fields. Use {} for missing attributes.
7. Do NOT invent information not present in the caption.
8. Person names to watch for: Alice, Tasha, Shure, Lucia, Katrina, Nicous, Choiszt, Bob, Violet.
9. The caption is from an egocentric camera.
   The phrase "I" refers to the CAMERA WEARER
   provided before the caption.

10. When the camera wearer performs an action,
    include their name inside "persons".

11. Example:
    If CAMERA WEARER is Jake and the caption says:
    "I picked up the phone"

    then output:
    "persons": "Jake"

EXAMPLES:

Caption: "I looked at my phone, opened the timer, and set it to 10 seconds. Then I turned on the timer."
Output:
[
  {"persons": "Jake", "action": "looked at", "object": "phone", "tool": null, "app": null,
   "source_location": null, "target_location": null, "speech_content": "",
   "attributes": {}},
  {"persons": "Jake", "action": "opened", "object": null, "tool": null, "app": "timer",
   "source_location": null, "target_location": null, "speech_content": "",
   "attributes": {}},
  {"persons": "Jake", "action": "set", "object": "timer", "tool": null, "app": null,
   "source_location": null, "target_location": null, "speech_content": "",
   "attributes": {"duration": "10 seconds"}}
]

Caption: "I was looking at my phone when Tasha said, 'Hahaha, I usually just choose what everyone else chooses.' Then I saw Shure using scissors to cut open a box."
Output:
[
  {"persons": "Tasha", "action": "said", "object": null, "tool": null, "app": null,
   "source_location": null, "target_location": null,
   "speech_content": "Tasha said: Hahaha, I usually just choose what everyone else chooses",
   "attributes": {}},
  {"persons": "Shure", "action": "cut open", "object": "box", "tool": "scissors", "app": null,
   "source_location": null, "target_location": null, "speech_content": "",
   "attributes": {}}
]

Caption: "Katrina and Alice were assembling the shelf using screws while Shure said, 'coffee — without coffee I would die.' I walked to the whiteboard to write the shopping list."
Output:
[
  {"persons": "Katrina, Alice", "action": "assembled", "object": "shelf", "tool": "screws", "app": null,
   "source_location": null, "target_location": null, "speech_content": "",
   "attributes": {}},
  {"persons": "Shure", "action": "said", "object": null, "tool": null, "app": null,
   "source_location": null, "target_location": null,
   "speech_content": "Shure said: coffee — without coffee I would die",
   "attributes": {}},
  {"persons": "Jake", "action": "wrote", "object": "shopping list", "tool": null, "app": null,
   "source_location": null, "target_location": "whiteboard", "speech_content": "",
   "attributes": {}}
]

Caption: "I picked up the pot from the table and moved it to the kitchen counter. Inside the pot was a whisk."
Output:
[
  {"persons": "Jake", "action": "moved", "object": "pot", "tool": null, "app": null,
   "source_location": "table", "target_location": "kitchen counter", "speech_content": "",
   "attributes": {"contents": "whisk"}}
]

Now extract all FIOs from the following caption:
'''


# ── Slot/day narrative prompts ────────────────────────────────────────────────

SLOT_NARRATIVE_PROMPT = '''You summarise 1-hour windows of egocentric life video for retrieval.

Rules:
- Write exactly 2-3 sentences.
- First person ("I" / "we"). Past tense.
- Be SPECIFIC: name the activity, persons, objects, locations, and any speech.
- Do NOT interpret or infer — only state what happened.
- If someone spoke, include what they said.
- Include approximate time if given (e.g. "around 3pm").

Summarise this 1-hour window:
'''

DAY_NARRATIVE_PROMPT = '''You summarise a full day of egocentric life video for retrieval.

Rules:
- Write exactly 5-7 sentences covering the whole day chronologically.
- First person ("I" / "we"). Past tense.
- Name the key activities, persons present, objects used, and important conversations.
- Be specific enough that an answer like "Photography" or "Sesame sauce" could be retrieved.
- Do NOT interpret — only state what happened.

Summarise this day using the hourly summaries below:
'''