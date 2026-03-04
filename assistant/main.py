import re
import json
import os
import base64
import uuid
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory
from colorama import Fore, Style, init


from assistant import SemanticController
from memory_manager import MemoryManager

app = Flask(__name__)
CORS(app)  # Allows Electron to talk to it

# Initialize Colorama for Windows terminal colors
init(autoreset=True)


class Main:
    def __init__(self):
        self.llm = SemanticController()
        self.memory = MemoryManager()
        self.short_term_memory = []  # Keeps track of the current session context
        self.summary = ""

    def needs_web_search_autonomous(self, user_input, relevant_memories):
        classifier_system = (
            "You are a strict classifier.\n"
            "Return ONE TOKEN ONLY: YES or NO.\n"
            "No punctuation. No extra words. No explanations.\n"
        )

        classifier_user = (
            f"User input:\n{user_input}\n\n"
            f"Context from memory:\n{relevant_memories}\n\n"
            "Decide if an internet search is REQUIRED for factual correctness.\n"
            "Reply YES only if the information from your memory is not enought\n"
            "Reply NO only if there is no need for actual up to date information\n"
            "Do NOT guess, be absolutely honest with your answer\n"
            "Return only YES or NO."
        )

        raw = self.llm.ask( 

            prompt=f"{classifier_system}\n\n{classifier_user}",
            purpose = "web check" 
        ).strip().upper()

        # Parse first YES/NO only (ignore anything else)
        token = raw.split()[0] if raw else "NO"
        decision = "YES" if token.startswith("Y") else "NO"

        print(f"{Fore.GREEN}[AI decision on web search: {decision}]")
        return decision == "YES"

    def make_chat_summary(self, chat_history: str) -> str:
        summary_prompt = (
            "You generate a concise chat-list summary (like a sidebar entry).\n"
            f"Input: the full conversation transcript: {chat_history}.\n"
            "Output:\n"
            "- A short TITLE (3–7 words) that captures the main topic.\n"
            "- A short SNIPPET (1 sentence, max 18 words) describing what the user wanted/did.\n"
            "- Nothing else.\n"
            "Return ONLY valid JSON. No extra text.\n"
            "Schema:\n"
            '"title": value, "snippet": value\n'
            "Rules:\n"
            "- Be faithful: do not invent details not present in the transcript.\n"
            "- Prefer the user’s intent and the main task over small talk.\n"
            "- If the conversation has multiple topics, pick the dominant one or the final topic.\n"
            "- Avoid names, private/sensitive info, and long numbers unless essential.\n"
            "- No emojis, no quotes, no markdown, no prefixes like “Title:” or “Snippet:”.\n"
            "- Use the same language as the user’s last message.\n"
            "- Keep it generic: mention the type of request (e.g., “debugging”, “planning”, “comparison”) not every detail.\n"
            "Example Output:\n"
            '{"title": "Weather Inquiry", "snippet": "User asked about current weather in Tashkent."}\n'
            "STRICT: ONLY OUTPUT JSON."
        )
        raw_output = self.llm.ask(summary_prompt, purpose = "chat summary")

        try:
            # Attempt to parse the LLM's response as real JSON
            if "{" in raw_output:
                match = re.search(r"\{.*\}", raw_output, re.DOTALL)
                if match:
                    return json.loads(match.group(0))

            # Fallback if no JSON found
            return {"title": "Chat", "snippet": "Processing..."}

        except Exception as e:
            print(f"JSON Parse Error: {e}")
            return {"title": "Error", "snippet": "Could not summarize"}

    def auto_update_profile(self, user_input: str, assistant_response: str):
        ALLOWED_PROFILE_FIELDS = {
            "Name",
            "Education",
            "Field_of_Study",
            "Career_Goal",
            "Current_Status",
            "Preferences",
            "Location",
        }

        extraction_prompt = (
            "SYSTEM: You are a strict information extraction engine.\n"
            "You extract STABLE facts about the USER.\n"
            "Return JSON ONLY. No explanations.\n\n"
            "ALLOWED FIELDS:\n"
            "- Name\n"
            "- Education\n"
            "- Field_of_Study\n"
            "- Career_Goal\n"
            "- Current_Status\n"
            "- Preferences\n"
            "- Location\n\n"
            "RULES:\n"
            "- Ignore greetings, questions, emotions, meta talk\n"
            "- Ignore anything about the AI/system\n"
            "- Do NOT guess or invent\n"
            '- If no stable facts exist, return: {"facts": []}\n\n'
            "CONFIDENCE RULES:\n"
            "- 0.95 to 1.00 = explicitly stated by user (e.g. 'My name is X', 'I live in Y').\n"
            "- 0.70 to 0.90 = clearly stated but slightly ambiguous.\n"
            "- 0.50 to 0.69 = implied/inferred -> include it ONLY if still very likely.\n"
            "- below 0.50 = do NOT include.\n\n"
            "OUTPUT FORMAT:\n"
            "{\n"
            '  "facts": [\n'
            '    {"field": "Field", "value": "Value", "confidence": 0.0}\n'
            "  ]\n"
            "}\n\n"
            f"USER INPUT:\n{user_input}"
        )

        if len(user_input.strip()) < 10:
            print(f"{Fore.BLACK}[Observer]: Short input. No checks needed.")
            return
        if self.llm.last_intent in ("CODE", "DEBUG"):
            return

        raw = self.llm.ask( 
            extraction_prompt, 
            purpose = "profile updage"
        ).strip()

        try:
            raw = raw.strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start : end + 1]
            data = json.loads(raw)
        except Exception:
            print(f"{Fore.BLACK}[Observer]: Invalid JSON, ignored.")
            return

        facts = data.get("facts", [])
        if not facts:
            print(f"{Fore.BLACK}[Observer]: No usable facts extracted.")
            return

        merged_any = False

        for fact in facts:
            field = fact.get("field")
            value = fact.get("value")
            confidence = fact.get("confidence", 0.0)

            # Safety filters
            if not field or not value:
                continue
            if confidence < 0.6:
                continue
            if value.strip().endswith("?"):
                continue
            if field not in ALLOWED_PROFILE_FIELDS:
                continue

            if self.memory.merge_profile_fact(field, value, confidence):
                merged_any = True

        if merged_any:
            print(f"{Fore.CYAN}[Observer]: Profile updated (merged).")

    def seed_from_legacy_transcript(self, transcript: str):
        """
        transcript format:
        User: ... | Assistant: ...[TURN_END]
        User: ... | Assistant: ...[TURN_END]
        """
        if not transcript or not isinstance(transcript, str):
            return

        seeded = []
        turns = transcript.split("[TURN_END]")
        for turn in turns:
            turn = turn.strip()
            if not turn:
                continue

            parts = [p.strip() for p in turn.split(" | ") if p.strip()]
            for part in parts:
                if part.startswith("User: "):
                    seeded.append({"role": "user", "content": part[6:]})
                elif part.startswith("Assistant: "):
                    seeded.append({"role": "assistant", "content": part[11:]})

        # replace current short-term with seeded history
        if seeded:
            self.short_term_memory = seeded

    def process_query(self, user_input, mode, images=None, restored_memory = None):
        user_text_original = user_input  # keep clean copy for memory

        # 1) Decide intent FIRST (mode overrides router)
        if mode == "Coder":
            intent = "CODE"
        elif mode == "Assistant":
            intent = "ASSISTANT"
        else:
            intent = self.llm.get_intent(user_input)

        # 2) Handle image (prepare b64 + save file + add instruction to prompt)
        images_b64 = []
        img_tags = []
        model_override = None  # default

        if images and len(images) > 0:
            print(f"{Fore.MAGENTA}[Vision] Image detected. Switching to vision model.")

            is_autonomous = False
            os.makedirs("memory_images", exist_ok=True)

            for img in images:
                img_bytes = img.read()

                # Encode for vision model
                images_b64.append(
                    base64.b64encode(img_bytes).decode("utf-8")
                )

                # Save locally for memory
                img_id = f"{uuid.uuid4()}.png"
                with open(os.path.join("memory_images", img_id), "wb") as f:
                    f.write(img_bytes)

                img_tags.append(f"[IMAGE: {img_id}]")

            # 🔥 HARD OVERRIDE — vision always wins
            model_override = self.llm.vision_model

            # Prevent coder model from hijacking
            intent = "ASSISTANT"

            # Add instruction ONLY for model
            user_input = f"(The user provided {len(images_b64)} image(s). Analyze them carefully.)\n{user_input}"

                # Add instruction for the model ONLY

        # 3) Retrieve context
        context_data = self.memory.get_raw_context(user_input, k=15)
        recent_history = "\n".join(context_data["history"][-6:])
        user_profile = context_data["profile_summary"]

        # 4) Web search logic (same as your code)
        is_forced_search = user_text_original.lower().startswith("/search")
        web_context = ""

        if is_forced_search:
            print(f"{Fore.MAGENTA}Triggered manual web search")
            search_query_input = user_text_original[7:].strip()
            refined_query = self.llm.generate_search_query(self.short_term_memory, search_query_input)
            web_context = self.llm.web_lookup(refined_query)
            intent = "WEB"

        is_autonomous = False

        # Hard priority 1: explicit /search
        if is_forced_search:
            is_autonomous = True

        # Hard priority 2: router selected WEB
        elif intent == "WEB":
            is_autonomous = True
        elif intent in ("CODE", "DEBUG"):
            is_autonomous = False
        # Otherwise use classifier
        else:
            is_autonomous = self.needs_web_search_autonomous(user_input, recent_history)

        if (is_forced_search or is_autonomous) and not web_context:
            search_query = self.llm.generate_search_query(self.short_term_memory, user_input)
            web_context = self.llm.web_lookup(search_query)

        # 5) Prompt
        tashkent_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S GMT+5")
        role = "Expert Programmer" if intent in ("CODE", "DEBUG") else "Personal Assistant"

        system_prompt = (
            "### CORE IDENTITY (READ THIS FIRST) ###\n"
            f"{user_profile if user_profile else 'User is a new contact.'}\n\n"
            "### RECENT HISTORY (LONG-TERM MEMORY SNIPPETS) ###\n"
            f"{recent_history}\n\n"
            "### CURRENT CHAT ###\n"
            f"Summary:{self.summary}, history: {self.short_term_memory}\n\n"
            "### INSTRUCTIONS ###\n"
            f"You are a {role}.\n"
            "You MUST answer in the SAME language as the user's request.\n"
            f"Current Time: {tashkent_now} (Tashkent, Uzbekistan)\n"
            f"Web Results:\n{web_context}\n"
        )

        # 6) LLM call
        response = self.llm.ask(
            user_input,
            system_prompt=system_prompt,
            intent=intent,
            images_b64=images_b64,
            model_override=model_override,
            purpose = "prompt"
        )

        # 7) Save to short-term memory (ALWAYS user then assistant)

        combined_tags = " ".join(img_tags) + " " if img_tags else ""

        self.short_term_memory.append({"role": "user", "content": combined_tags + user_input})
        self.short_term_memory.append({"role": "assistant", "content": response})

        # 8) Summary uses a string transcript
        transcript = "\n".join([f"{m['role']}: {m['content']}" for m in self.short_term_memory])
        self.summary = self.make_chat_summary(transcript)

        # 9) Profile update
        self.auto_update_profile(user_text_original, response)

        model_label = "CODER" if intent in ("CODE", "DEBUG") else "ASSISTANT"
        return {"response": response, "intent": model_label}

assistant = Main()


@app.route("/ask", methods=["POST"])
def ask():
    # -----------------------------
    # 1️⃣ Normalize request input
    # -----------------------------
    is_multipart = (
        request.content_type 
        and request.content_type.startswith("multipart/form-data")
    )

    if is_multipart:
        query = request.form.get("query", "")
        mode = request.form.get("mode", "Auto")
        chat_id = request.form.get("chat_id")
        images = request.files.getlist("image")
    else:
        data = request.json or {}
        query = data.get("query", "")
        mode = data.get("mode", "Auto")
        chat_id = data.get("chat_id")
        images = None

    # -----------------------------
    # 2️⃣ Seed legacy ONCE
    # -----------------------------
    if chat_id:
        try:
            legacy = assistant.memory.history.get(ids=[chat_id])
            if legacy and legacy.get("documents") and legacy["documents"][0]:
                transcript = legacy["documents"][0]

                assistant.seed_from_legacy_transcript(transcript)
                assistant.summary = assistant.make_chat_summary(transcript)

                print(f"{Fore.CYAN}[Legacy seeded: {chat_id}]")
        except Exception as e:
            print(f"[Legacy seed error]: {e}")

    # -----------------------------
    # 3️⃣ Process query
    # -----------------------------
    print(f"{Fore.YELLOW}[Server]: Received {mode} query: {query[:60]}... images={len(images) if images else 0}")

    result = assistant.process_query(query, mode, images=images)

    return jsonify(result)


@app.route("/sum", methods=["POST"])
def summary():
    if not assistant.summary:
        return jsonify({"title": "New chat", "snippet": ""})  # Fixed typo "tittle"
    return jsonify(assistant.summary)


@app.route("/save", methods=["POST"])
def save_session_to_long_term():
    chat_id = request.form.get("chat_id")

    if not assistant.short_term_memory:
        return jsonify({"status": "ignored", "message": "No memory to save"}), 200

    formatted_interactions = []

    # 1. Convert list of dicts into paired interaction strings
    for i in range(0, len(assistant.short_term_memory), 2):
        try:
            user_msg = assistant.short_term_memory[i]["content"]
            ast_msg = (
                assistant.short_term_memory[i + 1]["content"]
                if (i + 1) < len(assistant.short_term_memory)
                else "[No Response]"
            )
            formatted_interactions.append(
                f"User: {user_msg} | Assistant: {ast_msg}[TURN_END]"
            )
        except Exception as e:
            print(f"Error bundling turn: {e}")

    # 2. Join the list into one big string
    full_transcript = "\n".join(formatted_interactions)

    if not full_transcript.strip():
        return jsonify({"status": "error", "message": "Transcript was empty"}), 400

    # 3. Generate summary metadata
    metadata = assistant.make_chat_summary(full_transcript)

    metadata["pinned"] = 0
    metadata["timestamp"] = str(datetime.now())

    # 4. Save to ChromaDB
    if chat_id:
        assistant.memory.history.update(
            ids=[chat_id],
            documents=[full_transcript],
            metadatas=[metadata]
        )
        saved_id = chat_id
    else:
        # create new
        
        saved_id = assistant.memory.add_history(full_transcript, metadata)

    # 5. Clear short-term memory
    assistant.short_term_memory = []
    print(" [System] Session archived to Long-Term Memory.")

    return jsonify({"status": "success", "id": saved_id}), 200


@app.route("/history", methods=["GET"])
def get_history():
    try:
        data = assistant.memory.history.get(include=["metadatas"])
        print(f"DEBUG Chroma Data: {data}")

        history_list = []

        if data and "ids" in data and len(data["ids"]) > 0:
            for i in range(len(data["ids"])):
                raw_meta = data["metadatas"][i] if data.get("metadatas") else None
                meta = raw_meta if raw_meta is not None else {}

                history_list.append({
                    "id": data["ids"][i],
                    "title": meta.get("title","Legacy Chat"),
                    "snippet": meta.get("snippet","No summary available"),
                    "pinned": int(meta.get("pinned",0))
                })

        return jsonify(history_list[::-1])

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify([]), 500


@app.route("/chat/<chat_id>", methods=["GET"])
def get_chat_content(chat_id):
    try:
        result = assistant.memory.history.get(ids=[chat_id])

        if result and result["documents"]:
            return jsonify({"content": result["documents"][0]})
        return jsonify({"error": "Chat not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/prune", methods=["POST"])
def clear():
    print(f"{Fore.CYAN} [Saving memories and shutting down. Goodbye!]")
    assistant.memory.prune_memories()
    assistant.short_term_memory = []
    return "Memory saved"

@app.route("/image/<name>")
def get_image(name):
    return send_from_directory("memory_images", name)

@app.route("/delete", methods=["POST"])
def delete():
    chat_id = request.form.get("id", "")

    ok = assistant.memory.delete_history(chat_id)
    if not ok:
        return jsonify({"status": "error", "message": "Delete failed"}), 400

    return jsonify({"status": "success"}), 200

@app.route("/rename", methods=["POST"])
def rename_chat():
    chat_id = request.form.get("id", "")
    title = request.form.get("title", "")

    ok = assistant.memory.rename_history(chat_id, title)
    if not ok:
        return jsonify({"status": "error"}), 400

    return jsonify({"status": "success"}), 200


@app.route("/pin", methods=["POST"])
def pin_chat():
    chat_id = request.form.get("id", "")

    pinned = assistant.memory.toggle_pin_history(chat_id)
    if pinned is None:
        return jsonify({"status": "error"}), 400

    return jsonify({"status": "success", "pinned": pinned}), 200

if __name__ == "__main__":
    # Explicitly binding to 127.0.0.1 (localhost) on port 5000
    app.run(host="127.0.0.1", port=5000, debug=False)