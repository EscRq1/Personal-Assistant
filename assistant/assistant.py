import requests
import json
import time
import threading

from tools.web_search import search_web, fetch_page_text, is_online
from colorama import Fore
from resource_manager import resource_manager


class Controller:

    def __init__(self):
        # Router model aliases from models.ini
        self.assistant_model = "assistant"
        self.support_model = "observer"

        self.router_url = "http://127.0.0.1:8080"

        self.chat_endpoint = (
            f"{self.router_url}/v1/chat/completions"
        )

        self.temp = 0.8

        # Track what we believe is loaded.
        # Router itself is still the source of truth.

        self.request_timeout = 300
        # The router is configured with --models-max 1, so model switches and
        # inference requests must be serialized to avoid unloading a model
        # while another request is still using it.
        self.resources = resource_manager

    def parse_vision_output(self, raw):
        try:
            data = json.loads(raw)

            if not isinstance(data, dict):
                raise ValueError("Vision output is not an object")

            for key, value in data.items():
                if not key.startswith("img"):
                    raise ValueError(f"Unexpected key: {key}")
                if not isinstance(value, str):
                    raise ValueError(f"{key} is not a string")

            return data

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[VISION] Invalid JSON: {e}")
            return None

    # ============================================================
    # Assistant streaming
    # ============================================================

    def callAssistant(
        self,
        prompt,
        target_temp=0.8,
        system_prompt=None,
        images_b64=None,
    ):  #lock the current call so no other request can interrupt
        with self.resources.lock:
            #acquire the vram
            self.resources.acquire("assistant")

            messages = []

            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt,
                })

            messages.append({
                "role": "user",
                "content": prompt,
            })


            payload = {
                "model": self.assistant_model,
                "messages": messages,
                "temperature": target_temp,
                "max_tokens": -1,
                "stream": True,
                "cache_prompt": True,
            }

            print(
                f"{Fore.CYAN}"
                f"MODEL USED: Assistant"
            )

            with requests.post(
                self.chat_endpoint,
                json=payload,
                stream=True,
                timeout=self.request_timeout,
            ) as response:

                response.raise_for_status()

                for line in response.iter_lines():

                    if not line:
                        continue

                    line = line.decode("utf-8")

                    if not line.startswith("data: "):
                        continue

                    data = line[6:]

                    if data == "[DONE]":
                        break

                    chunk = json.loads(data)

                    token = (
                        chunk["choices"][0]
                        .get("delta", {})
                        .get("content")
                    )

                    if token:
                        yield token


    def callObserver(self, prompt, system_prompt=None, images_b64=None):
        #same lock
        with self.resources.lock:
            #acquire the vram
            self.resources.acquire("observer")


            messages = []

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": "\n".join(system_prompt)
                        if isinstance(system_prompt, (list, tuple))
                        else system_prompt,
                })

            if images_b64:

                content = [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]

                for img in images_b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:image/jpeg;base64,{img}"
                            )
                        },
                    })

                messages.append({
                    "role": "user",
                    "content": content,
                })

            else:

                messages.append({
                    "role": "user",
                    "content": prompt,
                })

            payload = {
                "model": self.support_model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": -1,
                "stream": False,
                "cache_prompt": True,
            }

            print(
                f"{Fore.CYAN}"
                f"MODEL USED: Observer"
            )

            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=self.request_timeout,
            )

            response.raise_for_status()

            data = response.json()

            message = data["choices"][0]["message"]

            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""


            if content.strip():
                return content

            if reasoning.strip():
                return reasoning

            return ""


    def ask(
        self,
        prompt: str,
        purpose: str,
        system_prompt=None,
        images_b64=None,
        model=None,
    ) -> str:

        model = model or "Observer"
        images_desc = {}
        print(
            f"MODEL USED: {model}, "
            f"PURPOSE: {purpose}"
        )

        print(
            "IMAGES COUNT:",
            len(images_b64) if images_b64 else 0,
        )
        if images_b64 and len(images_b64) > 0:
            print("[DEBUG]:Calling vision model")
            VISION_INSTRUCTIONS = (
                "You are a vision analysis module operating as a perception component for another language model.",
                "Your ONLY task is to analyze the provided images and produce concise, accurate descriptions for downstream reasoning.",

                "Do NOT answer the user's question.",
                "Do NOT provide advice, conclusions, or solutions.",
                "Do NOT engage in conversation.",
                "Do NOT ask questions.",
                "Do NOT mention that you are an AI or vision model.",
                "Do NOT invent information that cannot be determined from the image.",

                "Be concise. Prefer the shortest description that preserves all information useful for downstream reasoning.",
                "Most images should require fewer than 150 tokens.",
                "Only unusually information-dense images should exceed 300 tokens.",
                "Never exceed 500 tokens for a single image.",
                "Do not describe irrelevant visual or aesthetic details.",

                "For every image, first classify its primary content:",
                "1. GENERAL IMAGE / PHOTOGRAPH",
                "2. CODE / PROGRAMMING",
                "3. DOCUMENT / TEXT",
                "4. SCREENSHOT / SOFTWARE INTERFACE",
                "5. DIAGRAM / CHART / TECHNICAL FIGURE",
                "6. MIXED CONTENT",

                "=== GENERAL IMAGE / PHOTOGRAPH ===",
                "Describe only information relevant to understanding the scene.",
                "Include the main subjects, important objects, actions, spatial relationships, environment, and notable visual details.",
                "Mention people only when their appearance, action, position, or number is relevant.",
                "Do not waste tokens on generic aesthetic descriptions.",

                "=== CODE / PROGRAMMING ===",
                "Prioritize programming information over visual appearance.",
                "Identify the programming language when possible.",
                "Transcribe relevant visible code, commands, filenames, paths, functions, variables, errors, warnings, and compiler output.",
                "Preserve important code structure.",
                "Do not rewrite, optimize, or correct the code.",
                "If text is unclear, explicitly indicate the uncertainty instead of guessing.",
                "Do not transcribe irrelevant or repeated code.",

                "=== DOCUMENT / TEXT ===",
                "Prioritize information-bearing text.",
                "Extract relevant titles, headings, paragraphs, lists, tables, dates, numbers, labels, and values.",
                "Preserve document structure when useful.",
                "Transcribe important text accurately.",
                "Do not reproduce large amounts of text when a concise extraction preserves the relevant information.",
                "If important text is unreadable, explicitly state that.",

                "=== SCREENSHOT / SOFTWARE INTERFACE ===",
                "Prioritize the software state and information relevant to diagnosing or understanding it.",
                "Identify the application when possible.",
                "Include important visible controls, selected options, fields, values, dialogs, errors, warnings, status indicators, commands, and output.",
                "For terminal screenshots, prioritize commands, paths, errors, warnings, and relevant output.",
                "Ignore irrelevant interface decoration.",

                "=== DIAGRAM / CHART / TECHNICAL FIGURE ===",
                "Prioritize the underlying technical information.",
                "Describe important labels, components, nodes, connections, arrows, axes, values, legends, and relationships.",
                "Explain the structure or data flow only when it is directly visible.",
                "Do not focus on decorative styling.",

                "=== MIXED CONTENT ===",
                "Identify the relevant content types and apply the appropriate description method to each.",
                "Prioritize information useful for downstream reasoning.",

                "=== UNCERTAINTY ===",
                "Never fabricate missing information.",
                "Clearly distinguish visible information from inference.",
                "If something is blurry, cropped, obscured, or ambiguous, explicitly state the uncertainty.",
                "Do not guess unreadable text.",

                "=== OUTPUT FORMAT ===",
                "Your entire response MUST be valid JSON.",
                "Create exactly one key for every input image.",
                "Use image order: img1, img2, img3, etc.",
                "Each value must be a concise string containing the description of that image.",

                "Required format:",
                "{",
                '  "img1": "description of image 1",',
                '  "img2": "description of image 2"',
                "}",

                'For one image: { "img1": "description of image 1" }',

                "Do NOT output Markdown.",
                "Do NOT use JSON code fences.",
                "Do NOT output anything before or after the JSON object.",
                "Do NOT add keys other than img1, img2, img3, etc.",
                "Ensure the output is valid JSON.",
                "Escape quotation marks, backslashes, and newlines correctly.",
                "Do not reason aloud.",
                "Do not explain your process.",
                "Do not include analysis, planning, drafts, or reasoning.",
                "Output ONLY the JSON object."
            )
            user_prompt = (
                f"User provided the following text with the image: {prompt}"
            )

            raw_desc = self.callObserver(
                prompt=user_prompt,
                system_prompt=VISION_INSTRUCTIONS,
                images_b64=images_b64
            )

            images_desc = self.parse_vision_output(raw_desc)

        
            vision_context = "\n".join(
                f"{img}: {desc}"
                for img, desc in images_desc.items()
            )
            if prompt and len(prompt) > 0:
                user_instruction = prompt.strip()
            else:
                user_instruction = (
                    "The user provided image(s) without additional text. "
                    "Analyze the image content and respond naturally based on what is visible. "
                )

            final_prompt = (
                f"{user_instruction}\n\n"
                f"=== USER-PROVIDED IMAGES ===\n"
                f"{vision_context}"
            )
        else:
            final_prompt = prompt

        if model == "Assistant":

            return self.callAssistant(
                prompt=final_prompt,
                target_temp=self.temp,
                system_prompt=system_prompt,
                
            )

        return self.callObserver(
            prompt=prompt,
            system_prompt=system_prompt,
            images_b64=images_b64,
        )

    def generate_search_query(
        self,
        history: list,
        user_input: str,
        memory_context: str = "",
    ) -> str:
        """
        Uses conversation history to turn 'search for it'
        into a useful search query.
        """

        context = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in history[-3:]
        )

        prompt = (
            f"Conversation History:\n{context}\n\n"
            f"Relevant long-term memory:\n{memory_context[:6000]}\n\n"
            f"User's new command: '{user_input}'\n\n"
            "Task: The user wants to search the web. "
            "Based on the history and command, write the best "
            "single search query to put into a search engine. "
            "You are STRICTLY PROHIBITED to answer the user's prompt"
            "Output ONLY the search query, nothing else."
        )

        query = self.ask(
            prompt,
            purpose="generate web query"
        ).strip().strip('"').strip("'")

        print(f"DEBUG: Generated Search Query -> {query}")

        return query


    def web_lookup(self, query: str) -> str:

        if not is_online():
            print(
                " [Offline] Skipping web search... "
                "using local brain."
            )
            return ""

        print(f"DEBUG: Searching web for -> {query}")

        context_chunks = []

        try:
            search_results = search_web(query, max_results=3)

            for r in search_results:

                url = r.get("href") or r.get("url")

                print(f"DEBUG: Fetching -> {url}")

                text = fetch_page_text(url)

                print(f"DEBUG: Extracted length -> {len(text)}")

                if text:
                    # External pages are untrusted evidence. Cap each page
                    # and the total context so a single page cannot dominate
                    # the system prompt.
                    safe_text = text[:8000]
                    context_chunks.append(
                        f"Source ({url}):\n{safe_text}"
                    )

                if sum(len(x) for x in context_chunks) >= 20000:
                    break

        except Exception as e:
            print(f" [Error] Web search failed: {e}")
            return ""

        return "\n\n".join(context_chunks)