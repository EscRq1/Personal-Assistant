import os

import requests
import numpy as np

from tools.web_search import search_web, fetch_page_text, is_online
from colorama import Fore, Style, init


PLANNER_SYSTEM_PROMPT = ("""
        You are the TASK PLANNER of a multi-agent AI system.

        Your job is NOT to answer the user.

        Your job is to break the user's request into a minimal sequence of logical steps.

        The agent responsible for execution has already been selected by the system.

        You only produce the ordered task plan.

        Rules:
        - Never answer the user.
        - Never generate explanations or code.
        - Only produce the plan.
        - Keep the plan minimal and logical.
        - Return ONLY valid JSON.

        Output format:

        {
        "steps": [
            {
            "step_id": 1,
            "task": "description of the step"
            }
        ],
        "notes": "short explanation of the plan"
        }

        Example:

        User request:
        Fix the bug in my Python script.

        Output:

        {
        "steps": [
            {
            "step_id": 1,
            "task": "Read the Python file"
            },
            {
            "step_id": 2,
            "task": "Identify the bug in the code"
            },
            {
            "step_id": 3,
            "task": "Generate a corrected version of the code"
            }
        ],
        "notes": "Bug fixing requires analyzing the code and generating a fix."
        }                  
        """)


class Controller:
    def __init__(self, model_name="qwen3:7b"):
        self.assistant_model = "phi4-mini:latest"
        self.planner_model = "deepseek-r1:8b" # Generalist
        self.vision_model = "llava:13b"
        self.coder_model = "qwen3-coder:30b"  # Coding specialist
        self.generate_url = "http://localhost:11434/api/generate"
        self.chat_url = "http://localhost:11434/api/chat"
        self.last_intent = None

        self.temps = {
            "ASSISTANT": 0.8,  # 0.8 = More "wit" and conversational flow
            "CODER": 0.2,  # 0.2 = Strict syntax, less "hallucinated" code
        }

        self.context_size = {
            "ASSISTANT": 8192,
            "CODER": 16384,
        }

    def _call_ollama(self, model, prompt, system_prompt=None,
                 target_temp=0, target_context=8192, images_b64=None):

        is_vision = images_b64 and len(images_b64) > 0

        if is_vision:
            print("USING CHAT FORMAT FOR VISION")

            messages = []

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            messages.append({
                "role": "user",
                "content": prompt,
                "images": images_b64
            })

            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # lower temp for vision
                    "num_ctx": target_context,
                    "low_vram": True,
                }
            }

        else:
            # Normal text-only path
            if system_prompt:
                final_prompt = (
                    f"### SYSTEM INSTRUCTIONS ###\n{system_prompt}\n\n"
                    f"### USER INPUT ###\n{prompt}"
                )
            else:
                final_prompt = prompt

            payload = {
                "model": model,
                "prompt": final_prompt,
                "stream": False,
                "options": {
                    "temperature": target_temp,
                    "num_ctx": target_context,
                    "low_vram": True,
                }
            }

        if is_vision:
            response = requests.post(self.chat_url, json=payload)
        else:
            response = requests.post(self.generate_url, json=payload)
        response.raise_for_status()

        print(f"{Fore.CYAN}MODEL USED: {model}")

        data = response.json()

        # Text-only generate format
        if "response" in data:
            return data["response"]

        # Chat format (vision)
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]

        return ""
    
    
    def ask(self, prompt: str,  purpose: str, system_prompt=None, intent="PLANNER",
        images_b64=None, model_override=None) -> str:
        plan = None

        # If images exist → force vision model
        if intent == "PLANNER":
            model = self.planner_model
            temp = 0
            context = 4096
        else:
            if model_override:
                model = self.vision_model
                temp = self.temps["ASSISTANT"]
                context = self.context_size["ASSISTANT"]

            elif intent in ("CODE", "DEBUG"):
                model = self.coder_model
                temp = self.temps["CODER"]
                context = self.context_size["CODER"]
                plan = self._call_ollama(self.planner_model, prompt, system_prompt=PLANNER_SYSTEM_PROMPT, target_temp=0, target_context= 4096)

            elif intent in ("LOGIC", "MATH", "TECHNICAL"):
                model = self.planner_model
                temp = self.temps["ASSISTANT"]
                context = self.context_size["ASSISTANT"]
            else:
                model = self.assistant_model
                temp = self.temps["ASSISTANT"]
                context = self.context_size["ASSISTANT"]  

        if plan:
            print(f"Plan: {plan}")

            prompt = f"""
                Execution Plan:
                {plan}

                User Request:
                {prompt}
            """
            
        print(f"MODEL USED: {model}, PURPOSE: {purpose}")
        print("IMAGES COUNT:", len(images_b64) if images_b64 else 0)

        return self._call_ollama(
            model,
            prompt,
            system_prompt,
            target_temp=temp,
            target_context=context,
            images_b64=images_b64  # DO NOT wrap in conditional
        )
    
    def generate_search_query(self, history: list, user_input: str) -> str:
        """
        Uses conversation history to turn 'search for it' into 'Java 25 features'
        """
        # Create a text block of the last 3 turns
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-3:]])

        prompt = (
            f"Conversation History:\n{context}\n\n"
            f"User's new command: '{user_input}'\n\n"
            "Task: The user wants to search the web. Based on the history and command, "
            "write the best single search query to put into a search engine. "
            "Output ONLY the search query, nothing else."
        )

        query = self.ask(prompt, purpose="generate web query").strip().strip('"').strip("'")
        print(f"DEBUG: Generated Search Query -> {query}")  # Helpful for debugging
        return query

    def web_lookup(self, query: str) -> str:
        # 1. Check if we even have internet before trying
        if not is_online():
            print(" [Offline] Skipping web search... using local brain.")
            return ""

        print(f"DEBUG: Searching web for -> {query}")
        context_chunks = []

        try:
            # 2. Perform the search
            search_results = search_web(query, max_results=3)

            for r in search_results:
                url = r.get("href") or r.get("url")
                print(f"DEBUG: Fetching -> {url}")

                text = fetch_page_text(url)
                print(f"DEBUG: Extracted length -> {len(text)}")

                if text:
                    context_chunks.append(f"Source ({url}):\n{text}")

        except Exception as e:
            # Catching DDGSException or Timeout errors here
            print(f" [Error] Web search failed: {e}")
            return ""

        return "\n\n".join(context_chunks)


class SemanticController(Controller):
    def __init__(self):
        super().__init__()
        # 1. Configuration
        self.embed_url = "http://localhost:11434/api/embed"
        self.embed_model = "nomic-embed-text"
        self.cache_file = "intent_anchors.npy"

        # Prototype examples for each intent
        self.raw_anchors = {
            "CODE": [
                "write code for", "implement", "create a function", "build a script",
                "generate a class", "produce working code", "give me the code",
                "write unit tests", "add a feature", "extend this code",
                "refactor this", "clean this up", "simplify the code",
                "make it faster", "optimize performance", "reduce memory usage",
                "add type hints", "convert to async", "handle edge cases",
                "convert this to typescript", "port to rust", "rewrite in go",
                "now in python", "now in c++", "now in javascript", "cpp", "java",
                "javascript", "js", "continue the implementation", "finish the code", 
                "same thing but", "based on the code above", "use the previous code",
            ],
            "DEBUG": [
                "it crashes", "error message", "stack trace", "exception thrown",
                "segmentation fault", "runtime error", "compile error", "linker error",
                "module not found", "import error", "permission denied",
                "connection refused", "timeout error", "debug this", "help me fix", 
                "root cause", "why is this happening", "minimal repro", 
                "steps to reproduce", "patch this", "Traceback (most recent call last)", 
                "SyntaxError", "TypeError", "UnhandledPromiseRejection", "npm ERR", 
                "pip install fails", "docker build failed", "github actions failed",
            ],
            "TECHNICAL": [
                "how do i install", "setup", "configure", "set up the environment",
                "create virtualenv", "venv", "conda env", "poetry setup",
                "requirements.txt", "pyproject.toml", "package.json",
                "npm install", "pnpm", "yarn", "pip install",
                "build tools", "compiler toolchain", "cmake configure",
                "set up an api", "api key", "authentication", "oauth2",
                "jwt", "bearer token", "refresh token", "cors", "rate limiting", 
                "api gateway", "webhook", "callback url", "ngrok", "reverse proxy",
                "openapi", "swagger", "postman collection", "create a fastapi project", 
                "setup flask", "setup django", "create express app", "nestjs setup",
                "setup nextjs", "setup react", "vite project", "electron setup", 
                "create a backend", "create a frontend", "dockerfile", "docker compose", 
                "kubernetes", "helm chart", "deploy to aws", "deploy to gcp", 
                "deploy to azure", "systemd service", "nginx config", "apache config",
                "github actions", "ci pipeline", "cd pipeline", "environment variables", 
                ".env file", "secrets management", "connect to postgres", "mysql connection", 
                "mongodb setup", "redis setup", "migrations", "database url", "ports", 
                "firewall", "dns", "ssl certificate", "https setup", "letsencrypt", 
                "domain configuration",
            ],
            "MATH": [
                "solve this", "simplify", "step by step", "find the derivative", 
                "differentiate", "integrate", "limits", "series", "taylor expansion",
                "linear algebra", "matrix", "matrices", "eigenvalues", "eigenvectors",
                "determinant", "rank", "null space", "svd", "vector space", "basis", 
                "orthogonal", "projection", "probability", "conditional probability", 
                "bayes theorem", "random variable", "expectation", "variance",
                "distribution", "normal distribution", "bernoulli", "hypothesis testing", 
                "p value", "confidence interval", "regression", "maximum likelihood",
                "gradient descent", "convex", "lagrange multipliers", "loss function", 
                "backpropagation", "jacobian", "hessian",
            ],
            "LOGIC": [
                "prove", "proof", "show that", "if and only if", "iff",
                "implies", "therefore", "contradiction", "contrapositive",
                "logical equivalence", "tautology", "satisfiable", "unsatisfiable",
                "predicate logic", "propositional logic", "quantifiers",
                "forall", "exists", "∀", "∃", "discrete math", "combinatorics", 
                "permutations", "combinations", "graph theory", "graphs", "trees", 
                "shortest path", "induction proof", "mathematical induction",
                "pigeonhole principle", "invariant", "big o", "big-o", "time complexity", 
                "space complexity", "recurrence relation", "master theorem",
                "runtime", "asymptotic", "np", "np-complete", "reduction",
                "correctness proof",
            ],
            "WEB": [
                "search the web for", "look it up", "find sources",
                "latest update", "current version of", "release notes for",
                "compare", "benchmarks", "pricing", "requirements",
                "check official docs", "search github issues",
            ],
            "ASSISTANT": [
                "explain this", "explain concept", "explain simply",
                "help me understand", "teach me", "walk me through",
                "give an overview", "tell me about", "what is",
                "how does this work", "why does this happen",

                "general explanation", "background information",
                "describe this topic", "clarify this idea",
                "explain step by step", "simple explanation",

                "give advice", "recommendations", "suggestions",
                "help me decide", "what do you think about",
                "share your thoughts", "opinion on",

                "best practices", "roadmap", "learning path",
                "how should i start", "where should i begin",
                "tips for learning",

                "summarize", "summary", "simplify this",
                "make a checklist", "break it down",

                "general question", "curiosity question",
                "talk about this topic", "discuss this idea",

                "thank you", "thanks", "got it", "makes sense",
                "ok", "okay", "cool", "nice", "interesting"
            ]
        }

        # 2. Boot Optimization (Persistence)
        if os.path.exists(self.cache_file):
            print(" [System] Loading semantic anchors from cache...")
            self.anchor_vectors = np.load(self.cache_file, allow_pickle=True).item()
        else:
            print(" [System] No cache found. Generating new anchors...")
            self._generate_and_save_anchors()

        self.last_intent = "ASSISTANT"  # Default context

    def _get_embeddings(self, input_data):
        """Standardized wrapper for Ollama's /api/embed endpoint."""
        # Ollama expects 'input' to be a string or a list of strings
        payload = {"model": self.embed_model, "input": input_data}
        try:
            response = requests.post(self.embed_url, json=payload, timeout=10)
            response.raise_for_status()
            vectors = response.json().get("embeddings")

            if not vectors:
                print(f" [!] API returned no embeddings for: {str(input_data)[:20]}...")
                return None
            return vectors
        except Exception as e:
            print(f" [!] Embedding Error: {e}")
            return None

    def _generate_and_save_anchors(self):
        self.anchor_vectors = {}
        for intent, phrases in self.raw_anchors.items():
            print(f" -> Creating centroid for {intent}...")
            vectors = self._get_embeddings(phrases)

            if vectors:
                # Calculate the "Centroid" (Average vector for this category)
                centroid = np.mean(vectors, axis=0)
                # Validation: Magnitude check
                mag = np.linalg.norm(centroid)
                if mag == 0:
                    print(f" [!!!] Warning: {intent} centroid is a zero vector!")

                self.anchor_vectors[intent] = centroid
            else:
                self.anchor_vectors[intent] = np.zeros(768)

        np.save(self.cache_file, self.anchor_vectors)
        print(f" [Success] Cached anchors to {self.cache_file}")

    def _cosine_similarity(self, a, b):
        """Calculates distance in vector space."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def get_intent(self, user_input) -> str:
        # Get vector for user input (returns a list of lists, we take the first)
        result = self._get_embeddings(user_input)
        if not result:
            return self.last_intent

        input_vec = result[0]

        # Calculate similarity scores against our anchor centroids
        scores = {
            intent: self._cosine_similarity(input_vec, anchor)
            for intent, anchor in self.anchor_vectors.items()
        }

        # Find winner
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]

        # --- Threshold Logic ---
        # If the confidence is too low, we stick to the last intent
        # (This handles follow-ups like "make it better" or "thanks")
        if confidence < 0.45:
            print(f" {Fore.RED}[Router] Ambiguous ({confidence:.2f}). Defaulting to: {self.last_intent}")
            return self.last_intent

        self.last_intent = best_intent
        print(f"{Fore.GREEN} [Router] Selected: {best_intent} ({confidence:.2f})")
        return best_intent