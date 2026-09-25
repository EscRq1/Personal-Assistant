import gc
import threading
import time
import torch
from enum import Enum

import requests
from colorama import Fore


class GPUResource(str, Enum):
    NONE = "none"
    ASSISTANT = "assistant"
    OBSERVER = "observer"
    CODER = "coder"
    JINA = "jina"


class ResourceManager:
    """
    Central owner of GPU-heavy resources.

    Invariant:
        At most one of Assistant, Observer, or Jina owns the GPU
        at any given time.

    llama.cpp models are managed through the existing llama.cpp router.
    Jina is loaded lazily and owned by this manager.
    """

    def __init__(
        self,
        router_url: str = "http://127.0.0.1:8080",
        assistant_model: str = "assistant",
        observer_model: str = "observer",
        coder_model: str = "coder",
        request_timeout: int = 300,
    ):
        self.router_url = router_url.rstrip("/")
        self.assistant_model = assistant_model
        self.observer_model = observer_model
        self.coder_model = coder_model
        self.request_timeout = request_timeout

        self.current_owner = GPUResource.NONE

        # One global lock protects:
        #   resource switching
        #   model load/unload
        #   Jina creation/destruction
        #
        # Use RLock because some public methods call internal helpers
        # that may themselves touch manager state.
        self.lock = threading.RLock()

        # Jina is intentionally lazy-loaded.
        self._jina = None

        print(
            f"{Fore.CYAN}"
            "[RESOURCE MANAGER] initialized"
        )

    # ============================================================
    # State
    # ============================================================

    def owner(self) -> GPUResource:
        return self.current_owner

    # ============================================================
    # llama.cpp router
    # ============================================================

    def _get_models(self) -> list:
        response = requests.get(
            f"{self.router_url}/models",
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()
        models = data.get("data", data)

        if not isinstance(models, list):
            raise RuntimeError(
                "Unexpected /models response from llama.cpp router"
            )

        return models

    def _get_model_info(self, model_name: str):
        for model in self._get_models():
            if model.get("id") == model_name:
                return model

        return None

    def _is_loaded(self, model_name: str) -> bool:
        
        def getStatus():

            info = self._get_model_info(model_name)

            if not info:
                return False
            status = info.get("status")

            if isinstance(status, dict):
                status = status.get("value")

            return status
        status = getStatus()
        if status not in ["loaded","loading"]:
            return False
        elif status == "loading":
            deadline = time.monotonic() + self.request_timeout

            

            while time.monotonic() < deadline:
                status = getStatus()
                if status == "loaded":
                    return True

                if status != "loading":
                    return False

                time.sleep(0.5)

            raise TimeoutError(
                f"Timed out waiting for {model_name} to load"
            )

        return True

    def _unload_llama_model(self, model_name: str) -> None:
        print(
            f"{Fore.YELLOW}"
            f"[RESOURCE MANAGER] unloading llama model: {model_name}"
        )

        response = requests.post(
            f"{self.router_url}/models/unload",
            json={"model": model_name},
            timeout=30,
        )
        response.raise_for_status()

        deadline = time.monotonic() + 30.0

        while time.monotonic() < deadline:
            info = self._get_model_info(model_name)

            if info is None:
                print(
                    f"{Fore.GREEN}"
                    f"[RESOURCE MANAGER] {model_name} fully unloaded"
                )
                return

            status = info.get("status")

            if isinstance(status, dict):
                status = status.get("value")

            if status == "unloaded":
                print(
                    f"{Fore.GREEN}"
                    f"[RESOURCE MANAGER] {model_name} fully unloaded"
                )
                return

            time.sleep(0.2)

        raise TimeoutError(
            f"Timed out waiting for '{model_name}' to unload"
        )
    def _load_llama_model(self, model_name: str) -> None:
        print(
            f"{Fore.YELLOW}"
            f"[RESOURCE MANAGER] loading llama model: "
            f"{model_name}"
        )

        response = requests.post(
            f"{self.router_url}/models/load",
            json={"model": model_name},
            timeout=30,
        )
        response.raise_for_status()

        deadline = time.monotonic() + self.request_timeout

        while time.monotonic() < deadline:
            try:
                if self._is_loaded(model_name):
                    print(
                        f"{Fore.GREEN}"
                        f"[RESOURCE MANAGER] {model_name} loaded"
                    )
                    return

            except requests.RequestException as exc:
                print(
                    f"{Fore.YELLOW}"
                    f"[RESOURCE MANAGER] waiting for "
                    f"{model_name}: {exc}"
                )

            time.sleep(0.5)

        raise TimeoutError(
            f"Timed out waiting for llama.cpp model "
            f"'{model_name}' to load."
        )

    def _release_llama_models(self) -> None:
        """
        Ensure both llama.cpp models are unloaded.

        This is intentionally based on router state instead of only
        self.current_owner, because the router is the actual source
        of truth.
        """

        for model_name in (
            self.assistant_model,
            self.observer_model,
            self.coder_model,
        ):
            try:
                if self._is_loaded(model_name):
                    self._unload_llama_model(model_name)
            except requests.RequestException as exc:
                print(
                    f"{Fore.RED}"
                    f"[RESOURCE MANAGER] error checking/unloading "
                    f"{model_name}: {exc}"
                )
                raise

        # Give llama.cpp time to tear down CUDA allocations.
        time.sleep(0.5)

    # ============================================================
    # Jina
    # ============================================================

    def _create_jina(self):
        """
        Create the Jina reranker lazily.

        Import is deliberately local so importing the application
        does not immediately initialize a GPU model.
        """

        if self._jina is not None:
            return self._jina

        print(
            f"{Fore.YELLOW}"
            "[RESOURCE MANAGER] loading Jina reranker..."
        )

        from jina_reranker import JinaReranker

        self._jina = JinaReranker(
            device="cuda",
            max_query_length=256,
            max_doc_length=2048,
        )

        print(
            f"{Fore.GREEN}"
            "[RESOURCE MANAGER] Jina loaded"
        )

        return self._jina

    def _release_jina(self):
        if self._jina is None:
            return

        print("[RESOURCE MANAGER] Releasing Jina reranker...")

        try:
            self._jina.release()
        finally:
            self._jina = None
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            time.sleep(0.5)

        print("[RESOURCE MANAGER] Jina released")


        print(
            f"{Fore.GREEN}"
            "[RESOURCE MANAGER] Jina released"
        )
    def _allocate_assistant(self):
        # If Assistant already exists, simply adopt it.
        if self._is_loaded(self.assistant_model):
            self.current_owner = GPUResource.ASSISTANT
            print(
                f"{Fore.GREEN}"
                "[RESOURCE MANAGER] assistant already loaded"
            )
            return

        # Router state is authoritative if Python state is stale.
        if self._is_loaded(self.observer_model) or self._is_loaded(self.coder_model):
            self._release_llama_models()

        self._load_llama_model(self.assistant_model)
        self.current_owner = GPUResource.ASSISTANT

    def _allocate_observer(self):
        # If Observer already exists, simply adopt it.
        if self._is_loaded(self.observer_model):
            self.current_owner = GPUResource.OBSERVER
            print(
                f"{Fore.GREEN}"
                "[RESOURCE MANAGER] observer already loaded"
            )
            return

        # Router state is authoritative if Python state is stale.
        if self._is_loaded(self.assistant_model) or self._is_loaded(self.coder_model):
            self._release_llama_models()

        self._load_llama_model(self.observer_model)
        self.current_owner = GPUResource.OBSERVER

    def _allocate_coder(self):
        if self._is_loaded(self.coder_model):
            self.current_owner = GPUResource.CODER
            print(
                f"{Fore.GREEN}"
                "[RESOURCE MANAGER] coder already loaded"
            )
            return
        
        if self._is_loaded(self.observer_model) or self._is_loaded(self.assistant_model):
            self._release_llama_models()
        
        self._load_llama_model(self.coder_model)
        self.current_owner = GPUResource.CODER

    def _allocate_jina(self):
        # Router is the source of truth.
        # Jina cannot coexist with a loaded llama.cpp model.
        if any(self._is_loaded(model) for model in [self.assistant_model, self.coder_model, self.observer_model]):
            self._release_llama_models()

        jina = self._create_jina()

        self.current_owner = GPUResource.JINA

        return jina

    def _release_current(self):
        # Release whatever is actually resident.
        self._release_llama_models()

        if self._jina is not None:
            self._release_jina()

        self.current_owner = GPUResource.NONE

    def reconcile(self):
        """
        Synchronize Python-side ownership with the llama.cpp router.

        Call once after the router is ready and before serving requests.
        """
        with self.lock:
            assistant_loaded = self._is_loaded(self.assistant_model)
            observer_loaded = self._is_loaded(self.observer_model)
            coder_loaded = self._is_loaded(self.coder_model)

            loaded_resources = [
                resource
                for resource, loaded in (
                    (GPUResource.ASSISTANT, assistant_loaded),
                    (GPUResource.OBSERVER, observer_loaded),
                    (GPUResource.CODER, coder_loaded),
                )
                if loaded
            ]

            if len(loaded_resources) > 1:
                raise RuntimeError(
                    "Invalid router state: multiple llama.cpp models are loaded: "
                    + ", ".join(resource.value for resource in loaded_resources)
                )

            if assistant_loaded:
                self.current_owner = GPUResource.ASSISTANT
            elif observer_loaded:
                self.current_owner = GPUResource.OBSERVER
            elif coder_loaded:
                self.current_owner = GPUResource.CODER
            elif self._jina is not None:
                self.current_owner = GPUResource.JINA
            else:
                self.current_owner = GPUResource.NONE

            print(
                f"{Fore.CYAN}"
                f"[RESOURCE MANAGER] reconciled owner: "
                f"{self.current_owner.value}"
            )

    # ============================================================
    # Resource switching
    # ============================================================

    def acquire(self, resource: GPUResource | str):
        if isinstance(resource, str):
            try:
                resource = GPUResource(resource.lower())
            except ValueError as exc:
                raise ValueError(
                    f"Unknown resource: {resource!r}"
                ) from exc

        with self.lock:

            # ------------------------------------------------------------
            # Router is the source of truth.
            # If the requested resource is already resident,
            # adopt it instead of releasing/reloading it.
            # ------------------------------------------------------------

            if resource == GPUResource.ASSISTANT:
                if self._is_loaded(self.assistant_model):
                    self.current_owner = GPUResource.ASSISTANT
                    return

            elif resource == GPUResource.OBSERVER:
                if self._is_loaded(self.observer_model):
                    self.current_owner = GPUResource.OBSERVER
                    return
            elif resource == GPUResource.CODER:
                if self._is_loaded(self.coder_model):
                    self.current_owner = GPUResource.CODER
                    return
            elif resource == GPUResource.JINA:
                if (
                    self.current_owner == GPUResource.JINA
                    and self._jina is not None
                ):
                    return

            elif resource == GPUResource.NONE:
                self._release_current()
                return

            # ------------------------------------------------------------
            # Requested resource is not already resident.
            # Perform the actual switch.
            # ------------------------------------------------------------

            self._release_current()

            if resource == GPUResource.ASSISTANT:
                return self._allocate_assistant()

            if resource == GPUResource.OBSERVER:
                return self._allocate_observer()
            if resource == GPUResource.CODER:
                return self._allocate_coder()

            if resource == GPUResource.JINA:
                return self._allocate_jina()

            raise RuntimeError(
                f"Unhandled resource: {resource}"
            )

    def get_jina(self):
        """
        Return the currently resident Jina instance.

        This intentionally does NOT acquire Jina automatically.
        Call acquire(GPUResource.JINA) first.
        """

        with self.lock:
            if self.current_owner != GPUResource.JINA:
                raise RuntimeError(
                    "Jina is not the current GPU owner"
                )

            if self._jina is None:
                raise RuntimeError(
                    "Jina resource is marked active but not initialized"
                )

            return self._jina


resource_manager = ResourceManager()

