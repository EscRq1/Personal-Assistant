import gc
import threading
import time
from enum import Enum

import requests
from colorama import Fore


class GPUResource(str, Enum):
    NONE = "none"
    ASSISTANT = "assistant"
    OBSERVER = "observer"
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
        request_timeout: int = 300,
    ):
        self.router_url = router_url.rstrip("/")
        self.assistant_model = assistant_model
        self.observer_model = observer_model
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
        info = self._get_model_info(model_name)

        if not info:
            return False

        status = info.get("status")

        if isinstance(status, dict):
            status = status.get("value")

        return status == "loaded"

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

    def _release_jina(self) -> None:
        if self._jina is None:
            return

        print(
            f"{Fore.YELLOW}"
            "[RESOURCE MANAGER] releasing Jina..."
        )

        try:
            # Drop the reference first.
            self._jina = None

            # Ask Python/PyTorch to release objects and cached CUDA
            # allocations. This is intentionally defensive.
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except ImportError:
                pass

        finally:
            # Give CUDA a brief moment to finish teardown.
            time.sleep(0.5)

        print(
            f"{Fore.GREEN}"
            "[RESOURCE MANAGER] Jina released"
        )

    # ============================================================
    # Resource switching
    # ============================================================

    def acquire(self, resource: GPUResource | str):
        """
        Acquire exclusive GPU ownership.

        Supported resources:
            GPUResource.ASSISTANT
            GPUResource.OBSERVER
            GPUResource.JINA
        """

        if isinstance(resource, str):
            try:
                resource = GPUResource(resource.lower())
            except ValueError as exc:
                raise ValueError(
                    f"Unknown GPU resource: {resource!r}"
                ) from exc

        with self.lock:
            # Fast path.
            if self.current_owner == resource:
                return self._get_current_resource(resource)

            print(
                f"{Fore.CYAN}"
                f"[RESOURCE MANAGER] switching "
                f"{self.current_owner.value} -> {resource.value}"
            )

            # ----------------------------------------------------
            # Release current owner
            # ----------------------------------------------------

            if self.current_owner in (
                GPUResource.ASSISTANT,
                GPUResource.OBSERVER,
            ):
                self._release_llama_models()

            elif self.current_owner == GPUResource.JINA:
                self._release_jina()

            # ----------------------------------------------------
            # Defensive cleanup
            #
            # Don't trust our bookkeeping completely. Make sure
            # nothing from another owner remains.
            # ----------------------------------------------------

            if resource == GPUResource.ASSISTANT:
                self._release_jina()
                self._release_llama_models()

                self._load_llama_model(self.assistant_model)
                self.current_owner = GPUResource.ASSISTANT

                return None

            if resource == GPUResource.OBSERVER:
                self._release_jina()
                self._release_llama_models()

                self._load_llama_model(self.observer_model)
                self.current_owner = GPUResource.OBSERVER

                return None

            if resource == GPUResource.JINA:
                self._release_llama_models()

                jina = self._create_jina()
                self.current_owner = GPUResource.JINA

                return jina

            if resource == GPUResource.NONE:
                self._release_llama_models()
                self._release_jina()

                self.current_owner = GPUResource.NONE
                return None

            raise RuntimeError(
                f"Unhandled GPU resource: {resource}"
            )

    def release(self) -> None:
        """
        Release all GPU-owned resources.
        """

        with self.lock:
            print(
                f"{Fore.CYAN}"
                "[RESOURCE MANAGER] releasing all GPU resources"
            )

            self._release_llama_models()
            self._release_jina()

            self.current_owner = GPUResource.NONE

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

    # ============================================================
    # Context manager helper
    # ============================================================

    class _Lease:
        def __init__(self, manager, resource):
            self.manager = manager
            self.resource = resource

        def __enter__(self):
            return self.manager.acquire(self.resource)

        def __exit__(self, exc_type, exc_value, traceback):
            # Important:
            # We intentionally DO NOT release here.
            #
            # Keeping the current model resident across consecutive
            # calls avoids unnecessary model reloads.
            return False

    def use(self, resource: GPUResource | str):
        """
        Convenience helper:

            with resources.use(GPUResource.OBSERVER):
                ...

        The context manager acquires the resource but does not
        automatically unload it afterward. The next acquire() will
        perform the switch.
        """

        return self._Lease(self, resource)



resource_manager = ResourceManager()