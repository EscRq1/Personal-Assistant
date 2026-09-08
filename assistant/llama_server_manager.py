import os
import socket
import subprocess
import threading
import time


class LlamaRouterManager:

    def __init__(self):
        self.server_path = (
            "/home/escrq/Desktop/llama.cpp/build/bin/llama-server"
        )

        self.lib_dir = (
            "/home/escrq/Desktop/llama.cpp/build/bin"
        )

        self.models_preset = (
            "/home/escrq/Desktop/Assistant/models.ini"
        )

        self.host = "127.0.0.1"
        self.port = 8080

        self.process = None
        self.lock = threading.RLock()

    # ============================================================
    # Start
    # ============================================================

    def start(self):

        with self.lock:

            if (
                self.process is not None
                and self.process.poll() is None
            ):
                return

            env = os.environ.copy()

            # ----------------------------------------------------
            # Flask is launched with:
            #
            #     CUDA_VISIBLE_DEVICES=1
            #
            # We DO NOT want that restriction inherited by
            # llama-router, because the router needs all 3 GPUs.
            # ----------------------------------------------------

            env["CUDA_VISIBLE_DEVICES"] = "0,1,2"

            env["LD_LIBRARY_PATH"] = (
                f"{self.lib_dir}:"
                f"{env.get('LD_LIBRARY_PATH', '')}"
            )

            command = [
                self.server_path,
                "--models-preset",
                self.models_preset,
                "--host",
                self.host,
                "--port",
                str(self.port),
            ]

            print("[ROUTER] Starting llama.cpp router")

            self.process = subprocess.Popen(
                command,
                env=env,
            )

            print(
                f"[ROUTER] PID: {self.process.pid}"
            )

            self._wait_for_port()

            print(
                f"[ROUTER] Ready on "
                f"http://{self.host}:{self.port}"
            )

    # ============================================================
    # Wait for router
    # ============================================================

    def _wait_for_port(self, timeout=180):

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:

            # Router process died while starting.
            if (
                self.process is not None
                and self.process.poll() is not None
            ):
                code = self.process.returncode

                self.process = None

                raise RuntimeError(
                    f"llama.cpp router exited during startup "
                    f"with code {code}"
                )

            try:

                with socket.create_connection(
                    (self.host, self.port),
                    timeout=1,
                ):
                    return

            except OSError:
                time.sleep(1)

        raise RuntimeError(
            f"llama.cpp router did not start within "
            f"{timeout} seconds"
        )

    # ============================================================
    # Graceful termination
    # ============================================================

    def terminate(self):

        with self.lock:

            if self.process is None:
                return

            if self.process.poll() is not None:
                self.process = None
                return

            print(
                f"[ROUTER] Sending SIGTERM "
                f"to PID {self.process.pid}"
            )

            self.process.terminate()

    # ============================================================
    # Force kill
    # ============================================================

    def kill(self):

        with self.lock:

            if self.process is None:
                return

            if self.process.poll() is None:

                print(
                    f"[ROUTER] Force killing "
                    f"PID {self.process.pid}"
                )

                self.process.kill()
                self.process.wait()

            self.process = None

    # ============================================================
    # Restart
    # ============================================================

    def restart(self):

        with self.lock:

            self.kill()
            self.start()