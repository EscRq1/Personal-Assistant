import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


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

        # PID of the process group created by start_new_session=True.
        # The router becomes the group leader, so:
        #
        #     pgid == router PID
        #
        self.process_group_id = None

        self.lock = threading.RLock()

    # ============================================================
    # Start
    # ============================================================

    def start(self):

        with self.lock:

            # ----------------------------------------------------
            # Already running.
            # ----------------------------------------------------

            if (
                self.process is not None
                and self.process.poll() is None
            ):
                return

            # ----------------------------------------------------
            # Clean up stale process object.
            # ----------------------------------------------------

            if self.process is not None:
                self.process = None
                self.process_group_id = None

            env = os.environ.copy()

            # ----------------------------------------------------
            # FastAPI may have CUDA_VISIBLE_DEVICES=1.
            #
            # Router must see all 3 GPUs.
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

            # ----------------------------------------------------
            # IMPORTANT:
            #
            # Create a completely separate process session/group.
            #
            # This allows us to terminate the router AND all
            # llama-server children it launches as one unit.
            # ----------------------------------------------------

            self.process = subprocess.Popen(
                command,
                env=env,
                start_new_session=True,
            )

            self.process_group_id = self.process.pid

            print(
                f"[ROUTER] PID: {self.process.pid}"
            )

            print(
                f"[ROUTER] Process group: "
                f"{self.process_group_id}"
            )

            try:

                # ------------------------------------------------
                # Wait until the router is genuinely ready.
                # ------------------------------------------------

                self._wait_for_ready()

            except Exception:

                # ------------------------------------------------
                # Startup failed.
                #
                # Do NOT leave the router/model child alive.
                # ------------------------------------------------

                print(
                    "[ROUTER] Startup failed; "
                    "terminating process group"
                )

                self._kill_process_group()

                raise

            print(
                f"[ROUTER] Ready on "
                f"http://{self.host}:{self.port}"
            )

    # ============================================================
    # Wait for router / server readiness
    # ============================================================

    def _wait_for_ready(self, timeout=180):

        deadline = time.monotonic() + timeout

        health_url = (
            f"http://{self.host}:{self.port}/health"
        )

        while time.monotonic() < deadline:

            # ----------------------------------------------------
            # Router process died during startup.
            # ----------------------------------------------------

            if (
                self.process is None
                or self.process.poll() is not None
            ):

                code = (
                    self.process.returncode
                    if self.process is not None
                    else "unknown"
                )

                self.process = None
                self.process_group_id = None

                raise RuntimeError(
                    "llama.cpp router exited during startup "
                    f"with code {code}"
                )

            # ----------------------------------------------------
            # First check that the TCP port exists.
            # ----------------------------------------------------

            try:

                with socket.create_connection(
                    (self.host, self.port),
                    timeout=1,
                ):
                    pass

            except OSError:

                time.sleep(1)
                continue

            # ----------------------------------------------------
            # Port is open.
            #
            # Now ask llama-server whether it is actually healthy.
            #
            # This is stronger than merely checking that something
            # is listening on port 8080.
            # ----------------------------------------------------

            try:

                request = urllib.request.Request(
                    health_url,
                    method="GET",
                )

                with urllib.request.urlopen(
                    request,
                    timeout=2,
                ) as response:

                    status = response.status

                    if status == 200:
                        return

            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                OSError,
            ):
                pass

            time.sleep(1)

        raise RuntimeError(
            "llama.cpp router did not become healthy "
            f"within {timeout} seconds"
        )

    # ============================================================
    # Graceful termination
    # ============================================================

    def terminate(self, timeout=10):

        with self.lock:

            if self.process is None:
                return

            pid = self.process.pid
            pgid = self.process_group_id

            # ----------------------------------------------------
            # If the parent is already dead, we still may have
            # child model servers alive in the process group.
            # Therefore we DO NOT return immediately.
            # ----------------------------------------------------

            if pgid is None:
                self.process = None
                return

            print(
                f"[ROUTER] Sending SIGTERM "
                f"to process group {pgid}"
            )

            try:
                os.killpg(
                    pgid,
                    signal.SIGTERM,
                )

            except ProcessLookupError:
                pass

            # ----------------------------------------------------
            # Wait for the router parent.
            #
            # The process group receives SIGTERM together, while
            # this wait confirms the router itself has exited.
            # ----------------------------------------------------

            deadline = time.monotonic() + timeout

            while time.monotonic() < deadline:

                if (
                    self.process is None
                    or self.process.poll() is not None
                ):
                    break

                time.sleep(0.1)

            # ----------------------------------------------------
            # If router did not shut down, escalate.
            # ----------------------------------------------------

            if (
                self.process is not None
                and self.process.poll() is None
            ):
                print(
                    "[ROUTER] Graceful shutdown timed out; "
                    "force killing process group"
                )

                self._kill_process_group()

            else:

                # ------------------------------------------------
                # Parent is gone.
                #
                # We still clear our state here. The process group
                # was already sent SIGTERM above.
                # ------------------------------------------------

                self.process = None
                self.process_group_id = None

    # ============================================================
    # Force kill
    # ============================================================

    def kill(self):

        with self.lock:

            if (
                self.process is None
                and self.process_group_id is None
            ):
                return

            print(
                "[ROUTER] Force killing router "
                "process group"
            )

            self._kill_process_group()

    # ============================================================
    # Internal process-group kill
    # ============================================================

    def _kill_process_group(self):

        pgid = self.process_group_id

        if pgid is None:
            self.process = None
            return

        # --------------------------------------------------------
        # Because start_new_session=True was used, this PGID
        # belongs exclusively to our router session.
        # --------------------------------------------------------

        try:

            os.killpg(
                pgid,
                signal.SIGKILL,
            )

            print(
                f"[ROUTER] Sent SIGKILL "
                f"to process group {pgid}"
            )

        except ProcessLookupError:

            # The process group is already gone.
            pass

        # --------------------------------------------------------
        # Wait for the router parent.
        #
        # Important: this is synchronization, not an arbitrary
        # sleep. We wait for the actual subprocess state change.
        # --------------------------------------------------------

        if self.process is not None:

            try:
                self.process.wait(timeout=10)

            except subprocess.TimeoutExpired:

                # ------------------------------------------------
                # Extremely defensive fallback.
                # ------------------------------------------------

                print(
                    "[ROUTER] Router did not exit after "
                    "SIGKILL within timeout"
                )

        self.process = None
        self.process_group_id = None

    # ============================================================
    # Restart
    # ============================================================

    def restart(self):

        with self.lock:

            print("[ROUTER] Restart requested")

            # ----------------------------------------------------
            # Kill the complete router/model process tree.
            # ----------------------------------------------------

            self.kill()

            # ----------------------------------------------------
            # Only after the old process group has been killed and
            # the router parent has been reaped do we start again.
            # ----------------------------------------------------

            self.start()