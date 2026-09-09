const container = document.getElementById("gc");
const newBtn = document.querySelector(".new-btn");
const chatList = document.querySelector(".sb-container");
const renameBtn = document.getElementById("rename-btn");
const pinBtn = document.getElementById("pin-btn");
const delBtn = document.getElementById("del-btn");
const menu = document.getElementById("chat-menu");
const outputDiv = document.querySelector(".output");
const header = document.getElementById("sum-header");
const ain = document.getElementById("ai-input");
const imgBtn = document.getElementById("img-btn");
const imgInput = document.getElementById("img-input");
const attachmentTray = document.getElementById("attachment-tray");
const toggleInput = document.querySelector(".switch input[type='checkbox']");
const controlBtn = document.getElementById("control-btn");
const controlImg = document.getElementById("control-img");

let currentAbortController = null;
let currentGenerationId = 0;
let isGenerating = false;
let isStopping = false;
let chatLoadGeneration = 0;
let navigationInFlight = null;
let assistantRenderFrame = null;
let pendingAssistantRender = null;
let isCollapsed = true;
let isHistory = false;
let currentChatId = null;
let saveInFlight = null;
let isSaving = false;
let isSearching = false;
let pendingImages = [];

const isEmpty = () => outputDiv.children.length === 0;
const API = "http://localhost:5000";
const isBusy = () => isSaving || isGenerating || isStopping;

function setBusy(busy) {
    isSaving = busy;
    newBtn.style.pointerEvents = busy ? "none" : "auto";
    chatList.style.pointerEvents = busy ? "none" : "auto";
    if (busy) {
        ain.disabled = imgBtn.disabled = toggleInput.disabled = true;
    } else if (!isGenerating && !isStopping) {
        ain.disabled = imgBtn.disabled = toggleInput.disabled = false;
    }
    document.body.style.cursor = busy ? "progress" : "default";
}

async function saveCurrentChat() {
    if (saveInFlight) return saveInFlight;
    if (isGenerating || isStopping) {
        console.warn("Save requested while generation is active.");
        return null;
    }

    const snapshotChatId = currentChatId;
    saveInFlight = (async () => {
        setBusy(true);
        try {
            const form = new FormData();
            if (snapshotChatId) form.append("chat_id", snapshotChatId);

            const saveRes = await fetch(`${API}/save`, { method: "POST", body: form });
            if (!saveRes.ok) throw new Error(`Save request failed: ${saveRes.status}`);

            const saveJson = await saveRes.json();
            if (saveJson.id) currentChatId = saveJson.id;

            const pruneRes = await fetch(`${API}/prune`, { method: "POST" });
            if (!pruneRes.ok) console.warn(`Prune request failed: ${pruneRes.status}`);

            await loadSidebarHistory();
            return saveJson;
        } catch (err) {
            console.error("Save failed:", err);
            throw err;
        } finally {
            setBusy(false);
            saveInFlight = null;
        }
    })();
    return saveInFlight;
}

function isPinned(value) {
    return value === true || value === 1 || value === "1" ||
        String(value).trim().toLowerCase() === "true";
}

async function loadSidebarHistory() {
    try {
        const response = await fetch(`${API}/history`);
        const payload = await response.json().catch(() => null);
        if (!response.ok) {
            throw new Error(payload?.message || `History request failed: ${response.status}`);
        }

        const history = Array.isArray(payload) ? payload :
            Array.isArray(payload?.history) ? payload.history : [];
        const fragment = document.createDocumentFragment();

        for (const item of history) {
            const id = String(item.id ?? "");
            const pinned = isPinned(item.pinned);
            const div = document.createElement("div");
            div.dataset.chatId = id;
            div.dataset.pinned = pinned ? "1" : "0";

            const title = document.createElement("p");
            if (pinned) {
                const pin = document.createElement("span");
                pin.className = "pin";
                pin.textContent = "📌";
                title.appendChild(pin);
            }
            title.appendChild(document.createTextNode(String(item.title || "Legacy Chat")));

            const menuButton = document.createElement("button");
            menuButton.className = "menu-btn";
            menuButton.dataset.id = id;
            menuButton.setAttribute("aria-label", "Chat menu");
            const icon = document.createElement("img");
            icon.src = "../../assets/ellipsis-vertical-solid-full.svg";
            icon.alt = "";
            menuButton.appendChild(icon);

            div.append(title, menuButton);
            div.addEventListener("click", async event => {
                if (event.target.closest(".menu-btn") || isBusy()) return;
                header.innerText = item.title || "Legacy Chat";
                header.style.setProperty(
                    "--snippet",
                    JSON.stringify(item.snippet || "No summary available")
                );
                await loadChat(id);
            });
            fragment.appendChild(div);
        }
        chatList.replaceChildren(fragment);
    } catch (error) {
        console.error("History load failed:", error);
    }
}
loadSidebarHistory();

chatList.addEventListener("click", e => {
    const menuButton = e.target.closest(".menu-btn");
    if (!menuButton) return;
    e.stopPropagation();
    const rect = menuButton.getBoundingClientRect();
    menu.style.left = rect.right + "px";
    menu.style.top = rect.top + "px";
    menu.style.display = "flex";
    menu.dataset.id = menuButton.dataset.id;
});

document.addEventListener("click", event => {
    if (!event.target.closest(".chat-menu") && !event.target.closest(".menu-btn")) {
        menu.style.display = "none";
        menu.dataset.id = "";
    }
});

function askRename() {
    return new Promise(resolve => {
        const overlay = document.createElement("div");
        Object.assign(overlay.style, {
            position: "fixed",
            inset: "0",
            background: "rgba(0,0,0,0.4)",
            backdropFilter: "blur(2px)",
            zIndex: "99998",
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
        });

        const modal = document.createElement("div");
        Object.assign(modal.style, {
            background: "#fff",
            padding: "24px 22px",
            borderRadius: "14px",
            boxShadow: "0 20px 60px rgba(0,0,0,0.25)",
            zIndex: "99999",
            minWidth: "320px",
            fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
        });

        modal.innerHTML = `
            <div style="font-weight:600;margin-bottom:12px;font-size:15px">Rename chat</div>
            <input id="rename-input" placeholder="New chat name" style="width:100%;padding:10px 12px;border:1px solid #e2e2e2;border-radius:8px;outline:none;font-size:14px;margin-bottom:14px;box-sizing:border-box">
            <div style="display:flex;gap:8px;justify-content:flex-end">
                <button id="rename-cancel" style="padding:8px 14px;border:1px solid #e2e2e2;background:#f7f7f7;border-radius:8px;cursor:pointer;font-size:13px">Cancel</button>
                <button id="rename-ok" style="padding:8px 16px;border:none;background:#111;color:#fff;border-radius:8px;cursor:pointer;font-size:13px">OK</button>
            </div>`;

        overlay.appendChild(modal);
        document.body.appendChild(overlay);
        const input = modal.querySelector("#rename-input");
        input.focus();
        input.select();

        const close = value => {
            overlay.remove();
            resolve(value);
        };
        modal.addEventListener("click", e => e.stopPropagation());
        overlay.addEventListener("click", () => close(null));

        const submit = () => {
            const value = input.value.trim();
            if (value) close(value);
        };
        modal.querySelector("#rename-ok").onclick = submit;
        modal.querySelector("#rename-cancel").onclick = () => close(null);
        input.addEventListener("keydown", e => {
            if (e.key === "Enter") submit();
            else if (e.key === "Escape") close(null);
        });
    });
}

renameBtn.addEventListener("click", () => chatMenu(1));
pinBtn.addEventListener("click", () => chatMenu(2));
delBtn.addEventListener("click", () => chatMenu(3));

async function chatMenu(type) {
    const id = menu.dataset.id;
    if (!id || isBusy()) return;

    try {
        menu.style.display = "none";

        if (type === 1) {
            const newName = await askRename();
            if (!newName) return;
            const form = new FormData();
            form.append("id", id);
            form.append("title", newName);
            const response = await fetch(`${API}/rename`, { method: "POST", body: form });
            if (!response.ok) throw new Error(`Rename request failed: ${response.status}`);
            await loadSidebarHistory();
        } else if (type === 2) {
            const form = new FormData();
            form.append("id", id);
            pinBtn.disabled = true;
            const response = await fetch(`${API}/pin`, { method: "POST", body: form });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(result.message || `Pin request failed: ${response.status}`);
            await loadSidebarHistory();
            pinBtn.disabled = false;
        } else if (type === 3) {
            const form = new FormData();
            form.append("id", id);
            const response = await fetch(`${API}/delete`, { method: "POST", body: form });
            if (!response.ok) throw new Error(`Delete request failed: ${response.status}`);
            await loadSidebarHistory();
            resetChat();
        } else {
            console.error("Unknown menu action");
        }
    } catch (error) {
        console.error(type === 1 ? "Rename failed:" : type === 2 ? "Pin toggle failed:" : "Delete failed:", error);
    } finally {
        pinBtn.disabled = false;
        menu.dataset.id = "";
    }
}

async function loadChat(chatId) {
    if (isBusy()) return;
    const requestedChatId = String(chatId);
    const loadId = ++chatLoadGeneration;

    navigationInFlight = (async () => {
        try {
            if (saveInFlight) await saveInFlight;
            if (isBusy()) return;

            if (!isHistory && !isEmpty()) {
                const saved = await saveCurrentChat();
                if (saved === null && isBusy()) return;
            }
            if (loadId !== chatLoadGeneration) return;

            const response = await fetch(`${API}/chat/${encodeURIComponent(requestedChatId)}`);
            if (!response.ok) throw new Error(`Chat request failed: ${response.status}`);
            const data = await response.json();
            if (!data || !Array.isArray(data.messages)) throw new Error("Invalid chat messages received");
            if (loadId !== chatLoadGeneration) return;

            currentChatId = requestedChatId;
            isHistory = true;
            outputDiv.replaceChildren();
            for (const message of data.messages) {
                if (!message || !["user", "assistant"].includes(message.role) ||
                    typeof message.content !== "string") continue;
                appendStoredMessage(outputDiv, message.role, message.content, false);
            }
            outputDiv.scrollTop = outputDiv.scrollHeight;
        } catch (err) {
            console.error("Chat load failed:", err);
        }
    })();

    try {
        await navigationInFlight;
    } finally {
        if (loadId === chatLoadGeneration) navigationInFlight = null;
    }
}

function resetChat() {
    ++chatLoadGeneration;
    outputDiv.replaceChildren();
    clearInputUI();
    header.innerText = "New Chat";
    header.style.setProperty("--snippet", '"Ask something to start a new chat"');
    isHistory = false;
    currentChatId = null;
}

newBtn.addEventListener("click", async () => {
    if (isBusy()) return;
    if (!isHistory && !isEmpty()) await saveCurrentChat();
    resetChat();
});

document.getElementById("btn").addEventListener("click", () => {
    isCollapsed = !isCollapsed;
    container.style.gridTemplateColumns = isCollapsed ? "70px 1fr" : "300px 1fr";
    newBtn.style.display = isCollapsed ? "none" : "flex";
    chatList.style.display = isCollapsed ? "none" : "block";
});

toggleInput.addEventListener("change", event => {
    isSearching = event.target.checked;
});

function resizeInput() {
    const maxHeight = 200;
    ain.style.height = "auto";
    ain.style.height = Math.min(ain.scrollHeight, maxHeight) + "px";
    ain.style.overflowY = ain.scrollHeight > maxHeight ? "auto" : "hidden";
}
ain.addEventListener("input", resizeInput);

function updateMessage(element, role, content) {
    if (role === "assistant") {
        try {
            element.innerHTML = window.electronAPI.parseMarkdown(content);
        } catch {
            element.textContent = String(content ?? "");
        }
    } else {
        element.textContent = String(content ?? "");
    }
}

function appendMessage(output, role, content = "", shouldScroll = true) {
    const message = document.createElement("div");
    message.className = role === "user" ? "msg user" : "msg assistant";

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = role === "user" ? "You" : "Assistant";

    const text = document.createElement("div");
    text.className = "text";
    message.append(meta, text);
    output.appendChild(message);
    updateMessage(text, role, content);

    if (shouldScroll) output.scrollTop = output.scrollHeight;
    return text;
}

function createMissingImageLabel(imageName) {
    const fallback = document.createElement("span");
    fallback.className = "image-missing";
    fallback.textContent = `[Image unavailable: ${imageName}]`;
    return fallback;
}

function appendStoredMessage(output, role, content = "", shouldScroll = true) {
    if (role !== "user") return appendMessage(output, role, content, shouldScroll);

    const imageNames = [];
    const remainingText = String(content).replace(
        /\[IMAGE:\s*([^\]\r\n]+)\]\s*/g,
        (_, imageName) => {
            imageNames.push(imageName.trim());
            return "";
        }
    );

    if (!imageNames.length) return appendMessage(output, role, content, shouldScroll);

    const message = document.createElement("div");
    message.className = "msg user";
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = "You";
    const text = document.createElement("div");
    text.className = "text";
    const imageList = document.createElement("div");
    imageList.className = "image-list";

    for (const imageName of imageNames) {
        const img = document.createElement("img");
        img.className = "message-image";
        img.alt = "Attached image";
        img.loading = "lazy";
        img.src = `${API}/image/${encodeURIComponent(imageName)}`;
        img.addEventListener("error", () => img.replaceWith(createMissingImageLabel(imageName)), { once: true });
        imageList.appendChild(img);
    }

    text.appendChild(imageList);
    const cleanText = remainingText.trim();
    if (cleanText) {
        const body = document.createElement("div");
        body.className = "stored-user-text";
        body.textContent = cleanText;
        text.appendChild(body);
    }
    message.append(meta, text);
    output.appendChild(message);
    if (shouldScroll) output.scrollTop = output.scrollHeight;
    return text;
}

function scheduleAssistantRender(element, content, generationId) {
    pendingAssistantRender = { element, content, generationId };
    if (assistantRenderFrame !== null) return;

    assistantRenderFrame = requestAnimationFrame(() => {
        assistantRenderFrame = null;
        const pending = pendingAssistantRender;
        pendingAssistantRender = null;
        if (!pending || pending.generationId !== currentGenerationId || !isGenerating) return;
        updateMessage(pending.element, "assistant", pending.content);
        outputDiv.scrollTop = outputDiv.scrollHeight;
    });
}

function flushAssistantRender() {
    if (assistantRenderFrame !== null) {
        cancelAnimationFrame(assistantRenderFrame);
        assistantRenderFrame = null;
    }
    const pending = pendingAssistantRender;
    pendingAssistantRender = null;
    if (!pending || pending.generationId !== currentGenerationId) return;
    updateMessage(pending.element, "assistant", pending.content);
    outputDiv.scrollTop = outputDiv.scrollHeight;
}

function revokeObjectURL(url) {
    if (!url) return;
    try { URL.revokeObjectURL(url); } catch {}
}

function clearInputUI() {
    ain.value = "";
    ain.style.height = "auto";
    ain.style.overflowY = "hidden";
    attachmentTray.querySelectorAll(".attachment-item[data-object-url]").forEach(item => {
        revokeObjectURL(item.dataset.objectUrl);
    });
    attachmentTray.replaceChildren();
    imgInput.value = "";
    pendingImages = [];
}

function setGenerationUI(active) {
    isGenerating = active;
    if (active) {
        controlImg.src = "../../assets/square-solid-full.svg";
        controlBtn.style.setProperty("--snippet", '"Stop answering"');
        controlBtn.disabled = isStopping;
        ain.disabled = false;
        imgBtn.disabled = true;
        toggleInput.disabled = true;
        controlBtn.onclick = stopAnswering;
    } else {
        controlImg.src = "../../assets/arrow-right-long-solid-full.svg";
        controlBtn.style.setProperty("--snippet", '"Send message"');
        controlBtn.disabled = false;
        ain.disabled = imgBtn.disabled = toggleInput.disabled = isSaving || isStopping;
        controlBtn.onclick = send;
    }
}

function finishGeneration(generationId) {
    if (generationId !== currentGenerationId) return false;
    flushAssistantRender();
    currentAbortController = null;
    setGenerationUI(false);
    return true;
}

async function sendMessage() {
    if (isBusy()) return;

    const query = ain.value.trim();
    if (!query && !pendingImages.length) return;

    const generationId = ++currentGenerationId;
    const controller = new AbortController();
    currentAbortController = controller;
    const generationFiles = [...pendingImages];
    setGenerationUI(true);

    if (query) appendMessage(outputDiv, "user", query);

    if (generationFiles.length) {
        const message = document.createElement("div");
        message.className = "msg user";
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = "Pics";
        const text = document.createElement("div");
        text.className = "text image-list";

        for (const file of generationFiles) {
            const url = URL.createObjectURL(file);
            const img = document.createElement("img");
            img.className = "message-image";
            img.alt = file.name || "Attached image";
            img.src = url;
            img.addEventListener("load", () => revokeObjectURL(url), { once: true });
            img.addEventListener("error", () => {
                revokeObjectURL(url);
                img.replaceWith(createMissingImageLabel(file.name || "image"));
            }, { once: true });
            text.appendChild(img);
        }
        message.append(meta, text);
        outputDiv.appendChild(message);
        outputDiv.scrollTop = outputDiv.scrollHeight;
    }

    clearInputUI();

    const loadingId = `loading-${Date.now()}-${generationId}`;
    const loading = document.createElement("div");
    loading.id = loadingId;
    loading.className = "msg assistant thinking";
    loading.textContent = "Assistant is thinking...";
    outputDiv.appendChild(loading);
    outputDiv.scrollTop = outputDiv.scrollHeight;
    isHistory = false;

    let assistantTextElement = null;

    try {
        const form = new FormData();
        form.append("query", query);
        form.append("websearch", isSearching);
        for (const image of generationFiles) form.append("image", image);
        if (currentChatId) form.append("chat_id", currentChatId);

        const response = await fetch(`${API}/ask`, {
            method: "POST",
            body: form,
            signal: controller.signal
        });
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        if (!response.body) throw new Error("Response body is not streamable");

        document.getElementById(loadingId)?.remove();
        assistantTextElement = appendMessage(outputDiv, "assistant", "");

        let fullResponse = "";
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let streamFinished = false;

        while (true) {
            if (generationId !== currentGenerationId) break;
            const { value, done } = await reader.read();
            if (done) {
                streamFinished = true;
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split("\n\n");
            buffer = events.pop() ?? "";

            for (const event of events) {
                const line = event.trim();
                if (!line.startsWith("data: ")) continue;
                const data = line.slice(6);

                if (data === "[DONE]") {
                    streamFinished = true;
                    buffer = "";
                    try { await reader.cancel(); } catch {}
                    break;
                }

                try {
                    const token = JSON.parse(data).token;
                    if (!token) continue;
                    fullResponse += token;
                    scheduleAssistantRender(assistantTextElement, fullResponse, generationId);
                } catch (error) {
                    console.error("Stream parsing error:", error, data);
                }
            }
            if (streamFinished) break;
        }

        buffer += decoder.decode();
        if (generationId === currentGenerationId && buffer.trim()) {
            const finalLine = buffer.trim();
            if (finalLine.startsWith("data: ")) {
                const data = finalLine.slice(6);
                if (data !== "[DONE]") {
                    try {
                        const token = JSON.parse(data).token;
                        if (token) fullResponse += token;
                    } catch {}
                }
            }
        }

        finishGeneration(generationId);
        if (controller.signal.aborted || generationId !== currentGenerationId || isStopping) return;

        try {
            const sumRes = await fetch(`${API}/sum`, { method: "POST" });
            if (!sumRes.ok) throw new Error(`Summary request failed: ${sumRes.status}`);
            const summaryBlock = await sumRes.json();
            header.innerText = summaryBlock.title || "New Chat";
            header.style.setProperty(
                "--snippet",
                JSON.stringify(summaryBlock.snippet || "No summary available.")
            );
        } catch (summaryError) {
            console.warn("Summary update failed:", summaryError);
        }
    } catch (err) {
        flushAssistantRender();
        if (err?.name === "AbortError" || controller.signal.aborted) {
            document.getElementById(loadingId)?.remove();
            console.log("Generation stopped by user.");
            if (generationId === currentGenerationId) {
                const stopped = appendMessage(outputDiv, "assistant", "");
                stopped.classList.add("generation-stopped");
                stopped.textContent = "Generation stopped by user";
            }
            return;
        }

        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.textContent = `Error: ${err?.message || "Request failed."}`;
        console.error(err);
    } finally {
        if (generationId === currentGenerationId) {
            currentAbortController = null;
            setGenerationUI(false);
        }
    }
}

function send() {
    if (isBusy()) return;
    void sendMessage();
}

async function stopAnswering() {
    if (!isGenerating || isStopping) return;

    const controller = currentAbortController;
    if (!controller) {
        isGenerating = false;
        isStopping = false;
        setGenerationUI(false);
        return;
    }

    isStopping = true;
    ++currentGenerationId;
    currentAbortController = null;
    isGenerating = false;
    controlBtn.disabled = true;
    controlImg.src = "../../assets/square-solid-full.svg";
    controlBtn.style.setProperty("--snippet", '"Stopping..."');

    void fetch(`${API}/stop`, { method: "POST", keepalive: true }).catch(error => {
        console.warn("Backend stop request failed:", error);
    });

    try { controller.abort(); } catch {}
    isStopping = false;
    setGenerationUI(false);
}

imgBtn.addEventListener("click", () => {
    if (isBusy()) return;
    imgInput.click();
});

function addToTray(files) {
    const picked = files ? Array.from(files) : Array.from(imgInput.files || []);
    if (!picked.length) return;

    for (const file of picked) {
        if (!file.type?.startsWith("image/")) continue;
        pendingImages.push(file);

        const url = URL.createObjectURL(file);
        const wrapper = document.createElement("div");
        wrapper.className = "attachment-item";
        wrapper.dataset.objectUrl = url;

        const img = document.createElement("img");
        img.src = url;
        img.alt = file.name || "Attached image";

        const remove = document.createElement("div");
        remove.className = "attachment-remove";
        remove.setAttribute("role", "button");
        remove.setAttribute("tabindex", "0");
        remove.setAttribute("aria-label", "Remove attachment");
        remove.textContent = "×";

        const removeFile = () => {
            const index = pendingImages.indexOf(file);
            if (index !== -1) pendingImages.splice(index, 1);
            revokeObjectURL(url);
            wrapper.remove();
        };

        remove.addEventListener("click", removeFile);
        remove.addEventListener("keydown", event => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                removeFile();
            }
        });

        wrapper.append(img, remove);
        attachmentTray.appendChild(wrapper);
    }
    imgInput.value = "";
}

imgInput.addEventListener("change", () => addToTray());

document.addEventListener("paste", event => {
    if (isBusy()) return;
    const clipboard = event.clipboardData;
    if (!clipboard) return;

    const blobs = [];
    for (const item of Array.from(clipboard.items)) {
        if (!item.type?.startsWith("image/")) continue;
        const blob = item.getAsFile();
        if (blob) blobs.push(blob);
    }
    if (blobs.length) addToTray(blobs);
});

ain.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        if (!isBusy()) send();
    }
});

controlBtn.onclick = send;