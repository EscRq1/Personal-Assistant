const container = document.getElementById("gc");
const newBtn = document.querySelector(".new-btn");
const saveBtn = document.getElementById("save-btn");
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
const headerMenuBtn = document.getElementById("header-menu-btn");
const headerMenu = document.getElementById("header-menu");
const selectContainers = document.querySelectorAll(".select-container");
const modeSelectorAssistant = document.getElementById("hm-assist-1");
const modeSelectorCoder = document.getElementById("hm-coder-1");
const budgetSelectorAssistant = document.getElementById("hm-assist-2");
const budgetSelectorCoder = document.getElementById("hm-coder-2");
const toolBudgetSelector = document.getElementById("hm-tools");

/* ------------------------------------------------------------------- state */

let currentAbortController = null;
let currentGenerationId = 0;
let stoppedGenerationId = 0;
let isGenerating = false;
let isStopping = false;
let chatLoadGeneration = 0;
let navigationInFlight = null;
let assistantRenderFrame = null;
let pendingAssistantRender = null;
let isCollapsed = true;
let isHistory = false;
let currentChatId = null;
let currentDraftId = crypto.randomUUID();
let saveInFlight = null;
let isSaving = false;
let isSearching = false;
let pendingAttachments = [];
let menuOpen = false;
let stickToBottom = true;

/* Turn mini-map state */
let turnMapEl = null;
let turnMapBarsEl = null;
let turnMapListEl = null;
let turnBars = [];
let turnObserver = null;
let visibleTurns = new Map();
let turnScrollRoot = document.querySelector(".content") || outputDiv;
let turnMapMO = null;
let turnMapBuildFrame = null;
let turnMapScrollFrame = null;
let activeTurnIndex = -1;
let turnIdCounter = 0;

const isEmpty = () => outputDiv.children.length === 0;

const API = "http://localhost:5000";

/* ignoreNavigation: for use INSIDE loadChat, where navigationInFlight is
   deliberately set by us and must not count as "busy". */
const isBusy = (ignoreNavigation = false) =>
    isSaving || isGenerating || isStopping ||
    (!ignoreNavigation && navigationInFlight !== null);

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

/* ------------------------------------------------------------------- saving */

async function saveCurrentChat(option) {
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

            let saveRes;
            if (option == 0) {
                saveRes = await fetch(`${API}/save`, { method: "POST", body: form });
                if (!saveRes.ok) throw new Error(`Save request failed: ${saveRes.status}`);
            } else {
                saveRes = await fetch(`${API}/finilize`, { method: "POST", body: form });
                if (!saveRes.ok) throw new Error(`Save request failed: ${saveRes.status}`);
            }

            const saveJson = await saveRes.json();
            if (saveJson.chat_id) currentChatId = saveJson.chat_id;

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

/* ----------------------------------------------------------------- sidebar */

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

/* ------------------------------------------------------------- chat actions */

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
            if (id === currentChatId) resetChat();
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

/* -------------------------------------------------------------- navigation */

async function loadChat(chatId) {
    if (isBusy()) return;

    const requestedChatId = String(chatId);
    const loadId = ++chatLoadGeneration;

    // Hold our own promise so the finally can verify ownership before clearing.
    const navTask = (async () => {
        try {
            if (saveInFlight) await saveInFlight;
            if (isBusy(true)) return;              // was isBusy() — saw our own flag

            if (!isHistory && !isEmpty()) {
                try {
                    const saved = await saveCurrentChat(1);
                    if (saved === null && isBusy(true)) return;
                } catch (err) {
                    console.error("Auto-save before navigation failed:", err);
                    // proceed with navigation anyway
                }
            }

            if (loadId !== chatLoadGeneration) return;

            const response = await fetch(`${API}/chat/${encodeURIComponent(requestedChatId)}`);
            if (!response.ok) throw new Error(`Chat request failed: ${response.status}`);

            const data = await response.json();
            if (!data || !Array.isArray(data.messages)) throw new Error("Invalid chat messages received");

            if (loadId !== chatLoadGeneration) return;

            currentChatId = requestedChatId;
            isHistory = true;
            stickToBottom = true;
            outputDiv.replaceChildren();

            for (const message of data.messages) {
                if (!message || !["user", "assistant"].includes(message.role) ||
                    typeof message.content !== "string") continue;

                appendStoredMessage(outputDiv, message.role, message.content, false);
            }

            if (stickToBottom) outputDiv.scrollTop = outputDiv.scrollHeight;

            // The stored chat has now been fully materialized. Build the turn
            // map immediately instead of relying only on MutationObserver/rAF.
            refreshTurnMapNow();

        } catch (err) {
            console.error("Chat load failed:", err);
        }
    })();

    navigationInFlight = navTask;

    try {
        await navTask;
        refreshTurnMap();
    } finally {
        // Identity check, not generation check: resetChat()/a newer load may have
        // bumped chatLoadGeneration while we awaited. Clearing on generation left
        // the flag permanently set after delete/new-chat -> frozen sidebar.
        if (navigationInFlight === navTask) navigationInFlight = null;
    }
}

function resetChat() {
    ++chatLoadGeneration;
    navigationInFlight = null;          // explicit release — prevents permanent freeze
    outputDiv.replaceChildren();
    stickToBottom = true;
    clearInputUI();

    header.innerText = "New Chat";
    header.style.setProperty("--snippet", '"Ask something to start a new chat"');

    isHistory = false;
    currentChatId = null;
    currentDraftId = crypto.randomUUID();
    refreshTurnMapNow();
}

newBtn.addEventListener("click", async () => {
    if (isBusy()) return;

    if (!isHistory && !isEmpty()) {
        try { await saveCurrentChat(1); }
        catch (err) { console.error("Auto-save before new chat failed:", err); }
    }

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

/* --------------------------------------------------------------- rendering */

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

    if (shouldScroll && stickToBottom) output.scrollTop = output.scrollHeight;
    return text;
}

function createMissingImageLabel(imageName) {
    const fallback = document.createElement("span");
    fallback.className = "image-missing";
    fallback.textContent = `[Image unavailable: ${imageName}]`;
    return fallback;
}

/* Stored user messages embed image references inline. Accepted shapes
   (case-insensitive): "[image: name]", "[image=name]", "[img: name]",
   "[pic name]", or a bare bracketed filename "[shot.png]".

   NOTE: confirm against whatever main.py actually writes into img_tags —
   the previous pattern (/\\\r\n]+)\\]\sg) had an unmatched ")" and made the
   whole script fail to parse. */

const STORED_IMAGE_TAG_RE =
    /!?\[(?:(?:image|img|pic|photo)\s*[:=]?\s*([^\]\r\n]+?)|([^\]\r\n]+\.(?:png|jpe?g|webp|gif|bmp)))\s*\]/gi;

function appendStoredMessage(output, role, content = "", shouldScroll = true) {
    if (role !== "user") return appendMessage(output, role, content, shouldScroll);

    const imageNames = [];
    const attachedFiles = [];
    const withoutFiles = String(content).replace(/\[SANDBOX_FILES: ([^\r\n]+)\]/g, (tag, data) => {
        try {
            const files = JSON.parse(data);
            if (!Array.isArray(files)) return tag;
            attachedFiles.push(...files.filter(file => file && typeof file.name === "string"));
            return "";
        } catch { return tag; }
    });

    const remainingText = withoutFiles.replace(
        STORED_IMAGE_TAG_RE,
        (match, namedName, fileOnlyName) => {
            const imageName = (namedName || fileOnlyName || "").trim();
            if (!imageName) return match;
            imageNames.push(imageName);
            return "";
        }
    );

    if (!imageNames.length && !attachedFiles.length) return appendMessage(output, role, content, shouldScroll);

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
        img.style.maxWidth = "400px";
        img.style.maxHeight = "400px";
        img.src = `${API}/image/${encodeURIComponent(imageName)}`;
        img.addEventListener("error", () => img.replaceWith(createMissingImageLabel(imageName)), { once: true });
        imageList.appendChild(img);
    }

    text.appendChild(imageList);
    for (const file of attachedFiles) text.appendChild(createFileCard(file));

    const cleanText = remainingText.trim();

    if (cleanText) {
        const body = document.createElement("div");
        body.className = "stored-user-text";
        body.textContent = cleanText;
        text.appendChild(body);
    } else {
        // Image-only stored turn: mark it so previews can label it.
        message.classList.add("pics-only");
    }

    message.append(meta, text);
    output.appendChild(message);

    if (shouldScroll && stickToBottom) output.scrollTop = output.scrollHeight;
    return text;
}

/* Auto-scroll is gated on stickToBottom so scrolling up mid-stream (or while
   using the turn map) is not yanked back to the bottom. */

function scheduleAssistantRender(element, content, generationId) {
    pendingAssistantRender = { element, content, generationId };
    if (assistantRenderFrame !== null) return;

    assistantRenderFrame = requestAnimationFrame(() => {
        assistantRenderFrame = null;
        const pending = pendingAssistantRender;
        pendingAssistantRender = null;

        if (!pending || pending.generationId !== currentGenerationId || !isGenerating) return;

        updateMessage(pending.element, "assistant", pending.content);
        if (stickToBottom) outputDiv.scrollTop = outputDiv.scrollHeight;
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
    if (stickToBottom) outputDiv.scrollTop = outputDiv.scrollHeight;
}

function revokeObjectURL(url) {
    if (!url) return;
    try { URL.revokeObjectURL(url); } catch { }
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
    pendingAttachments = [];
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

/* ------------------------------------------------------------- generation */

function appendLiveUserTurn(output, query, files, shouldScroll = true) {
    const message = document.createElement("div");
    message.className = "msg user";

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = "You";

    const text = document.createElement("div");
    text.className = "text";

    const cleanQuery = String(query ?? "").trim();

    if (files.length) {
        const imageList = document.createElement("div");
        imageList.className = "image-list";

        for (const file of files) {
            if (!isPreviewImage(file)) {
                imageList.appendChild(createFileCard(file));
                continue;
            }
            const url = URL.createObjectURL(file);
            const img = document.createElement("img");
            img.className = "message-image";
            img.alt = file.name || "Attached image";
            img.src = url;
            img.style.maxWidth = "400px";
            img.style.maxHeight = "400px";
            img.addEventListener("load", () => revokeObjectURL(url), { once: true });
            img.addEventListener("error", () => {
                revokeObjectURL(url);
                img.replaceWith(createMissingImageLabel(file.name || "image"));
            }, { once: true });

            imageList.appendChild(img);
        }

        text.appendChild(imageList);
    }

    if (cleanQuery) {
        const body = document.createElement("div");
        body.className = "stored-user-text";
        body.textContent = cleanQuery;
        text.appendChild(body);
    } else if (files.length) {
        message.classList.add("pics-only");
    }

    message.append(meta, text);
    output.appendChild(message);

    if (shouldScroll && stickToBottom) output.scrollTop = output.scrollHeight;
    return text;
}

const INLINE_INPUT_LINE_CAP = 500;

function splitOversizedInput(text) {
    const lines = String(text).split("\n");
    if (lines.length <= INLINE_INPUT_LINE_CAP) return null;
    const inline = lines.slice(0, INLINE_INPUT_LINE_CAP).join("\n");
    const overflow = lines.slice(INLINE_INPUT_LINE_CAP).join("\n");
    if (!overflow) return null;
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const file = new File([overflow], `pasted-snippet-${stamp}.txt`, {
        type: "text/plain",
    });
    return { inline, file };
}

async function sendMessage() {
    if (isBusy()) return;

    const query = ain.value.trim();
    if (!query && !pendingAttachments.length) return;

    const generationId = ++currentGenerationId;
    const controller = new AbortController();
    currentAbortController = controller;

    const generationFiles = [...pendingAttachments];

    const split = splitOversizedInput(query);
    if (split) {
        query = split.inline;
        generationFiles.push(split.file);
    }

    const requestChatId = currentChatId;
    const requestDraftId = currentDraftId;
    let requestAccepted = false;

    setGenerationUI(true);
    const userText = appendLiveUserTurn(outputDiv, query, generationFiles);
    clearInputUI();

    const loadingId = `loading-${Date.now()}-${generationId}`;
    const loading = document.createElement("div");
    loading.id = loadingId;
    loading.className = "msg assistant thinking";
    loading.textContent = generationFiles.length
        ? "Uploading attachments..."
        : "Assistant is thinking...";

    outputDiv.appendChild(loading);
    if (stickToBottom) outputDiv.scrollTop = outputDiv.scrollHeight;

    isHistory = false;
    let assistantTextElement = null;
    let thinkingBubble = null;
    let workingBubble = null;

    function ensureWorkingBubble() {
        if (workingBubble) return workingBubble;
        if (loading.isConnected) loading.remove();
        const wrapper = document.createElement("div");
        wrapper.className = "msg assistant thinking-wrap";
        outputDiv.appendChild(wrapper);
        workingBubble = createWorkingBubble(wrapper);
        if (stickToBottom) outputDiv.scrollTop = outputDiv.scrollHeight;
        return workingBubble;
    }

    // The collapsible reasoning bubble is appended right below the
    // working bubble as soon as the first reasoning chunk arrives.
    function ensureThinkingBubble() {
        if (thinkingBubble) return thinkingBubble;
        const wrapper = ensureWorkingBubble().bubble.parentElement;
        thinkingBubble = createThinkingBubble(wrapper);
        return thinkingBubble;
    }

    function firstRealToken() {
        if (thinkingBubble) thinkingBubble.finish();
    }


    try {
        const form = new FormData();
        form.append("query", query);
        form.append("websearch", isSearching);
        form.append("m-assist", modeSelectorAssistant.value);
        form.append("b-assist", budgetSelectorAssistant.value);
        form.append("m-coder", modeSelectorCoder.value);
        form.append("b-coder", budgetSelectorCoder.value);
        form.append("b-tools", toolBudgetSelector.value);

        for (const file of generationFiles) {
            form.append("file", file, file.name);
        }

        if (requestChatId) form.append("chat_id", requestChatId);
        else form.append("draft_id", requestDraftId);

        ensureWorkingBubble();

        const response = await fetch(`${API}/ask`, {
            method: "POST",
            body: form,
            signal: controller.signal
        });

        if (!response.ok) {
            const problem = await response.json().catch(() => ({}));
            throw new Error(
                problem.error ||
                problem.detail ||
                `Server returned ${response.status}`
            );
        }

        requestAccepted = true;
        if (!response.body) throw new Error("Response body is not streamable");

        let fullResponse = "";
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let streamFinished = false;

        function handleEvent(event) {
            const line = event.trim();
            if (!line.startsWith("data: ")) return false;

            const data = line.slice(6);
            if (data === "[DONE]") return true;

            try {
                const packet = JSON.parse(data);

                if (Array.isArray(packet.sandbox_files)) {
                    showUploadedFiles(userText, packet.sandbox_files);

                }
                const reasoning = packet.reasoning;
                if (reasoning) {
                    if (!assistantTextElement) {
                        ensureThinkingBubble().append(reasoning);
                    }
                    return false;
                }

                const token = packet.token;
                if (token) {
                    if (!assistantTextElement) {
                        firstRealToken(); 
                        assistantTextElement = appendMessage(
                            outputDiv, 
                            "assistant", 
                            ""
                        );
                    }

                    fullResponse += token;
                    scheduleAssistantRender(
                        assistantTextElement,
                        fullResponse,
                        generationId
                    );
                }
            } catch (error) {
                console.error("Stream parsing error:", error, data);
            }

            return false;
        }

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
                if (handleEvent(event)) {
                    streamFinished = true;
                    buffer = "";
                    try { await reader.cancel(); } catch { }
                    break;
                }
            }

            if (streamFinished) break;
        }

        // Process an event that arrived without a final "\n\n".
        buffer += decoder.decode();

        if (generationId === currentGenerationId && buffer.trim()) {
            for (const event of buffer.split("\n\n")) {
                if (handleEvent(event)) break;
            }
        }

        // Also removes it if the stream ended without producing a token.
        if (thinkingBubble) thinkingBubble.finish();
        if (workingBubble) workingBubble.stop();
        loading.remove();
        finishGeneration(generationId);

        if (
            controller.signal.aborted ||
            generationId !== currentGenerationId ||
            isStopping
        ) {
            return;
        }

        try {
            const sumRes = await fetch(`${API}/sum`, { method: "POST" });
            if (!sumRes.ok) {
                throw new Error(`Summary request failed: ${sumRes.status}`);
            }

            const summaryBlock = await sumRes.json();
            header.innerText = summaryBlock.title || "New Chat";
            header.style.setProperty(
                "--snippet",
                JSON.stringify(
                    summaryBlock.snippet || "No summary available."
                )
            );
        } catch (summaryError) {
            console.warn("Summary update failed:", summaryError);
        }
    } catch (err) {
        flushAssistantRender();
        if (workingBubble) workingBubble.stop();
        if (thinkingBubble) thinkingBubble.finish();

        if (
            !requestAccepted &&
            currentChatId === requestChatId &&
            currentDraftId === requestDraftId
        ) {
            // Restore the input if the request was rejected or disconnected
            // before the server accepted it.
            if (!ain.value) ain.value = query;
            addToTray(generationFiles, true);
            userText.closest(".msg")?.remove();
        }

        if (err?.name === "AbortError" || controller.signal.aborted) {
            if (thinkingBubble) thinkingBubble.finish();
            loading.remove();
            console.log("Generation stopped by user.");

            if (
                generationId === currentGenerationId ||
                generationId === stoppedGenerationId
            ) {
                const stopped = appendMessage(outputDiv, "assistant", "");
                stopped.classList.add("generation-stopped");
                stopped.textContent = "Generation stopped by user";
            }

            return;
        }

        if (loading.isConnected) {
            loading.textContent =
                `Error: ${err?.message || "Request failed."}`;
        } else {
            const errorText = appendMessage(outputDiv, "assistant", "");
            errorText.textContent =
                `Error: ${err?.message || "Request failed."}`;
        }

        console.error(err);
    } finally {
        if (generationId === currentGenerationId) {
            refreshTurnMap();
            currentAbortController = null;
            setGenerationUI(false);
        }
    }
}

function formatWorkTime(ms) {
    const totalSec = Math.max(0, Math.floor(ms / 1000));
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = totalSec % 60;
    const pad = (n) => String(n).padStart(2, "0");
    if (h > 0) return `${h}h ${pad(m)}m ${pad(s)}s`;
    if (m > 0) return `${m}m ${pad(s)}s`;
    return `${s}s`;
}

function createThinkingBubble(parent, label = "Assistant is thinking...") {
    const bubble = document.createElement("div");
    bubble.className = "thinking-bubble";
    bubble.classList.add("active");   // BUG FIX: spinner is CSS-hidden unless .active

    const header = document.createElement("div");
    header.className = "thinking-header";
    header.setAttribute("role", "button");
    header.setAttribute("tabindex", "0");

    const spinner = document.createElement("span");
    spinner.className = "thinking-spinner";

    const labelEl = document.createElement("span");
    labelEl.className = "thinking-label";
    labelEl.textContent = label;

    const chevron = document.createElement("span");
    chevron.className = "thinking-chevron";

    header.append(spinner, labelEl, chevron);

    const body = document.createElement("div");
    body.className = "thinking-body";

    const text = document.createElement("div");
    text.className = "thinking-text";

    body.appendChild(text);
    bubble.append(header, body);
    parent.appendChild(bubble);

    const toggle = () => bubble.classList.toggle("open");
    header.addEventListener("click", toggle);
    header.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            toggle();
        }
    });

    return {
        bubble,
        header,
        labelEl,
        text,
        append(chunk) {
            text.textContent += chunk;
            // BUG FIX: scroll the real scroll container, not the wrapper
            if (stickToBottom) outputDiv.scrollTop = outputDiv.scrollHeight;
        },
        finish() {
            bubble.classList.remove("active");
            this.labelEl.textContent = "Chain-Of-Thoughts";
            if (this.text.textContent.trim()) {
                this.bubble.classList.add("has-content");
            }
        }
    };
}


function createWorkingBubble(parent) {
    const bubble = document.createElement("div");
    bubble.className = "working-note";

    const labelEl = document.createElement("span");
    labelEl.className = "working-label";
    labelEl.textContent = "Working for 0s";

    bubble.appendChild(labelEl);
    parent.appendChild(bubble);

    const startedAt = performance.now();
    let stopped = false;
    const timer = setInterval(() => {
        if (!stopped) {
            labelEl.textContent =
                "Working for " + formatWorkTime(performance.now() - startedAt);
        }
    }, 250);

    return {
        bubble,
        labelEl,
        stop() {
            if (stopped) return;
            stopped = true;
            clearInterval(timer);
            bubble.classList.add("done");
            this.labelEl.textContent =
                "Worked for " + formatWorkTime(performance.now() - startedAt);
        }
    };

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
    stoppedGenerationId = currentGenerationId;
    ++currentGenerationId;

    currentAbortController = null;
    isGenerating = false;
    controlBtn.disabled = true;
    controlImg.src = "../../assets/square-solid-full.svg";
    controlBtn.style.setProperty("--snippet", '"Stopping..."');

    try { controller.abort(); } catch { }

    const startedAt = Date.now();
    await fetch(`${API}/stop`, { method: "POST", keepalive: true })
        .catch(error => console.warn("Backend stop request failed:", error));

    // Small floor so the "Stopping..." state is visible and the 6-stage
    // backend watchdog has a head start before new input is re-enabled.
    const elapsed = Date.now() - startedAt;
    if (elapsed < 300) await new Promise(r => setTimeout(r, 300 - elapsed));

    isStopping = false;
    setGenerationUI(false);
}

/* ------------------------------------------------------------ attachments */

imgBtn.addEventListener("click", () => {
    if (isBusy()) return;
    imgInput.click();
});

// Reuse the existing picker/tray; no HTML or stylesheet replacement required.
imgInput.removeAttribute("accept");
imgInput.multiple = true;
imgBtn.title = "Attach files";
imgBtn.setAttribute("aria-label", "Attach files");

const MAX_ATTACHMENT_FILES = 20;
const MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024;
const MAX_ATTACHMENTS_BYTES = 128 * 1024 * 1024;
const MAX_IMAGE_BYTES = 16 * 1024 * 1024;
const MAX_IMAGES_BYTES = 32 * 1024 * 1024;
const PREVIEW_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"]);
const isPreviewImage = file => PREVIEW_IMAGE_TYPES.has(file.type);

function formatFileSize(size) {
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KiB`;
    return `${(size / (1024 * 1024)).toFixed(1)} MiB`;
}

function createFileCard(file) {
    const card = document.createElement("div");
    card.className = "sandbox-file-card";
    const name = document.createElement("span");
    name.className = "sandbox-file-name";
    name.textContent = file.name || "Attached file";
    const size = document.createElement("small");
    size.textContent = formatFileSize(file.size || 0);
    card.append(name, size);
    card.title = file.path || file.name || "Attached file";
    return card;
}

function showUploadedFiles(userText, files) {
    let status = userText.querySelector(".sandbox-upload-status");
    if (!status) {
        status = document.createElement("div");
        status.className = "sandbox-upload-status";
        userText.appendChild(status);
    }
    status.textContent = `${files.length} file${files.length === 1 ? "" : "s"} added to this chat`;
    status.title = files.map(file => file.path).join("\n");
}

function addToTray(files, restoring = false) {
    if (!restoring && isBusy()) return;
    const picked = files ? Array.from(files) : Array.from(imgInput.files || []);
    if (!picked.length) return;
    const errors = [];
    let total = pendingAttachments.reduce((sum, file) => sum + file.size, 0);
    let imageTotal = pendingAttachments.filter(isPreviewImage).reduce((sum, file) => sum + file.size, 0);
    for (const file of picked) {
        if (pendingAttachments.includes(file)) continue;
        if (pendingAttachments.length >= MAX_ATTACHMENT_FILES || file.size > MAX_ATTACHMENT_BYTES ||
            total + file.size > MAX_ATTACHMENTS_BYTES ||
            (isPreviewImage(file) && (file.size > MAX_IMAGE_BYTES || imageTotal + file.size > MAX_IMAGES_BYTES))) {
            errors.push(file.name);
            continue;
        }
        pendingAttachments.push(file);
        total += file.size;
        if (isPreviewImage(file)) imageTotal += file.size;
        const wrapper = document.createElement("div");
        wrapper.className = "attachment-item";
        wrapper.title = file.name;
        let url = null;
        let preview;
        if (isPreviewImage(file)) {
            url = URL.createObjectURL(file);
            wrapper.dataset.objectUrl = url;
            preview = document.createElement("img");
            preview.src = url;
            preview.alt = file.name || "Attached image";
        } else {
            wrapper.classList.add("sandbox-file-attachment");
            preview = createFileCard(file);
        }
        const remove = document.createElement("div");
        remove.className = "attachment-remove";
        remove.setAttribute("role", "button");
        remove.setAttribute("tabindex", "0");
        remove.setAttribute("aria-label", "Remove attachment");
        remove.textContent = "×";
        const removeFile = () => {
            const index = pendingAttachments.indexOf(file);
            if (index !== -1) pendingAttachments.splice(index, 1);
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
        wrapper.append(preview, remove);
        attachmentTray.appendChild(wrapper);
    }
    imgInput.value = "";
    if (errors.length) window.alert(`Could not attach: ${errors.join(", ")}. Limits: 20 files, 32 MiB per file, 128 MiB total; images 16 MiB each / 32 MiB total.`);
}

imgInput.addEventListener("change", () => addToTray());
document.addEventListener("paste", event => {
    if (isBusy()) return;
    const files = Array.from(event.clipboardData?.items || [])
        .filter(item => item.kind === "file").map(item => item.getAsFile()).filter(Boolean);
    if (files.length) {
        event.preventDefault();
        addToTray(files);
    }
    
    const target = event.target;
    const pasted = event.clipboardData?.getData("text/plain");

    if (target === ain && typeof pasted === "string") {
        const pastedLines = pasted.split("\n");

        if (pastedLines.length > INLINE_INPUT_LINE_CAP) {
            // The paste itself is too large to prefill: the whole paste
            // becomes an attached text file. The input value is untouched, so
            // anything already typed stays exactly as it was.
            event.preventDefault();
            const stamp = new Date().toISOString().replace(/[:.]/g, "-");
            const file = new File([pasted], `pasted-snippet-${stamp}.txt`, {
                type: "text/plain",
            });
            addToTray([file]);
        }
    }
});
document.addEventListener("dragover", event => {
    if (Array.from(event.dataTransfer?.types || []).includes("Files")) event.preventDefault();
});
document.addEventListener("drop", event => {
    if (!event.dataTransfer?.files?.length) return;
    event.preventDefault();
    if (!isBusy()) addToTray(event.dataTransfer.files);
});

const attachmentStyle = document.createElement("style");
attachmentStyle.textContent = `
.sandbox-file-card { display: inline-flex; flex-direction: column; gap: 4px; padding: 10px 14px; border: 1px solid currentColor; border-radius: 10px; max-width: 240px; margin: 4px; }
.sandbox-file-name { overflow-wrap: anywhere; font-weight: 600; }
.sandbox-file-card small, .sandbox-upload-status { opacity: .75; font-size: .8rem; }
.sandbox-file-attachment { width: auto !important; height: auto !important; max-width: 260px; }
.sandbox-upload-status { margin-top: 8px; }
`;
document.head.appendChild(attachmentStyle);

ain.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        if (!isBusy()) send();
    }
});

controlBtn.onclick = send;

/* ----------------------------------------------------------- header menu */

selectContainers.forEach(container => {
    const select = container.querySelector("select");
    if (!select) return;

    select.addEventListener("focus", () => container.classList.add("no-after"));
    select.addEventListener("blur", () => container.classList.remove("no-after"));
});

function setMenu(open) {
    if (menuOpen === open) return;
    menuOpen = open;

    headerMenu.classList.toggle("open", open);
    headerMenuBtn.setAttribute("aria-expanded", String(open));
}

headerMenuBtn.addEventListener("click", () => setMenu(!menuOpen));

/* Outside click: capture phase so it runs before chatList's stopPropagation.
   closest("#header-menu") keeps every click INSIDE the menu (including on
   selects) from closing it. */
document.addEventListener("click", event => {
    if (!menuOpen) return;
    if (event.target.closest("#header-menu-btn") || event.target.closest("#header-menu")) return;
    setMenu(false);
}, true);

document.addEventListener("keydown", event => {
    if (event.key !== "Escape" || !menuOpen) return;
    // Esc pressed to dismiss a native <select> popup shouldn't close the whole menu.
    if (document.activeElement && document.activeElement.tagName === "SELECT") return;
    setMenu(false);
});

/* ============================================================================
   TURN MINI-MAP

   One logical bar per user turn. A turn is represented by exactly one
   .msg.user bubble, including text + attachments. The active bar is derived
   from the user's scroll position rather than IntersectionObserver ratios.
   ========================================================================== */

/* Injected so the feature works even if style.css / index.html were not
   updated. Move to style.css and delete this block if you prefer. */

const TURN_MAP_CSS = `
.turns-col{
    grid-area: list;
    position: relative;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding: 0 10px;
    overflow: visible;
}

.turn-map{
    position: relative;
    align-self: center;
    display: flex;
    align-items: center;
    padding: 8px 4px;
    z-index: 60;
}

/* Invisible bridge keeps the parent hovered while the pointer moves from
   the fixed bars into the absolutely-positioned list. */
.turn-map::before{
    content: "";
    position: absolute;
    top: -12px;
    left: -300px;
    width: 300px;
    height: calc(100% + 24px);
    z-index: 1;
}

.turn-map-bars{
    position: relative;
    z-index: 2;
    display: flex;
    flex-direction: column;
    gap: 5px;
    align-items: flex-end;
}

.turn-bar{
    width: 22px;
    height: 3px;
    border-radius: 2px;
    background: rgba(128,128,128,.28);
    cursor: pointer;
    transition: width .15s ease, background .15s ease, height .15s ease;
}

.turn-bar:hover{
    width: 28px;
    background: rgba(60,60,60,.55);
}

.turn-bar.active{
    width: 28px;
    height: 5px;
    background: rgba(0,0,0,.82);
}

.turn-map-list{
    position: absolute;
    z-index: 3;
    right: calc(100% + 8px);
    top: 50%;
    transform: translateY(-50%);
    width: 280px;
    max-height: min(360px, 50vh);
    overflow-y: auto;
    overscroll-behavior: contain;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,.25);
    background: #ffffff;
    box-shadow: 0 10px 28px rgba(0,0,0,.18);
    opacity: 0;
    visibility: hidden;
    transition: opacity .18s ease, visibility .18s ease;
    pointer-events: none;
}

.turn-map:hover .turn-map-list,
.turn-map-list:hover{
    opacity: 1;
    visibility: visible;
    pointer-events: auto;
}

.turn-map-empty{
    padding: 8px 10px;
    font-size: 12px;
    opacity: .5;
}

.turn-map-item{
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 7px 9px;
    border: 0;
    border-radius: 7px;
    background: transparent;
    color: inherit;
    cursor: pointer;
    font: inherit;
    font-size: 12.5px;
    text-align: left;
}

.turn-map-item:hover,
.turn-map-item:focus-visible{
    background: rgba(128,128,128,.14);
    outline: none;
}

.turn-map-item.active{
    background: rgba(33,150,243,.16);
}

.turn-map-item .tm-idx{
    flex: 0 0 auto;
    min-width: 18px;
    font-size: 10.5px;
    opacity: .5;
    font-variant-numeric: tabular-nums;
}

.turn-map-item .tm-preview{
    flex: 1 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

@media (max-width: 640px){
    .turn-bar{ width: 16px; }
    .turn-bar:hover,
    .turn-bar.active{ width: 20px; }
    .turn-map-list{ width: 220px; }
}
`;

function injectTurnMapStyles() {
    if (document.getElementById("turn-map-styles")) return;

    const style = document.createElement("style");
    style.id = "turn-map-styles";
    style.textContent = TURN_MAP_CSS;
    document.head.appendChild(style);
}

function ensureTurnMapMarkup() {
    let col = document.querySelector(".turns-col") || document.getElementById("turns-col");

    if (!col) {
        col = document.createElement("div");
        col.className = "turns-col";
        col.id = "turns-col";

        const content = document.querySelector(".content");
        if (!content) return null;
        content.appendChild(col);
    }

    const existing = col.querySelector(".turn-map");
    if (existing) {
        turnMapEl = existing;
        turnMapBarsEl = existing.querySelector(".turn-map-bars");
        turnMapListEl = existing.querySelector(".turn-map-list");

        if (turnMapBarsEl && turnMapListEl) return true;
    }

    col.querySelectorAll(".turns-anchor, .turns-toggle, .turns-popover").forEach(el => el.remove());

    const map = document.createElement("div");
    map.className = "turn-map";
    map.id = "turnMap";
    map.setAttribute("aria-label", "Turn mini-map");

    const bars = document.createElement("div");
    bars.className = "turn-map-bars";
    bars.id = "turnMapBars";

    const list = document.createElement("div");
    list.className = "turn-map-list";
    list.id = "turnMapList";
    list.setAttribute("role", "listbox");
    list.setAttribute("aria-label", "Turns in this conversation");

    map.append(bars, list);
    col.appendChild(map);

    turnMapEl = map;
    turnMapBarsEl = bars;
    turnMapListEl = list;

    return true;
}

function previewFromBubble(bubble) {
    const clone = bubble.cloneNode(true);
    clone.querySelectorAll(
        ".meta, .image-list, .msg-meta, .msg-actions, .msg-images, .msg-loading"
    ).forEach(el => el.remove());

    const text = (clone.textContent || "").replace(/\s+/g, " ").trim();
    if (!text) return bubble.classList.contains("pics-only") ? "📎 Image" : "";

    return text.length > 80 ? text.slice(0, 80) + "\u2026" : text;
}

function getTurnPreview(bubble) {
    const preview = previewFromBubble(bubble);
    return preview || "Turn";
}

function setActiveTurn(index) {
    if (index === activeTurnIndex) return;

    activeTurnIndex = index;

    turnBars.forEach((tb, i) => {
        const active = i === activeTurnIndex;
        tb.bar.classList.toggle("active", active);
        tb.item.classList.toggle("active", active);
    });
}

function updateActiveTurn() {
    if (!turnBars.length) {
        setActiveTurn(-1);
        return;
    }

    if (visibleTurns.size === 0) {
        // Fallback: no IO data yet — pick the last bubble whose top is
        // above the 35% reference line (your original fallback logic).
        const rootRect = turnScrollRoot.getBoundingClientRect();
        const referenceY = rootRect.top + Math.min(rootRect.height * 0.35, 220);

        let bestIdx = 0;
        for (let i = 0; i < turnBars.length; i++) {
            if (turnBars[i].bubble.getBoundingClientRect().top <= referenceY) bestIdx = i;
            else break;
        }
        setActiveTurn(bestIdx);
        return;
    }

    // Pick the turn whose vertical center is closest to the viewport center.
    const rootRect = turnScrollRoot.getBoundingClientRect();
    const viewportCenter = rootRect.top + rootRect.height / 2;

    let bestIdx  = -1;
    let bestDist = Infinity;

    for (const [idx] of visibleTurns) {
        const tb = turnBars[idx];
        if (!tb) continue;
        const rect = tb.bubble.getBoundingClientRect();
        const bubbleCenter = rect.top + rect.height / 2;
        const dist = Math.abs(bubbleCenter - viewportCenter);
        if (dist < bestDist) { bestDist = dist; bestIdx = idx; }
    }

    setActiveTurn(bestIdx);
}

function scheduleActiveTurnUpdate() {
    if (turnMapScrollFrame !== null) return;

    turnMapScrollFrame = requestAnimationFrame(() => {
        turnMapScrollFrame = null;
        updateActiveTurn();
    });
}

function buildTurnMap() {
    if (!turnMapBarsEl || !turnMapListEl) return;

    const bubbles = Array.from(outputDiv.children).filter(node =>
        node instanceof HTMLElement && node.classList.contains("msg") && node.classList.contains("user")
    );

    if (turnObserver) {
        turnObserver.disconnect();
        turnObserver = null;
    }

    visibleTurns.clear();
    turnMapBarsEl.replaceChildren();
    turnMapListEl.replaceChildren();
    turnBars = [];

    bubbles.forEach(bubble => {
        const preview = getTurnPreview(bubble);
        if (!preview) return;

        if (!bubble.dataset.turnId) {
            bubble.dataset.turnId = `turn-${++turnIdCounter}`;
        }

        const bar = document.createElement("div");
        bar.className = "turn-bar";
        bar.dataset.turnId = bubble.dataset.turnId;
        bar.setAttribute("role", "button");
        bar.setAttribute("tabindex", "0");

        const turnNumber = turnBars.length + 1;
        bar.setAttribute("aria-label", `Go to turn ${turnNumber}: ${preview}`);
        bar.title = `${turnNumber}. ${preview}`;

        const navigate = () => scrollToUserBubble(bubble);
        bar.addEventListener("click", navigate);
        bar.addEventListener("keydown", event => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                navigate();
            }
        });

        const item = document.createElement("button");
        item.type = "button";
        item.className = "turn-map-item";
        item.dataset.turnId = bubble.dataset.turnId;
        item.setAttribute("role", "option");

        const idx = document.createElement("span");
        idx.className = "tm-idx";
        idx.textContent = String(turnNumber);

        const prev = document.createElement("span");
        prev.className = "tm-preview";
        prev.textContent = preview;

        item.append(idx, prev);
        item.addEventListener("click", navigate);

        turnMapBarsEl.appendChild(bar);
        turnMapListEl.appendChild(item);

        turnBars.push({ bar, item, bubble, preview, turnId: bubble.dataset.turnId });
    });

    if (!turnBars.length) {
        const empty = document.createElement("div");
        empty.className = "turn-map-empty";
        empty.textContent = "No turns yet";
        turnMapListEl.appendChild(empty);
        setActiveTurn(-1);
        return;
    }

    /* Observe against the viewport. The chat is currently scrolled by .content,
       so using .output as the observer root would never see those movements. */
    turnObserver = new IntersectionObserver(entries => {
        for (const entry of entries) {
            const idx = Number(entry.target.dataset.turnMapWatch);
            if (!Number.isInteger(idx)) continue;

            if (entry.isIntersecting) visibleTurns.set(idx, entry.intersectionRatio);
            else visibleTurns.delete(idx);
        }

        scheduleActiveTurnUpdate();
    }, {
        root: null,
        rootMargin: "-50px 0px -55% 0px",
        threshold: [0, 0.1, 0.25, 0.5, 0.75, 1]
    });

    turnBars.forEach((tb, index) => {
        tb.bubble.dataset.turnMapWatch = String(index);
        turnObserver.observe(tb.bubble);
    });

    activeTurnIndex = -1;
    scheduleActiveTurnUpdate();
}

function refreshTurnMap() {
    if (!turnMapBarsEl || !turnMapListEl) {
        if (!ensureTurnMapMarkup()) return;
    }

    if (turnMapBuildFrame !== null) return;

    turnMapBuildFrame = requestAnimationFrame(() => {
        turnMapBuildFrame = null;
        buildTurnMap();
    });
}

function refreshTurnMapNow() {
    if (!ensureTurnMapMarkup()) return;

    if (turnMapBuildFrame !== null) {
        cancelAnimationFrame(turnMapBuildFrame);
        turnMapBuildFrame = null;
    }

    buildTurnMap();
}

function scrollToUserBubble(bubble) {
    if (!bubble || !bubble.isConnected) return;

    const rootRect = turnScrollRoot.getBoundingClientRect();
    const bubbleRect = bubble.getBoundingClientRect();

    const target = turnScrollRoot.scrollTop +
        (bubbleRect.top - rootRect.top) - 12;

    stickToBottom = false;

    turnScrollRoot.scrollTo({
        top: Math.max(0, target),
        behavior: "smooth"
    });

    scheduleActiveTurnUpdate();
}

function maybeStickToBottom() {
    const gap = outputDiv.scrollHeight - outputDiv.scrollTop - outputDiv.clientHeight;
    stickToBottom = gap < 60;
}

turnScrollRoot.addEventListener("scroll", scheduleActiveTurnUpdate, { passive: true });

outputDiv.addEventListener("scroll", () => {
    maybeStickToBottom();
    scheduleActiveTurnUpdate();
}, { passive: true });

window.addEventListener("resize", () => {
    const nextRoot = document.querySelector(".content") || outputDiv;

    if (nextRoot !== turnScrollRoot) {
        turnScrollRoot = nextRoot;
        refreshTurnMapNow();
    }

    scheduleActiveTurnUpdate();
});

function initTurnMap() {
    injectTurnMapStyles();

    if (!ensureTurnMapMarkup()) {
        console.error("Turn map: .content container not found");
        return;
    }

    buildTurnMap();

    if (turnMapMO) turnMapMO.disconnect();

    turnMapMO = new MutationObserver(mutations => {
        let turnsChanged = false;

        for (const mutation of mutations) {
            if (mutation.target !== outputDiv || mutation.type !== "childList") continue;

            const relevant = [...mutation.addedNodes, ...mutation.removedNodes].some(node =>
                node instanceof HTMLElement && node.classList.contains("msg") && node.classList.contains("user")
            );

            if (relevant) {
                turnsChanged = true;
                break;
            }
        }

        if (turnsChanged) refreshTurnMap();
    });

    turnMapMO.observe(outputDiv, { childList: true });
    refreshTurnMap();
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initTurnMap);
} else {
    initTurnMap();
}

/* ---------------------------------------------------------------- save btn */

saveBtn.addEventListener("click", async () => {
    try {
        const result = await saveCurrentChat(0);
        if (result === null) {
            console.warn("Save skipped (busy).");
            return;
        }
        console.log("Chat saved.", result);
    } catch (err) {
        console.error("Save failed:", err);
        // toast: "Save failed"
    }
});