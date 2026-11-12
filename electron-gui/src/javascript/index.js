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
const headerSnippet = document.getElementById("sum-snippet");
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
let titleRevision = 0;

const isEmpty = () => outputDiv.children.length === 0;

const API = "http://localhost:5000";

function setChatHeading(title, snippet) {
    header.textContent = title;
    headerSnippet.textContent = snippet;
    titleRevision++;
}

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

// Cycle shimmer color each pass
const shimmerColors = ['#c7d2fe', '#a7f3d0', '#fde68a', '#fca5a5'];
let shimmerIdx = 0;

const svgStyle = document.createElement("style");
svgStyle.textContent = `
    img[src$=".svg"] {
        filter: brightness(0) saturate(100%) invert(100%);
    }
`;

document.head.appendChild(svgStyle);

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

                await loadChat(id, item.title, item.snippet);
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
            background: "#2d2e30",
            color: "#e8eaed",
            padding: "24px 22px",
            borderRadius: "14px",
            boxShadow: "0 20px 60px rgba(0,0,0,0.25)",
            zIndex: "99999",
            minWidth: "320px",
            fontFamily: "\"SN Pro\", system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
        });

        modal.innerHTML = `
            <div style="font-weight:600;margin-bottom:12px;font-size:15px">Rename chat</div>
            <input id="rename-input" placeholder="New chat name" style="width:100%;padding:10px 12px;border:1px solid #3c4043;background:#35363a;color:#e8eaed;border-radius:8px;outline:none;font-size:14px;margin-bottom:14px;box-sizing:border-box">
            <div style="display:flex;gap:8px;justify-content:flex-end">
                <button id="rename-cancel" style="padding:8px 14px;border:1px solid #3c4043;background:#35363a;color:#e8eaed;border-radius:8px;cursor:pointer;font-size:13px">Cancel</button>
                <button id="rename-ok" style="padding:8px 16px;border:none;background:#8ab4f8;color:#17171a;border-radius:8px;cursor:pointer;font-size:13px">OK</button>
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

            if (id === currentChatId) setChatHeading(newName, headerSnippet.textContent);
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

async function loadChat(chatId, title, snippet) {
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
            setChatHeading(title || "Legacy Chat", snippet || "No summary available.");
            isHistory = true;
            stickToBottom = true;
            outputDiv.replaceChildren();

            for (const message of data.messages) {
                if (!message || !["user", "assistant"].includes(message.role) ||
                    typeof message.content !== "string") continue;

                appendStoredMessage(outputDiv, message.role, message.content, false);
            }

            if (stickToBottom) outputDiv.scrollTop = outputDiv.scrollHeight;

        } catch (err) {
            console.error("Chat load failed:", err);
        }
    })();

    navigationInFlight = navTask;

    try {
        await navTask;
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

    setChatHeading("New chat", "Ask something to start a new chat");

    isHistory = false;
    currentChatId = null;
    currentDraftId = crypto.randomUUID();
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
    container.style.setProperty("--sidebar-width", isCollapsed ? "70px" : "min(300px, 48vw)");
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
        controlBtn.setAttribute("aria-label", "Stop answering");
        controlBtn.disabled = isStopping;
        ain.disabled = false;
        imgBtn.disabled = true;
        toggleInput.disabled = true;
        controlBtn.onclick = stopAnswering;
    } else {
        controlImg.src = "../../assets/arrow-right-long-solid-full.svg";
        controlBtn.style.setProperty("--snippet", '"Send message"');
        controlBtn.setAttribute("aria-label", "Send message");
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

    const generationFiles = [...pendingAttachments];
    const split = splitOversizedInput(query);
    if (split) {
        query = split.inline;
        generationFiles.push(split.file);
    }

    const generationId = ++currentGenerationId;
    const controller = new AbortController();
    currentAbortController = controller;


    const requestChatId = currentChatId;
    const requestDraftId = currentDraftId;
    const requestTitleRevision = titleRevision;
    const startedAt = performance.now();
    let requestAccepted = false;

    setGenerationUI(true);
    const userText = appendLiveUserTurn(outputDiv, query, generationFiles);
    clearInputUI();

    const loadingId = `loading-${Date.now()}-${generationId}`;
    const loading = document.createElement("div");
    loading.id = loadingId;
    loading.className = "msg assistant thinking";
    if (generationFiles.length) loading.classList.add("uploading");
    loading.textContent = generationFiles.length
        ? "Uploading attachments..."
        : "Thinking";

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
        workingBubble = createWorkingBubble(wrapper, startedAt);
        if (stickToBottom) outputDiv.scrollTop = outputDiv.scrollHeight;
        return workingBubble;
    }

    // One header displays live stages; its body accumulates reasoning.
    function ensureThinkingBubble() {
        if (thinkingBubble) return thinkingBubble;
        const wrapper = ensureWorkingBubble().bubble.parentElement;
        thinkingBubble = createThinkingBubble(wrapper);
        return thinkingBubble;
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

        if (!generationFiles.length) ensureThinkingBubble();

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
        ensureThinkingBubble();
        if (!response.body) throw new Error("Response body is not streamable");

        let fullResponse = "";
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let streamFinished = false;

        function handleEvent(event) {
            if (generationId !== currentGenerationId || controller.signal.aborted) {
                return false;
            }
            const line = event.trim();
            if (!line.startsWith("data: ")) return false;

            const data = line.slice(6);
            if (data === "[DONE]") return true;

            try {
                const packet = JSON.parse(data);

                const stage = packet.backend_stage;

                if (stage && typeof stage === "object" && !Array.isArray(stage)) {
                    const summary =
                        typeof stage.summary === "string" ? stage.summary : "";

                    const tool =
                        typeof stage.tool === "string" ? stage.tool.trim() : "";

                    const bubble = ensureThinkingBubble();
                    bubble.setStage(summary, tool);
                    bubble.appendStage(summary, tool);
                }

                if (Array.isArray(packet.sandbox_files)) {
                    showUploadedFiles(userText, packet.sandbox_files);

                }
                const reasoning = packet.reasoning;
                if (reasoning) {
                    ensureThinkingBubble().append(reasoning);
                }

                const token = packet.token;
                if (token) {
                    if (!assistantTextElement) {
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
            if (generationId === currentGenerationId && requestTitleRevision === titleRevision) {
                setChatHeading(summaryBlock.title || "New chat", summaryBlock.snippet || "No summary available.");
            }
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
            loading.classList.remove("uploading");
            loading.classList.add("error");
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

function createThinkingBubble(parent) {
    const SPARK_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="currentColor"><path d="M12 2.5l1.9 5.6a2 2 0 0 0 1.26 1.26l5.6 1.9-5.6 1.9a2 2 0 0 0-1.26 1.26L12 20.5l-1.9-5.6a2 2 0 0 0-1.26-1.26L3.24 11.78l5.6-1.9a2 2 0 0 0 1.26-1.26L12 2.5z"/></svg>';
    function createSparkSvg() {
        const t = document.createElement("template");
        t.innerHTML = SPARK_SVG.trim();
        return t.content.firstElementChild;
    }

    const TOOL_ICONS = new Map([
        ["web_call",        "../../assets/globe-solid-full.svg"],
        ["memory_call",     "../../assets/brain-solid-full.svg"],
        ["search_current_chat",     "../../assets/brain-solid-full.svg"],
        ["get_chat_turn",     "../../assets/file-pen-solid-full.svg"],
        ["call_coder",      "../../assets/wrench-solid-full.svg"],
        ["sandbox_exec",    "../../assets/gears-solid-full.svg"],
        ["sandbox_install", "../../assets/cubes-solid-full.svg"],
        ["write_notebook",  "../../assets/file-pen-solid-full.svg"],
    ]);

    const bubble = document.createElement("div");
    bubble.className = "thinking-bubble active";
    const header = document.createElement("button");
    header.type = "button";
    header.className = "thinking-header";
    header.disabled = true;
    // AI-Studio style spark icon (kept as "spinner" so existing JS/CSS
    // hooks still apply). It shimmers while active and holds a tool icon
    // when the model is running a tool.
    const spinner = document.createElement("span");
    spinner.className = "thinking-icon thinking-spark";
    spinner.appendChild(createSparkSvg());
    
    const labelEl = document.createElement("span");
    labelEl.className = "thinking-label";
    labelEl.textContent = "Thinking";
    labelEl.addEventListener('animationiteration', () => {
        shimmerIdx = (shimmerIdx + 1) % shimmerColors.length;
        labelEl.style.setProperty('--shimmer-color', shimmerColors[shimmerIdx]);
    });
    const chevron = document.createElement("span");
    chevron.className = "thinking-chevron";
    header.append(spinner, labelEl, chevron);

    const body = document.createElement("div");
    body.className = "thinking-body";
    body.id = `thinking-body-${currentGenerationId}`;

    const text = document.createElement("div");
    text.className = "thinking-text";
    body.appendChild(text);

    // Footer row, like AI Studio ("Expand to see model thoughts" /
    // "Collapse to hide model thoughts").
    const footer = document.createElement("button");
    footer.type = "button";
    footer.className = "thinking-footer";
    footer.hidden = true;

    const footerText = document.createElement("span");
    footerText.className = "thinking-footer-text";

    const footerChevron = document.createElement("span");
    footerChevron.className = "thinking-footer-chevron";

    footer.append(footerText, footerChevron);
    bubble.append(header, body, footer);
    parent.appendChild(bubble);

    let finished = false;
    let reasoningSegment = null;
    let lastLoggedStage = null;

    function setOpen(open) {
        bubble.classList.toggle("open", open);
        body.hidden = !open;
        header.setAttribute("aria-expanded", String(open));
        footer.setAttribute("aria-expanded", String(open));
        footerText.textContent = open
            ? "Collapse to hide model thoughts"
            : "Expand to see model thoughts";
    }

    setOpen(false);
    for (const button of [header, footer]) {
        button.setAttribute("aria-controls", body.id);
        button.addEventListener("click", () => setOpen(body.hidden));
    }

    function updateAfterAppend(followReasoning) {
        if (!bubble.classList.contains("has-content")) {
            bubble.classList.add("has-content");
            header.disabled = false;
            footer.hidden = false;
            setOpen(true);
            followReasoning = true;
        }
        if (followReasoning) body.scrollTop = body.scrollHeight;
        if (stickToBottom) outputDiv.scrollTop = outputDiv.scrollHeight;
    }

    const isFollowingReasoning = () => !body.hidden &&
        body.scrollHeight - body.scrollTop - body.clientHeight <= 24;

    return {
        setStage(stage, tool = "") {
            if (finished || typeof stage !== "string") return;

            // AI Studio keeps the header label as the shimmering "Thinking";
            // the live stage text is surfaced as a note inside the body
            // (see appendStage) rather than overwriting the header.
            labelEl.textContent = "Thinking";

            const toolName = typeof tool === "string" ? tool.trim() : "";

            if (toolName) {
                // Model is running a tool: show its icon in the header.
                spinner.className = "thinking-icon thinking-tool-icon";
                spinner.title = toolName;
                const icon = document.createElement("img");
                icon.src = TOOL_ICONS.get(toolName)
                    || "../../assets/wrench-solid-full.svg";
                icon.alt = "";
                icon.draggable = false;
                spinner.replaceChildren(icon);
            } else {
                // No tool: restore the shimmering "Thinking" spark.
                spinner.className = "thinking-icon thinking-spark";
                spinner.removeAttribute("title");
                spinner.replaceChildren(createSparkSvg());
            }

            spinner.setAttribute("aria-hidden", "true");
        },
        append(chunk) {
            if (finished || typeof chunk !== "string" || !chunk) return;
            const followReasoning = isFollowingReasoning();
            if (!reasoningSegment) {
                reasoningSegment = document.createElement("div");
                reasoningSegment.className = "thinking-reasoning-segment";
                reasoningSegment.appendChild(document.createTextNode(""));
                text.appendChild(reasoningSegment);
            }
            reasoningSegment.firstChild.appendData(chunk);
            updateAfterAppend(followReasoning);
        },
        appendStage(stage, tool = "") {
            if (finished || typeof stage !== "string") return;

            const cleaned = stage.trim();
            const key = JSON.stringify([tool, cleaned]);

            if (!cleaned || key === lastLoggedStage) return;
            lastLoggedStage = key;

            // Keep generic thinking states in the header only.
            const normalized = cleaned
                .toLowerCase()
                .replace(/[.…]+$/u, "")
                .trim();

            if ([
                "thinking",
                "assistant is thinking",
                "coder is thinking"
            ].includes(normalized)) {
                return;
            }

            const note = document.createElement("div");
            note.className = "thinking-stage-note";
            note.dataset.tool = tool;
            note.textContent = tool
                ? `[${tool}] ${cleaned}`
                : cleaned;

            const followReasoning = isFollowingReasoning();
            text.appendChild(note);
            reasoningSegment = null;
            updateAfterAppend(followReasoning);
        },
        finish() {
            if (finished) return;
            finished = true;
            bubble.classList.remove("active");
            if (text.textContent.trim()) {
                spinner.className = "thinking-icon thinking-spark";
                spinner.removeAttribute("title");
                spinner.replaceChildren(createSparkSvg());
            } else {
                bubble.remove();
            }
        }
    };
}



function createWorkingBubble(parent, startedAt) {
    const bubble = document.createElement("div");
    bubble.className = "working-note";

    const labelEl = document.createElement("span");
    labelEl.className = "working-label";
    labelEl.textContent = "Working for 0s";

    bubble.append(labelEl);
    parent.appendChild(bubble);

    let stopped = false;

    const timer = setInterval(() => {
        if (!stopped) {
            labelEl.textContent =
                "Working for " +
                formatWorkTime(performance.now() - startedAt);
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
            labelEl.textContent =
                "Worked for " +
                formatWorkTime(performance.now() - startedAt);

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
    controlBtn.setAttribute("aria-label", "Stopping");

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
const MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024
const MAX_ATTACHMENTS_BYTES = 128 * 1024 * 1024
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

/* -------------------------------------------------------------- turn map */

const turnsColumn = document.querySelector(".turns-col");
const turnMapBars = turnsColumn.querySelector(".turn-map-bars");
const turnMapList = turnsColumn.querySelector(".turn-map-list");
let turns = [];
let activeTurnIndex = -1;
let activeTurnFrame = null;
let turnJump = null;

function getTurnPreview(bubble) {
    const body = bubble.querySelector(".stored-user-text");
    const content = bubble.querySelector(".text");
    const attachment = content?.querySelector(".sandbox-file-name, .message-image, .image-missing");
    const text = (body?.textContent || (!attachment && content?.textContent) || "")
        .replace(/\s+/g, " ").trim();

    if (text) return text.length > 80 ? text.slice(0, 80) + "…" : text;
    if (attachment?.classList.contains("sandbox-file-name")) return attachment.textContent;
    return attachment ? "Attached image" : "Turn";
}

function setActiveTurn(index) {
    if (index === activeTurnIndex) return;

    const previous = turns[activeTurnIndex];
    previous?.bar.classList.remove("active");
    previous?.item.removeAttribute("aria-current");

    activeTurnIndex = index;
    const current = turns[index];
    if (!current) return;
    current.bar.classList.add("active");
    current.item.setAttribute("aria-current", "true");

    // Keep the active marker in view while reading the transcript.
    if (!turnsColumn.matches(":hover") && !turnsColumn.contains(document.activeElement)) {
        const bar = current.bar.getBoundingClientRect();
        const rail = turnMapBars.getBoundingClientRect();
        if (bar.top < rail.top) turnMapBars.scrollTop += bar.top - rail.top;
        else if (bar.bottom > rail.bottom) turnMapBars.scrollTop += bar.bottom - rail.bottom;
    }
}

function updateActiveTurn() {
    if (!turns.length) return;
    if (turnJump) {
        const index = turns.findIndex(turn => turn.bubble === turnJump.bubble);
        if (index !== -1) {
            setActiveTurn(index);
            return;
        }
    }

    const maxScroll = outputDiv.scrollHeight - outputDiv.clientHeight;
    if (maxScroll > 1 && outputDiv.scrollTop >= maxScroll - 1) {
        setActiveTurn(turns.length - 1);
        return;
    }
    if (outputDiv.scrollTop <= 1) {
        setActiveTurn(0);
        return;
    }

    const viewport = outputDiv.getBoundingClientRect();
    const referenceY = viewport.top + Math.min(viewport.height * 0.35, 220);
    // Find the last user message above the reading line.
    let left = 0;
    let right = turns.length;
    while (left < right) {
        const middle = Math.floor((left + right) / 2);
        if (turns[middle].bubble.getBoundingClientRect().top <= referenceY) left = middle + 1;
        else right = middle;
    }
    setActiveTurn(Math.max(0, left - 1));
}

function scheduleActiveTurnUpdate() {
    if (activeTurnFrame !== null) return;
    activeTurnFrame = requestAnimationFrame(() => {
        activeTurnFrame = null;
        updateActiveTurn();
    });
}

function scrollToUserBubble(bubble) {
    if (bubble.parentElement !== outputDiv) return;

    const viewport = outputDiv.getBoundingClientRect();
    const top = outputDiv.scrollTop + bubble.getBoundingClientRect().top - viewport.top - 12;
    outputDiv.scrollTop = top;

    // Multiple nearby turns may share the same clamped scroll position.
    // Keep the requested one selected until the transcript moves again.
    turnJump = { bubble, top: outputDiv.scrollTop };
    stickToBottom = bubble === turns[turns.length - 1]?.bubble &&
        outputDiv.scrollHeight - outputDiv.clientHeight - outputDiv.scrollTop <= 1;
    updateActiveTurn();
}

function buildTurnMap() {
    const bubbles = Array.from(outputDiv.children).filter(node => node.matches(".msg.user"));
    if (turnJump && !bubbles.includes(turnJump.bubble)) turnJump = null;

    const bars = document.createDocumentFragment();
    const items = document.createDocumentFragment();
    turns = bubbles.map((bubble, index) => {
        const preview = getTurnPreview(bubble);
        const number = index + 1;
        const navigate = () => scrollToUserBubble(bubble);

        const bar = document.createElement("button");
        bar.type = "button";
        bar.className = "turn-bar";
        bar.setAttribute("aria-label", `Go to turn ${number}: ${preview}`);
        bar.addEventListener("click", navigate);
        bars.appendChild(bar);

        const item = document.createElement("button");
        item.type = "button";
        item.className = "turn-map-item";
        const label = document.createElement("span");
        label.className = "tm-idx";
        label.textContent = String(number);
        const excerpt = document.createElement("span");
        excerpt.className = "tm-preview";
        excerpt.textContent = preview;
        item.append(label, excerpt);
        item.addEventListener("click", navigate);
        items.appendChild(item);

        return { bubble, bar, item };
    });

    turnsColumn.hidden = turns.length === 0;
    turnMapBars.replaceChildren(bars);
    turnMapList.replaceChildren(items);
    activeTurnIndex = -1;
    scheduleActiveTurnUpdate();
}

const turnLayoutObserver = new ResizeObserver(scheduleActiveTurnUpdate);
turnLayoutObserver.observe(outputDiv);

const turnMessageObserver = new MutationObserver(mutations => {
    let userTurnsChanged = false;
    for (const mutation of mutations) {
        for (const node of mutation.removedNodes) {
            if (!(node instanceof Element)) continue;
            turnLayoutObserver.unobserve(node);
            if (node.matches(".msg.user")) userTurnsChanged = true;
        }
        for (const node of mutation.addedNodes) {
            if (!(node instanceof Element)) continue;
            turnLayoutObserver.observe(node);
            if (node.matches(".msg.user")) userTurnsChanged = true;
        }
    }
    if (userTurnsChanged) buildTurnMap();
    else scheduleActiveTurnUpdate();
});
turnMessageObserver.observe(outputDiv, { childList: true });

outputDiv.addEventListener("scroll", () => {
    if (turnJump && Math.abs(outputDiv.scrollTop - turnJump.top) > 1) turnJump = null;
    if (!turnJump) {
        const gap = outputDiv.scrollHeight - outputDiv.scrollTop - outputDiv.clientHeight;
        stickToBottom = gap < 60;
    }
    scheduleActiveTurnUpdate();
}, { passive: true });

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
