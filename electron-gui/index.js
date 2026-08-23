// ==========================================
// 1. GLOBAL DOM ELEMENTS & VARIABLES
// ==========================================
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

const list = document.getElementById("m-list");
const btn = document.getElementById("m-btn");


const imgBtn = document.getElementById("img-btn");
const imgInput = document.getElementById("img-input");
const attachmentTray = document.getElementById("attachment-tray");

const toggleInput = document.querySelector(".switch input[type='checkbox']");

let isCollapsed = true;
let isHistory = false;
let isOpened = false;

let currentChatId = null;

let saveInFlight = null;
let isSaving = false;

let isSearching = false;


let pendingImages = [];

const isEmpty = () => outputDiv.children.length === 0;
console.log({
    renameBtn,
    pinBtn,
    delBtn,
    btn,
    list,
    imgBtn,
    imgInput
});
// ==========================================
// 2. BUSY STATE
// ==========================================
function setBusy(busy) {
    isSaving = busy;

    newBtn.style.pointerEvents = busy ? "none" : "auto";
    chatList.style.pointerEvents = busy ? "none" : "auto";

    document.body.style.cursor = busy ? "progress" : "default";
}


// ==========================================
// 3. SAVE CURRENT CHAT
// ==========================================
async function saveCurrentChat() {

    if (saveInFlight) return saveInFlight;

    saveInFlight = (async () => {

        setBusy(true);

        try {

            const form = new FormData();

            if (currentChatId) {
                form.append("chat_id", currentChatId);
            }

            const saveRes = await fetch("http://localhost:5000/save", {
                method: "POST",
                body: form
            });

            const saveJson = await saveRes.json();

            if (saveJson.id) {
                currentChatId = saveJson.id;
            }

            await fetch("http://localhost:5000/prune", { method: "POST" });

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


// ==========================================
// 4. SIDEBAR HISTORY
// ==========================================

function isPinned(value) {
    return value === true ||
        value === 1 ||
        value === "1" ||
        String(value).trim().toLowerCase() === "true";
}


async function loadSidebarHistory() {
    try {
        const response = await fetch(
            "http://localhost:5000/history"
        );

        const payload = await response
            .json()
            .catch(() => null);

        if (!response.ok) {
            throw new Error(
                payload?.message ||
                `History request failed: ${response.status}`
            );
        }

        // Supports both the current array response and a future
        // { history: [...] } response without breaking the UI.
        const history = Array.isArray(payload)
            ? payload
            : Array.isArray(payload?.history)
                ? payload.history
                : [];

        chatList.innerHTML = "";

        history.forEach((item) => {
            const div = document.createElement("div");

            div.dataset.chatId = item.id;
            div.dataset.pinned = isPinned(item.pinned)
                ? "1"
                : "0";

            div.insertAdjacentHTML("afterbegin", `
                <p>
                    ${
                        isPinned(item.pinned)
                            ? '<span class="pin">📌</span>'
                            : ""
                    }
                    ${escapeHtml(item.title || "Legacy Chat")}
                </p>

                <button
                    class="menu-btn"
                    data-id="${escapeHtml(item.id)}"
                    aria-label="Chat menu"
                >
                    <img
                        src="./assets/ellipsis-vertical-solid-full.svg"
                        alt=""
                    >
                </button>
            `);

            div.onclick = async (event) => {
                if (event.target.closest(".menu-btn")) return;
                if (isSaving) return;

                header.innerText =
                    item.title || "Legacy Chat";

                header.style.setProperty(
                    "--snippet",
                    JSON.stringify(
                        item.snippet ||
                        "No summary available"
                    )
                );

                await loadChat(item.id);
            };

            chatList.appendChild(div);
        });

    } catch (error) {
        console.error(
            "History load failed:",
            error
        );
    }
}

loadSidebarHistory();


// ==========================================
// 5. SIDEBAR MENU (DELEGATED)
// ==========================================
chatList.addEventListener("click", (e) => {

    const btn = e.target.closest(".menu-btn");

    if (!btn) return;

    e.stopPropagation();

    const rect = btn.getBoundingClientRect();

    menu.style.left = rect.right + "px";
    menu.style.top = rect.top + "px";

    menu.style.display = "flex";

    menu.dataset.id = btn.dataset.id;

});


document.addEventListener("click", () => {

    menu.style.display = "none";

});

async function askRename() {
  return new Promise(resolve => {
    // backdrop
    const overlay = document.createElement("div");
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.background = "rgba(0,0,0,0.4)";
    overlay.style.backdropFilter = "blur(2px)";
    overlay.style.zIndex = "99998";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";

    // modal box
    const modal = document.createElement("div");
    modal.style.background = "#fff";
    modal.style.padding = "24px 22px";
    modal.style.borderRadius = "14px";
    modal.style.boxShadow = "0 20px 60px rgba(0,0,0,0.25)";
    modal.style.zIndex = "99999";
    modal.style.minWidth = "320px";
    modal.style.fontFamily = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif";

    modal.innerHTML = `
      <div style="font-weight:600; margin-bottom:12px; font-size:15px;">Rename chat</div>
      <input id="rename-input" placeholder="New chat name" style="
        width:100%; padding:10px 12px; border:1px solid #e2e2e2; border-radius:8px;
        outline:none; font-size:14px; margin-bottom:14px; box-sizing:border-box;
      ">
      <div style="display:flex; gap:8px; justify-content:flex-end;">
        <button id="rename-cancel" style="
          padding:8px 14px; border:1px solid #e2e2e2; background:#f7f7f7; border-radius:8px;
          cursor:pointer; font-size:13px;
        ">Cancel</button>
        <button id="rename-ok" style="
          padding:8px 16px; border:none; background:#111; color:white; border-radius:8px;
          cursor:pointer; font-size:13px;
        ">OK</button>
      </div>
    `;

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    const input = modal.querySelector("#rename-input");
    input.focus();
    input.select();

    const close = (value) => {
      overlay.remove();
      resolve(value);
    };

    // prevent clicks inside modal from closing
    modal.addEventListener("click", e => e.stopPropagation());

    // outside click closes
    overlay.addEventListener("click", () => close(null));

    // buttons
    modal.querySelector("#rename-ok").onclick = () => {
      const value = input.value.trim();
      if (value) close(value);
    };
    modal.querySelector("#rename-cancel").onclick = () => close(null);

    // Enter to submit, Esc to cancel
    input.addEventListener("keydown", e => {
      if (e.key === "Enter") {
        const value = input.value.trim();
        if (value) close(value);
      }
      if (e.key === "Escape") close(null);
    });
  });
}
// ==========================================
// 6. MENU ACTIONS
// ==========================================
renameBtn.addEventListener("click", () => chatMenu(1));
pinBtn.addEventListener("click", () => chatMenu(2));
delBtn.addEventListener("click", () => chatMenu(3));


async function chatMenu(type) {

    const id = menu.dataset.id;

    if (!id) return;

    switch (type) {

        case 1: {
            menu.style.display = "none";
            const newName = await askRename();

            if (!newName) return;

            const form = new FormData();
            form.append("id", id);
            form.append("title", newName);

            await fetch("http://localhost:5000/rename", {
                method: "POST",
                body: form
            });

            await loadSidebarHistory();


            break;
        }

        case 2: {
            menu.style.display = "none";

            const pinForm = new FormData();
            pinForm.append("id", id);

            pinBtn.disabled = true;

            try {
                const response = await fetch(
                    "http://localhost:5000/pin",
                    {
                        method: "POST",
                        body: pinForm,
                    }
                );

                const result = await response
                    .json()
                    .catch(() => ({}));

                if (!response.ok) {
                    throw new Error(
                        result.message ||
                        `Pin request failed: ${response.status}`
                    );
                }

                // Wait for the refreshed, backend-sorted list.
                await loadSidebarHistory();

            } catch (error) {
                console.error(
                    "Pin toggle failed:",
                    error
                );

            } finally {
                pinBtn.disabled = false;
                menu.dataset.id = "";
            }

            break;
        }

        case 3: {
            menu.style.display = "none";
            const deleteForm = new FormData();
            deleteForm.append("id", id);

            await fetch("http://localhost:5000/delete", {
                method: "POST",
                body: deleteForm
            });
            loadSidebarHistory();
            outputDiv.innerHTML = "";
            header.innerText = "New Chat";
            header.style.setProperty("--snippet", `"Ask something to start a new chat"`);

            break;
        }

        default:
            console.error("Unknown menu action");

    }

}


// ==========================================
// 7. LOAD CHAT
// ==========================================
async function loadChat(chatId) {
    try {
        if (saveInFlight) {
            await saveInFlight;
        }

        const hadMessages = outputDiv.children.length > 0;

        if (hadMessages && !isHistory) {
            await saveCurrentChat();
        }

        // Set this only after the previous chat is saved.
        currentChatId = chatId;
        isHistory = true;

        outputDiv.innerHTML = "";

        const response = await fetch(
            `http://localhost:5000/chat/${chatId}`
        );

        if (!response.ok) {
            throw new Error(
                `Chat request failed: ${response.status}`
            );
        }

        const data = await response.json();
        const interactions = data.content.split("[TURN_END]");

        interactions.forEach((line) => {
            line = line.trim();
            if (!line) return;

            const parts = line.split(" | ");

            parts.forEach((part) => {
                part = part.trimStart();

                if (part.startsWith("User: ")) {
                    appendMessage(
                        outputDiv,
                        "user",
                        part.slice(6)
                    );
                } else if (part.startsWith("Assistant: ")) {
                    appendMessage(
                        outputDiv,
                        "assistant",
                        part.slice(11)
                    );
                }
            });
        });
    } catch (err) {
        console.error("Chat load failed:", err);
    }
}


// ==========================================
// 8. NEW CHAT
// ==========================================
newBtn.addEventListener("click", async () => {

    if (isSaving) return;

    if (!isHistory && !isEmpty()) {

        await saveCurrentChat();

    }

    outputDiv.innerHTML = "";

    header.innerText = "New Chat";
    header.style.setProperty("--snippet", `"Ask something to start a new chat"`);

    isHistory = false;
    currentChatId = null;

});


// ==========================================
// 9. SIDEBAR COLLAPSE
// ==========================================
document.getElementById("btn").addEventListener("click", () => {

    if (isCollapsed) {

        container.style.gridTemplateColumns = "300px 1fr";
        newBtn.style.display = "flex";
        chatList.style.display = "block";

        isCollapsed = false;

    } else {

        container.style.gridTemplateColumns = "70px 1fr";
        newBtn.style.display = "none";
        chatList.style.display = "none";

        isCollapsed = true;

    }

});


// ==========================================
// 10. MODE SELECTOR
// ==========================================
// 1. Just toggle the menu when the button is clicked. No propagation stopping.

toggleInput.addEventListener("change", (event) => {
        
    isSearching = event.target.checked;
    
});



// ==========================================
// 11. TEXTAREA AUTO EXPAND
// ==========================================
ain.addEventListener("input", () => {

    ain.style.height = "auto";
    ain.style.height = ain.scrollHeight + "px";

    if (ain.scrollHeight > 200) {
        ain.style.overflowY = "scroll";
    } else {
        ain.style.overflowY = "hidden";
    }

});


// ==========================================
// 12. MESSAGE RENDER
// ==========================================
function escapeHtml(str) {

    if (!str) return "";

    return String(str)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");

}

function updateMessage(element, role, content) {

    if (role === "assistant") {

        try {
            element.innerHTML =
                window.electronAPI.parseMarkdown(content);

        } catch {
            element.innerHTML =
                escapeHtml(content).replaceAll("\n", "<br>");
        }

    } else {

        element.innerHTML =
            escapeHtml(content).replaceAll("\n", "<br>");
    }
}

function appendMessage(outputDiv, role, content = "") {

    const cls = role === "user" ? "msg user" : "msg assistant";
    const meta = role === "user"
        ? "You"
        : "Assistant";

    const message = document.createElement("div");
    message.className = cls;

    const metaElement = document.createElement("div");
    metaElement.className = "meta";
    metaElement.innerText = meta;

    const textElement = document.createElement("div");
    textElement.className = "text";

    message.appendChild(metaElement);
    message.appendChild(textElement);

    outputDiv.appendChild(message);

    // Render initial content
    updateMessage(textElement, role, content);

    outputDiv.scrollTo({
        top: outputDiv.scrollHeight,
        behavior: "smooth"
    });

    return textElement;
}


async function sendMessage() {
    const query = ain.value.trim();

    if (!query && pendingImages.length === 0) return;


    // append text bubble
    if (query) appendMessage(outputDiv, "user", query);

    // append images bubble (if any)
    if (pendingImages.length > 0) {
        const imagesHtml = pendingImages
            .map((file) => {
                const url = URL.createObjectURL(file);
                return `<img src="${url}" style="max-width:260px;border-radius:12px;margin:4px;" />`;
            })
            .join("");

        outputDiv.insertAdjacentHTML(
            "beforeend",
            `
      <div class="msg user">
        <div class="meta">Pics</div>
        <div class="text">${imagesHtml}</div>
      </div>
      `
        );
    }

    // reset input UI
    ain.value = "";
    ain.style.height = "auto";

    // thinking bubble
    const loadingId = "loading-" + Date.now();
    outputDiv.insertAdjacentHTML(
        "beforeend",
        `<div id="${loadingId}" class="msg assistant thinking">Assistant is thinking...</div>`
    );
    outputDiv.scrollTop = outputDiv.scrollHeight;

    // we're now on an active chat session (not viewing history)
    isHistory = false;

    try {
        const form = new FormData();
        form.append("query", query);
        form.append("websearch", isSearching);

        pendingImages.forEach((img) => form.append("image", img));

        // continue the same long-term chat if we loaded one
        if (currentChatId) form.append("chat_id", currentChatId);

        pendingImages = [];
        attachmentTray.innerHTML = "";
        imgInput.value = "";

        const response = await fetch("http://localhost:5000/ask", {
            method: "POST",
            body: form,
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        if (!response.body) {
            throw new Error("Response body is not streamable");
        }
    

        // clear image UI state
        

        // remove loading bubble
        document.getElementById(loadingId)?.remove();


        // Create empty assistant message
        const assistantTextElement =
            appendMessage(outputDiv, "assistant", "");


        // Accumulate the raw model response
        let fullResponse = "";


        // Start reading SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let buffer = "";

        while (true) {

            const { value, done } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, {
                stream: true
            });


            // SSE events are separated by blank lines
            const events = buffer.split("\n\n");

            // Keep incomplete event for next network chunk
            buffer = events.pop();


            for (const event of events) {

                const line = event.trim();

                if (!line.startsWith("data: "))
                    continue;

                const data = line.slice(6);


                if (data === "[DONE]") {
                    continue;
                }


                try {

                    const parsed = JSON.parse(data);

                    const token = parsed.token;

                    if (!token)
                        continue;


                    // Add token to complete response
                    fullResponse += token;


                    // Re-render existing message
                    updateMessage(
                        assistantTextElement,
                        "assistant",
                        fullResponse
                    );


                    // Keep newest text visible
                    outputDiv.scrollTop =
                        outputDiv.scrollHeight;

                } catch (error) {

                    console.error(
                        "Stream parsing error:",
                        error,
                        data
                    );

                }
            }
        }


        // refresh sidebar
        await loadSidebarHistory();

        // refresh tooltip summary
        try {
            const sumRes = await fetch("http://localhost:5000/sum", { method: "POST" });
            const summaryBlock = await sumRes.json();

            header.innerText = summaryBlock.title || "New Chat";
            header.style.setProperty(
                "--snippet",
                `"${summaryBlock.snippet || "No summary available."}"`
            );
        } catch { }

    } catch (err) {
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) {
            loadingEl.innerText =
                `Error: ${err.message || "Request failed."}`;
        }
        console.error(err);
    }
}

// --- image button ---
imgBtn.addEventListener("click", () => imgInput.click());

// --- tray helper ---
function addToTray(files) {
    const picked = files ? Array.from(files) : Array.from(imgInput.files || []);
    if (!picked.length) return;

    picked.forEach((file) => {
        pendingImages.push(file);

        const url = URL.createObjectURL(file);

        const wrapper = document.createElement("div");
        wrapper.className = "attachment-item";
        wrapper.innerHTML = `
      <img src="${url}" />
      <div class="attachment-remove">×</div>
    `;

        wrapper.querySelector(".attachment-remove").addEventListener("click", () => {
            const idx = pendingImages.indexOf(file);
            if (idx !== -1) pendingImages.splice(idx, 1);
            wrapper.remove();
        });

        attachmentTray.appendChild(wrapper);
    });

    // allow picking same file again later
    imgInput.value = "";
}

// --- file picker ---
imgInput.addEventListener("change", () => addToTray());

// --- paste images ---
document.addEventListener("paste", (e) => {
    const clipboard = e.clipboardData;
    if (!clipboard) return;

    const blobs = [];
    for (const item of Array.from(clipboard.items)) {
        if (item.type && item.type.startsWith("image/")) {
            const blob = item.getAsFile();
            if (blob) blobs.push(blob);
        }
    }

    if (blobs.length > 0) addToTray(blobs);
});

// --- enter to send ---
ain.addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        await sendMessage();
    }
});