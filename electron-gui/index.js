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

const options = document.querySelectorAll(".code, .auto, .assistant");

const imgBtn = document.getElementById("img-btn");
const imgInput = document.getElementById("img-input");
const attachmentTray = document.getElementById("attachment-tray");

let isCollapsed = true;
let isHistory = false;
let isOpened = false;

let currentChatId = null;

let saveInFlight = null;
let isSaving = false;

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
async function loadSidebarHistory() {

    try {

        const response = await fetch("http://localhost:5000/history");
        const history = await response.json();

        chatList.innerHTML = "";

        history.forEach(item => {

            const div = document.createElement("div");

            div.insertAdjacentHTML("afterbegin", `
                <p>
                    ${item.pinned ? '<span class="pin">📌</span>' : ''}
                    ${item.title}
                </p>
                <button class="menu-btn" data-id="${item.id}">
                    <img src="./assets/ellipsis-vertical-solid-full.svg">
                </button>
            `);

            div.onclick = async (e) => {

                if (e.target.closest(".menu-btn")) return;
                if (isSaving) return;

                header.innerText = item.title;
                header.style.setProperty("--snippet", `"${item.snippet}"`);

                await loadChat(item.id);
            };

            chatList.appendChild(div);

        });

    } catch (err) {

        console.error("History load failed:", err);

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


function askRename() {

    return new Promise(resolve => {

        const modal = document.createElement("div");

        modal.style.position = "fixed";
        modal.style.top = "50%";
        modal.style.left = "50%";
        modal.style.transform = "translate(-50%, -50%)";
        modal.style.background = "white";
        modal.style.padding = "20px";
        modal.style.borderRadius = "10px";
        modal.style.boxShadow = "0 10px 30px rgba(0,0,0,0.2)";
        modal.style.zIndex = "99999";

        modal.innerHTML = `
            <input id="rename-input" placeholder="New chat name" style="width:200px;padding:5px">
            <button id="rename-ok">OK</button>
        `;

        document.body.appendChild(modal);

        const input = modal.querySelector("#rename-input");
        input.focus();

        modal.querySelector("#rename-ok").onclick = () => {
            const value = input.value.trim();
            modal.remove();
            resolve(value);
        };

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

            await fetch("http://localhost:5000/pin", {
                method: "POST",
                body: pinForm
            });
            loadSidebarHistory();

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

    currentChatId = chatId;

    try {

        if (saveInFlight) await saveInFlight;

        const hadMessages = outputDiv.children.length > 0;

        if (hadMessages && !isHistory) {
            await saveCurrentChat();
        }

        isHistory = true;

        outputDiv.innerHTML = "";

        const response = await fetch(`http://localhost:5000/chat/${chatId}`);
        const data = await response.json();

        const interactions = data.content.split("[TURN_END]");

        interactions.forEach(line => {

            line = line.trim();
            if (!line) return;

            const parts = line.split(" | ");

            parts.forEach(part => {

                part = part.trimStart();

                if (part.startsWith("User: ")) {
                    appendMessage(outputDiv, "user", part.slice(6));
                }

                else if (part.startsWith("Assistant: ")) {
                    appendMessage(outputDiv, "assistant", part.slice(11));
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
btn.addEventListener("click", (e) => {

    e.stopPropagation();
    list.classList.toggle("open");

});

list.addEventListener("click", (e) => e.stopPropagation());

document.addEventListener("click", () => {
    list.classList.remove("open");
});


options.forEach(element => {

    element.addEventListener("click", () => {

        const modeName = element.querySelector('p').innerText;

        btn.innerHTML = "Mode: " + modeName;

        list.style.display = "none";

    });

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


function appendMessage(outputDiv, role, content, intent = "") {

    const cls = role === "user" ? "msg user" : "msg assistant";
    const meta = role === "user"
        ? "You"
        : `Assistant <span class="tag">[${intent}]</span>`;

    let bodyContent = "";

    if (role === "assistant") {

        try {
            bodyContent = window.electronAPI.parseMarkdown(content);
        } catch {
            bodyContent = escapeHtml(content).replaceAll("\n", "<br>");
        }

    } else {

        bodyContent = escapeHtml(content).replaceAll("\n", "<br>");

    }

    outputDiv.insertAdjacentHTML("beforeend", `
        <div class="${cls}">
            <div class="meta">${meta}</div>
            <div class="text">${bodyContent}</div>
        </div>
    `);

    outputDiv.scrollTo({
        top: outputDiv.scrollHeight,
        behavior: 'smooth'
    });

}


async function sendMessage() {
    const query = ain.value.trim();

    if (!query && pendingImages.length === 0) return;

    const currentMode = btn.innerText.replace("Mode: ", "").trim();

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
        <div class="meta">You</div>
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
        form.append("mode", currentMode);

        pendingImages.forEach((img) => form.append("image", img));

        // continue the same long-term chat if we loaded one
        if (currentChatId) form.append("chat_id", currentChatId);

        const response = await fetch("http://localhost:5000/ask", {
            method: "POST",
            body: form,
        });

        const data = await response.json();

        // clear image UI state
        pendingImages = [];
        attachmentTray.innerHTML = "";
        imgInput.value = "";

        // remove loading bubble
        document.getElementById(loadingId)?.remove();

        // append assistant
        appendMessage(outputDiv, "assistant", data.response, data.intent);

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
        if (loadingEl) loadingEl.innerText = "Error: Python server is offline.";
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