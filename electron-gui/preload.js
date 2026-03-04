const { contextBridge, ipcRenderer } = require('electron');
const { marked } = require('marked');


window.addEventListener('DOMContentLoaded', () => {
    const replaceText = (selector, text) => {
        const element = document.getElementById(selector)
        if (element) element.innerText = text
    }

    for (const type of ['chrome', 'node', 'electron']) {
        replaceText(`${type}-version`, process.versions[type])
    }
})

contextBridge.exposeInMainWorld('electronAPI', {
    // This function can now be called in index.js as window.electronAPI.parseMarkdown()
    parseMarkdown: (text) => {
        try {
            return marked.parse(text);
        } catch (e) {
            console.error("Markdown parsing failed:", e);
            return text; // Fallback to raw text
        }
    }
});