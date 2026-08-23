const { contextBridge, ipcRenderer } = require('electron');
const { marked } = require('marked');
const katex = require('katex');


window.addEventListener('DOMContentLoaded', () => {
    const replaceText = (selector, text) => {
        const element = document.getElementById(selector)
        if (element) element.innerText = text
    }

    for (const type of ['chrome', 'node', 'electron']) {
        replaceText(`${type}-version`, process.versions[type])
    }
})

const mathRenderer = {
    math_block(token) {
        const renderedMath = katex.renderToString(token.text, { 
            displayMode: true, throwOnError: false 
        });
        return `<div class="math-bubble">${renderedMath}</div>`;
    },
    math_inline(token) {
        const renderedMath = katex.renderToString(token.text, { 
            displayMode: false, throwOnError: false 
        });
        return `<span>${renderedMath}</span>`;
    }
};

marked.use({
    extensions: [
        {
            name: 'math-block',
            level: 'block',
            start(src) { 
                // FIX: Allow the block to start anywhere, not just index 0
                return src.indexOf('$$'); 
            }, 
            tokenizer(src) {
                // We match from the first $$ to the next $$
                // We use a regex that allows leading whitespace/newlines
                const match = src.match(/^\s*\$\$([\s\S]+?)\$\$/);
                if (match) {
                    return {
                        type: 'math-block',
                        raw: match[0],
                        text: match[1].trim(),
                    };
                }
            },
            renderer(token) { return mathRenderer.math_block(token); }
        },
        {
            name: 'math-inline',
            level: 'inline', 
            start(src) { 
                const index = src.indexOf('$');
                if (index === -1) return -1;
                
                // SAFETY GUARD: If this $ is actually the start of a $$, 
                // we MUST let the block tokenizer handle it.
                if (src.indexOf('$$', index) === index) {
                    return -1; 
                }
                return index; 
            }, 
            tokenizer(src) {
                const match = src.match(/^\$([^$]+?)\$/); 
                if (match) {
                    return {
                        type: 'math-inline',
                        raw: match[0],
                        text: match[1].trim(),
                    };
                }
            },
            renderer(token) { return mathRenderer.math_inline(token); }
        }
    ]
});

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