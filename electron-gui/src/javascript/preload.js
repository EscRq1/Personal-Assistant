const { contextBridge, ipcRenderer } = require('electron');
const { marked } = require('marked');
const katex = require('katex');


// ============================================================
// Electron startup
// ============================================================

window.addEventListener('DOMContentLoaded', () => {
    const replaceText = (selector, text) => {
        const element = document.getElementById(selector);
        if (element) {
            element.innerText = text;
        }
    };

    for (const type of ['chrome', 'node', 'electron']) {
        replaceText(`${type}-version`, process.versions[type]);
    }
});


// ============================================================
// KaTeX
// ============================================================

function renderMath(text, displayMode) {
    try {
        return katex.renderToString(text, {
            displayMode,
            throwOnError: false,
            strict: false
        });
    } catch (error) {
        console.error('[KaTeX] render failed:', error);

        // Preserve the original expression rather than destroying it.
        return displayMode
            ? `<div class="math-bubble math-error">${escapeHtml(text)}</div>`
            : `<span class="math-error">${escapeHtml(text)}</span>`;
    }
}


function escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}


// ============================================================
// Robust Markdown + Math preprocessor
//
// Supported:
//
//   $$ ... $$      display
//   \[ ... \]      display
//   $ ... $        inline
//   \( ... \)      inline
//
// Math inside:
//   fenced code
//   inline code
//
// is NOT interpreted as math.
//
// The important idea:
//
//   raw text
//      ↓
//   protect/extract math
//      ↓
//   placeholders
//      ↓
//   marked.parse()
//      ↓
//   restore KaTeX HTML
//
// Therefore Markdown and KaTeX never compete for the same syntax.
// ============================================================

const PLACEHOLDER_PREFIX = '\uE000CHATGPT_MATH_';
const PLACEHOLDER_SUFFIX = '\uE001';

function makePlaceholder(index) {
    return `${PLACEHOLDER_PREFIX}${index}${PLACEHOLDER_SUFFIX}`;
}


// ------------------------------------------------------------
// Determine whether a $ is escaped.
//
// Example:
//   \$5
//
// must NOT be interpreted as math.
// ------------------------------------------------------------

function isEscaped(text, index) {
    let backslashes = 0;

    for (let i = index - 1; i >= 0 && text[i] === '\\'; --i) {
        backslashes++;
    }

    return (backslashes % 2) === 1;
}


// ------------------------------------------------------------
// Find the next unescaped closing delimiter.
// ------------------------------------------------------------

function findUnescaped(text, start, delimiter) {
    let pos = start;

    while (pos < text.length) {
        const index = text.indexOf(delimiter, pos);

        if (index === -1) {
            return -1;
        }

        if (!isEscaped(text, index)) {
            return index;
        }

        pos = index + delimiter.length;
    }

    return -1;
}


// ------------------------------------------------------------
// Fenced code detection
//
// Supports:
//
// ```
// code
// ```
//
// ~~~
// code
// ~~~
//
// We preserve these sections completely.
// ------------------------------------------------------------

function readFence(text, index) {
    if (text[index] !== '`' && text[index] !== '~') {
        return null;
    }

    const marker = text[index];

    let length = 0;

    while (
        index + length < text.length &&
        text[index + length] === marker
    ) {
        length++;
    }

    if (length < 3) {
        return null;
    }

    const fence = marker.repeat(length);

    // Find a closing fence on a line by itself.
    const closingRegex = new RegExp(
        `\\n[ \\t]*${escapeRegex(fence)}[ \\t]*(?:\\n|$)`
    );

    const bodyStart = index + length;
    const match = closingRegex.exec(text.slice(bodyStart));

    if (!match) {
        // Unclosed fence: protect the rest of the document.
        return {
            end: text.length
        };
    }

    const closingStart = bodyStart + match.index;
    const closingEnd = bodyStart + match.index + match[0].length;

    return {
        end: closingEnd
    };
}


function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}


// ------------------------------------------------------------
// Inline code detection
//
// Supports:
//
// `code`
// ``code with ` inside``
//
// Again: absolutely no math parsing inside inline code.
// ------------------------------------------------------------

function readInlineCode(text, index) {
    if (text[index] !== '`') {
        return null;
    }

    let length = 0;

    while (
        index + length < text.length &&
        text[index + length] === '`'
    ) {
        length++;
    }

    const delimiter = '`'.repeat(length);
    const close = text.indexOf(delimiter, index + length);

    if (close === -1) {
        return {
            end: text.length
        };
    }

    return {
        end: close + length
    };
}


// ------------------------------------------------------------
// Check whether a $ can reasonably begin inline math.
//
// This prevents things such as:
//
// $5
// $100
// $HOME
//
// from being interpreted as mathematics.
//
// We intentionally require a closing $ and disallow whitespace
// immediately after the opening delimiter.
//
// Example:
//
// $x^2$          -> yes
// $ x^2 $        -> no
// $5             -> no
// ------------------------------------------------------------

function looksLikeInlineDollarMath(text, index) {
    if (text[index] !== '$') {
        return false;
    }

    if (isEscaped(text, index)) {
        return false;
    }

    // $$ is block math, not inline math.
    if (text[index + 1] === '$') {
        return false;
    }

    const next = text[index + 1];

    if (!next || /\s/.test(next)) {
        return false;
    }

    return true;
}


// ============================================================
// Extract all math spans
// ============================================================

function extractMath(text) {
    const parts = [];
    const math = [];

    let buffer = '';
    let i = 0;


    function flushBuffer() {
        if (buffer) {
            parts.push({
                type: 'text',
                value: buffer
            });

            buffer = '';
        }
    }


    function addMath(expression, displayMode) {
        flushBuffer();

        const index = math.length;

        math.push({
            displayMode,
            expression
        });

        parts.push({
            type: 'placeholder',
            value: makePlaceholder(index)
        });
    }


    while (i < text.length) {

        // ----------------------------------------------------
        // Fenced code
        // ----------------------------------------------------

        const fence = readFence(text, i);

        if (fence) {
            buffer += text.slice(i, fence.end);
            i = fence.end;
            continue;
        }


        // ----------------------------------------------------
        // Inline code
        // ----------------------------------------------------

        const inlineCode = readInlineCode(text, i);

        if (inlineCode) {
            buffer += text.slice(i, inlineCode.end);
            i = inlineCode.end;
            continue;
        }


        // ----------------------------------------------------
        // Display math: $$
        // ----------------------------------------------------

        if (
            text[i] === '$' &&
            text[i + 1] === '$' &&
            !isEscaped(text, i)
        ) {
            const end = findUnescaped(
                text,
                i + 2,
                '$$'
            );

            if (end !== -1) {
                const expression = text
                    .slice(i + 2, end)
                    .trim();

                addMath(
                    expression,
                    true
                );

                i = end + 2;
                continue;
            }
        }


        // ----------------------------------------------------
        // Display math: \[ ... \]
        // ----------------------------------------------------

        if (
            text[i] === '\\' &&
            text[i + 1] === '['
        ) {
            const end = findUnescaped(
                text,
                i + 2,
                '\\]'
            );

            if (end !== -1) {
                const expression = text
                    .slice(i + 2, end)
                    .trim();

                addMath(
                    expression,
                    true
                );

                i = end + 2;
                continue;
            }
        }


        // ----------------------------------------------------
        // Inline math: \( ... \)
        // ----------------------------------------------------

        if (
            text[i] === '\\' &&
            text[i + 1] === '('
        ) {
            const end = findUnescaped(
                text,
                i + 2,
                '\\)'
            );

            if (end !== -1) {
                const expression = text
                    .slice(i + 2, end)
                    .trim();

                addMath(
                    expression,
                    false
                );

                i = end + 2;
                continue;
            }
        }


        // ----------------------------------------------------
        // Inline math: $ ... $
        // ----------------------------------------------------

        if (looksLikeInlineDollarMath(text, i)) {
            const end = findUnescaped(
                text,
                i + 1,
                '$'
            );

            if (
                end !== -1 &&
                end > i + 1
            ) {
                const expression = text
                    .slice(i + 1, end)
                    .trim();

                // Avoid treating "$foo bar$" as currency or
                // arbitrary text unless it looks mathematical.
                //
                // We deliberately allow ordinary alphabetic
                // expressions because things like:
                //
                // $x$
                // $QK^T$
                // $N \times N$
                //
                // are common.
                addMath(
                    expression,
                    false
                );

                i = end + 1;
                continue;
            }
        }


        // ----------------------------------------------------
        // Ordinary character
        // ----------------------------------------------------

        buffer += text[i];
        i++;
    }

    flushBuffer();

    return {
        parts,
        math
    };
}


// ============================================================
// Replace placeholders after Markdown parsing
// ============================================================

function restoreMath(html, math) {
    let output = html;

    for (let i = 0; i < math.length; i++) {
        const item = math[i];

        const rendered = renderMath(
            item.expression,
            item.displayMode
        );

        const placeholder = makePlaceholder(i);

        // A display equation that occupies an entire Markdown
        // paragraph would otherwise become:
        //
        // <p>...</p>
        //
        // containing a <div>, which is invalid HTML.
        //
        // Handle that exact case specially.
        const fullParagraph = new RegExp(
            `<p>\\s*${escapeRegex(placeholder)}\\s*</p>`
        );

        if (item.displayMode) {
            output = output.replace(
                fullParagraph,
                `<div class="math-bubble">${rendered}</div>`
            );
        }

        // Normal inline replacement.
        output = output.replaceAll(
            placeholder,
            item.displayMode
                ? `<div class="math-bubble">${rendered}</div>`
                : `<span class="math-inline">${rendered}</span>`
        );
    }

    return output;
}


// ============================================================
// Main Markdown parser
// ============================================================

function parseMarkdown(text) {
    try {
        const source = String(text ?? '');

        // ----------------------------------------------------
        // 1. Extract math before Markdown sees it.
        // ----------------------------------------------------

        const extracted = extractMath(source);

        const protectedMarkdown = extracted.parts
            .map(part => part.value)
            .join('');

        // ----------------------------------------------------
        // 2. Let Marked deal ONLY with Markdown.
        // ----------------------------------------------------

        const renderedMarkdown = marked.parse(
            protectedMarkdown
        );

        // ----------------------------------------------------
        // 3. Put KaTeX HTML back in.
        // ----------------------------------------------------

        return restoreMath(
            renderedMarkdown,
            extracted.math
        );

    } catch (error) {
        console.error(
            '[Markdown] parsing failed:',
            error
        );

        return escapeHtml(text);
    }
}


// ============================================================
// Marked configuration
// ============================================================
//
// We intentionally DON'T add a custom math tokenizer.
//
// Math is extracted before Marked sees the text.
// This prevents:
//
//   $$
//   $
//   \(
//   \[
//
// from fighting with paragraph/code tokenization.
// ============================================================

marked.setOptions({
    gfm: true,
    breaks: true
});


// ============================================================
// Electron API
// ============================================================

contextBridge.exposeInMainWorld('electronAPI', {

    parseMarkdown: (text) => {
        return parseMarkdown(text);
    }

});