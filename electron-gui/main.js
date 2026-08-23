const { app, BrowserWindow, net } = require('electron');
const path = require('path');

let win = null;
let shutdownStarted = false;
let allowQuit = false;

function createWindow() {
    win = new BrowserWindow({
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: false,
        },
    });

    win.loadFile(path.join(__dirname, 'index.html'));

    win.on('closed', () => {
        win = null;
    });
}

function postJson(pathname, payload = {}) {
    return new Promise((resolve, reject) => {
        const request = net.request({
            method: 'POST',
            protocol: 'http:',
            hostname: '127.0.0.1',
            port: 5000,
            path: pathname,
        });

        request.setHeader('Content-Type', 'application/json');

        let finished = false;
        const timeout = setTimeout(() => {
            if (finished) return;
            finished = true;
            request.abort();
            reject(new Error(`Python request timed out: ${pathname}`));
        }, 300000);

        const finish = (callback, value) => {
            if (finished) return;
            finished = true;
            clearTimeout(timeout);
            callback(value);
        };

        request.on('response', (response) => {
            let body = '';

            response.on('data', (chunk) => {
                body += chunk.toString();
            });

            response.on('end', () => {
                const result = {
                    statusCode: response.statusCode,
                    body,
                };

                if (response.statusCode >= 200 && response.statusCode < 300) {
                    finish(resolve, result);
                } else {
                    finish(
                        reject,
                        new Error(
                            `Python returned ${response.statusCode} for ${pathname}: ${body}`
                        )
                    );
                }
            });

            response.on('error', (error) => {
                finish(reject, error);
            });
        });

        request.on('error', (error) => {
            finish(reject, error);
        });

        request.write(JSON.stringify(payload));
        request.end();
    });
}

async function finalizeAndQuit() {
    if (shutdownStarted) return;
    shutdownStarted = true;

    try {
        // One Python request performs save -> final summary/profile update -> prune.
        const result = await postJson('/shutdown');
        console.log(`Shutdown status: ${result.statusCode}`);
    } catch (error) {
        // Do not trap the Electron process forever if Python is unavailable.
        console.error(`Python shutdown failed: ${error.message}`);
    } finally {
        allowQuit = true;
        app.quit();
    }
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

// Catch every normal quit path, not only closing the last window.
app.on('before-quit', (event) => {
    if (allowQuit) return;

    event.preventDefault();
    void finalizeAndQuit();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});