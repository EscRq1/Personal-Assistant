const { app, BrowserWindow, ipcMain, net } = require('electron');
const path = require('path');

let win;

function createWindow () {
    const win = new BrowserWindow({
        // ... width/height
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: false // <--- Add this line!
        }
    });

    win.loadFile('index.html')
}

app.whenReady().then(() => {
    createWindow()

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow()
        }
    })
})

app.on('window-all-closed', async () => {
    // 1. Tell Python to prune before we die
    try {
        const save = net.request({
            method: 'POST',
            protocol: 'http:',
            hostname: '127.0.0.1',
            port: 5000,
            path: '/save',
        })

        save.on('response', (response) => {
            console.log(`Save status: ${response.statusCode}`);
            proceedToQuit();
        });
        
        save.on('error', (error) => {
            console.error(`Save failed: ${error.message}`);
            proceedToQuit();
        });

        save.write(JSON.stringify({}));
        save.end();


        const request = net.request({
            method: 'POST',
            protocol: 'http:',
            hostname: '127.0.0.1',
            port: 5000,
            path: '/prune',
        });
        
        request.on('response', (response) => {
            console.log(`Prune status: ${response.statusCode}`);
            proceedToQuit();
        });
        
        request.on('error', (error) => {
            console.error(`Prune failed: ${error.message}`);
            proceedToQuit();
        });

        request.write(JSON.stringify({}));
        request.end();
        

    } catch (e) {
        proceedToQuit();
    }
});

function proceedToQuit() {
    if (process.platform !== 'darwin') {
        app.quit();
    }
}

