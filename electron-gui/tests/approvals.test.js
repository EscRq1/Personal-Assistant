const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function ui() {
    class Element {
        constructor(tag) { this.tag = tag; this.children = []; this.hidden = true; this.events = {}; }
        setAttribute() {}
        append(...items) { for (const child of items) { child.parent = this; this.children.push(child); } }
        remove() { this.parent.children = this.parent.children.filter(x => x !== this); }
        addEventListener(name, callback) { this.events[name] = callback; }
        querySelectorAll(tag) { return this.children.filter(x => x.tag === tag); }
    }
    const host = new Element('div');
    const calls = [];
    const scope = vm.createContext({
        document: {createElement: tag => new Element(tag), getElementById: () => host},
        setTimeout: () => 1, clearTimeout() {},
        window: {electronAPI: {approvalDecision: async (id, token, allow) => { calls.push({id, token, allow}); return {ok: true}; }}},
    });
    const source = fs.readFileSync(path.join(__dirname, '../src/javascript/index.js'), 'utf8');
    const helper = source.slice(source.indexOf('// Capabilities stay in UI memory,'));
    vm.runInContext('const API="http://127.0.0.1:5000"; let currentGenerationId=1;\n' + helper, scope);
    return {host, calls, run: source => vm.runInContext(source, scope)};
}

for (const allow of [true, false]) {
    test(`${allow ? 'Allow' : 'Decline'} submits the capability and decision once`, async () => {
        const {host, calls, run} = ui();
        run('showApprovalCard({id:"one", token:"capability", kind:"desktop", details:{reason:"Read desktop"}},1)');
        assert.equal(host.hidden, false);
        const actions = host.children[0].children.at(-1);
        await actions.children[allow ? 1 : 0].events.click();
        assert.deepEqual(calls, [{id: 'one', token: 'capability', allow}]);
        assert.equal(host.hidden, true);
        assert.equal(host.children.length, 0);
    });
}

test('package names and reasons are text, duplicate/stale requests are ignored, Stop clears cards', () => {
    const {host, run} = ui();
    run('showApprovalCard({id:"one",kind:"package", details:{packages:["pandas>=2"],reason:"<img src=x>"}},1)');
    assert.match(host.children[0].children[1].textContent, /pandas>=2/);
    assert.equal(host.children[0].children[2].textContent, '<img src=x>');
    run('showApprovalCard({id:"one",details:{}},1); showApprovalCard({id:"stale",details:{}},0)');
    assert.equal(host.children.length, 1);
    run('clearApprovalCards()');
    assert.equal(host.hidden, true);
    assert.equal(host.children.length, 0);
});

test('failed decision leaves the card actionable for retry', async () => {
    const {host, run} = ui();
    run('window.electronAPI.approvalDecision = async () => { throw new Error("offline"); }; showApprovalCard({id:"one",token:"cap",details:{}},1)');
    const actions = host.children[0].children.at(-1);
    await actions.children[1].events.click();
    assert.equal(host.children.length, 1);
    assert.equal(actions.children[1].disabled, false);
});

function nativeBridge() {
    const handlers = new Map();
    const scope = vm.createContext({
        require: name => name === 'electron' ? {
            app: {whenReady: () => ({then() {}}), on() {}},
            BrowserWindow: {}, net: {}, ipcMain: {handle: (name, callback) => handlers.set(name, callback)},
        } : require(name),
        __dirname: path.resolve(__dirname, '../src/javascript'), setTimeout, clearTimeout, console,
    });
    const main = fs.readFileSync(path.join(__dirname, '../src/javascript/main.js'), 'utf8');
    vm.runInContext(main, scope);
    vm.runInContext(`win = {webContents: {mainFrame: {url: pathToFileURL(path.join(__dirname, '../html/index.html')).href}}};
        globalThis.sent = []; postJson = async (url, body) => {sent.push({url, body}); return {statusCode:200};};
        globalThis.event = {sender:win.webContents,senderFrame:win.webContents.mainFrame};`, scope);
    return {scope, handler: handlers.get('approval-decision')};
}

test('native bridge only forwards decisions from the application main frame', async () => {
    const {scope, handler} = nativeBridge();
    const payload = {id: 'one', token: 'cap', allow: true};
    assert.equal((await handler({sender: {}, senderFrame: {}}, payload)).status, 403);
    assert.equal((await handler(scope.event, {...payload, id: '../../shutdown'})).status, 400);
    assert.equal(scope.sent.length, 0);
    assert.equal((await handler(scope.event, payload)).ok, true);
    assert.equal(scope.sent[0].url, '/approvals/one/decision');
    assert.equal(scope.sent[0].body.allow, true);
    assert.equal(scope.sent[0].body.token, 'cap');
    scope.event.senderFrame.url = 'https://example.org/';
    assert.equal((await handler(scope.event, payload)).status, 403);
});
