// node pixelmodel.test.js — the shipped int8 net must reject bad crops and stay finite on good ones.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const pixelModel = require("./pixelmodel.js");

let pass = 0, fail = 0;
function ok(cond, msg) {
  if (cond) pass++;
  else { fail++; console.log("FAIL:", msg); }
}

const weights = fs.readFileSync(path.join(__dirname, "pixelmodel_weights.js"), "utf8");
const ctx = { window: {} };
vm.runInNewContext(weights, ctx);
pixelModel.load(ctx.window.PIXEL_MODEL_JSON);
ok(pixelModel.model.features === "hog", "shipped model is the hog net");
ok(pixelModel.model.layers[0].nin === 324, "first layer sees 324 hog features");

const flat = new Uint8Array(1024).fill(128);
const r = pixelModel.classify(flat);
ok(pixelModel.CLASSES.includes(r.label), "label is a known class: " + r.label);
ok(r.probs.length === 4, "four class probabilities");
const sum = r.probs.reduce((a, b) => a + b, 0);
ok(Math.abs(sum - 1) < 1e-9, "probabilities sum to 1, got " + sum);
ok(r.probs.every(Number.isFinite), "probabilities are finite");

let threw = false;
try { pixelModel.classify(new Uint8Array(10)); } catch (e) { threw = /crop length/.test(e.message); }
ok(threw, "short crop throws");

const dirty = new Uint8Array(1024).fill(40);
threw = false;
try { pixelModel.classify(Object.assign(Array.from(dirty), { 3: NaN })); } catch (e) { threw = /non-finite pixel/.test(e.message); }
ok(threw, "NaN pixel throws");

const again = pixelModel.classify(flat);
ok(again.probs.every(Number.isFinite) && Math.abs(again.probs.reduce((a, b) => a + b, 0) - 1) < 1e-9, "a bad crop does not poison the next call");

// A one-pixel-wide edge should not explode the hog path.
const edge = new Uint8Array(1024);
for (let i = 0; i < 32; i++) edge[i * 32 + 16] = 255;
const e2 = pixelModel.classify(edge);
ok(e2.probs.every(p => p >= 0 && p <= 1), "edge crop stays inside [0,1]");

console.log(pass + " passed, " + fail + " failed");
process.exit(fail ? 1 : 0);
