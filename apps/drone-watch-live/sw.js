// Offline cache for the app shell. Video never leaves the device; this only caches the app's own files.
const CACHE = "drone-watch-v1";
const SHELL = ["./", "./index.html", "./manifest.webmanifest", "./icon.svg"];
self.addEventListener("install", e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())));
self.addEventListener("activate", e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener("fetch", e => {
  const u = new URL(e.request.url);
  if (u.origin !== location.origin || e.request.method !== "GET") return; // CDN (ONNX Runtime) goes to network
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
});
