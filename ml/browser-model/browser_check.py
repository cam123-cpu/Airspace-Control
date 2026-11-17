"""End-to-end check inside Chrome: load infer.js + model.json into the sim page, capture a never-seen seed, classify in-page."""
import asyncio, json, sys
from pathlib import Path
from playwright.async_api import async_playwright
HERE = Path(__file__).resolve().parent
async def main(html, seed):
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        pg = await b.new_page(); await pg.route("**/fonts.*/**", lambda r: r.abort())
        await pg.goto(Path(html).resolve().as_uri()); await pg.wait_for_timeout(300)
        for f in ["infer.js", "capture.js"]:
            await pg.evaluate("(src)=>{(0,eval)(src)}", (HERE / f).read_text())
        await pg.evaluate("(m)=>pixelModel.load(m)", json.loads((HERE / "model.json").read_text()))
        r = await pg.evaluate("""(seed)=>{const res={n:0,ok:0,dr:[0,0],ms:0};for(const scen of Object.keys(SCEN)){const c=__capture({scen,seed,salt:3,shots:4,tmax:200,thermalFrac:0.35});
          const bin=atob(c.b64);c.meta.forEach((m,i)=>{const g=new Uint8Array(1024);for(let k=0;k<1024;k++)g[k]=bin.charCodeAt(i*1024+k);const t0=performance.now();const o=pixelModel.classify(g);res.ms+=performance.now()-t0;
          res.n++;if(o.cls===m.cls)res.ok++;if(m.cls<2){res.dr[1]++;if(o.cls<2)res.dr[0]++}})}res.ms/=res.n;return res}""", seed)
        print(f"in-browser, seed {seed} (unseen): accuracy {r['ok']/r['n']:.3f} on {r['n']} crops; drone recall {r['dr'][0]/max(1,r['dr'][1]):.3f}; {r['ms']:.3f} ms per classify")
        await b.close()
asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "../../apps/simulator/index.html", int(sys.argv[2]) if len(sys.argv) > 2 else 303))
