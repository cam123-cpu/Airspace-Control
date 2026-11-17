/* Operator sensor views for Prison Airspace Sim.
   Global VIEWS: radarPPI · rangeDoppler · rfWaterfall · acousticSpectrogram, plus VIEWS.mount(container) and VIEWS.frame(dt).
   Views READ the sim state (W, UI, RADAR, RFS, ACS, ...) and never mutate it.
   Signal synthesis is behind VIEWS.setSynth({...}) so a fuller model (sensors2.js) can replace the built-in one. */
(function(){
"use strict";
const DEG=Math.PI/180;
const G=()=>window; // sim globals are top-level lexical bindings; read them by name below
function S(){return{W:(typeof W!=="undefined"?W:null),UI:(typeof UI!=="undefined"?UI:{})}}

// ---------- palette / fonts (cached once; css() is slow) ----------
let PAL=null;
function pal(){if(PAL)return PAL;const g=n=>{try{return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}catch(e){return""}};
  PAL={accent:g("--accent")||"#4cc2d8",warn:g("--warn")||"#f0a93b",crit:g("--crit")||"#ff5b52",ok:g("--ok")||"#5ac88b",rf:g("--rf")||"#c58cff",muted:g("--muted")||"#7d91a3",ink:g("--ink")||"#d5e1ea",line:g("--line")||"#21303d",
    mono:(g("--mono")||"ui-monospace,monospace"),bg:"#070b0f"};return PAL}
const F=(px,w)=>`${w||400} ${px}px ${pal().mono}`;

// ---------- perceptual colormaps as 256-entry LUTs ----------
function lut(stops){const o=new Uint8ClampedArray(256*3);for(let i=0;i<256;i++){const x=i/255*(stops.length-1),k=Math.min(stops.length-2,Math.floor(x)),f=x-k;for(let c=0;c<3;c++)o[i*3+c]=stops[k][c]+(stops[k+1][c]-stops[k][c])*f}return o}
const INFERNO=lut([[0,0,4],[31,12,72],[85,15,109],[136,34,106],[186,54,85],[227,89,51],[249,140,10],[249,201,50],[252,255,164]]);
const VIRIDIS=lut([[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]]);
const PHOS=lut([[0,0,0],[0,40,18],[10,110,50],[60,200,110],[190,255,210]]);
function paint(img,vals,map,lo,hi){const d=img.data,n=vals.length,k=255/(hi-lo);for(let i=0;i<n;i++){let v=(vals[i]-lo)*k;v=v<0?0:v>255?255:v|0;d[i*4]=map[v*3];d[i*4+1]=map[v*3+1];d[i*4+2]=map[v*3+2];d[i*4+3]=255}}

// ---------- small RNG (views own their randomness: never touch the sim's seeded RNG) ----------
let seed=12345;const rnd=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/4294967296};
const gn=()=>{let u=1-rnd(),v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)};
const expn=()=>-Math.log(1-rnd()+1e-12);

// ---------- helpers on sim state ----------
const p3=s=>[s.x,s.y,s.z];
const hyp=(a,b)=>Math.sqrt(a*a+b*b);
const az=(s,p)=>((Math.atan2(p[0]-s[0],p[1]-s[1])/DEG)+360)%360;
const d3=(a,b)=>Math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2);
function selTrack(){const{W}=S();if(!W)return null;const T=W.tracks.filter(t=>t.confirmed&&!t.hidden);
  if(VIEWS.selectedTrack){const t=T.find(x=>x.id===VIEWS.selectedTrack);if(t)return t}
  return T.find(t=>t.id===W.cam.cue)||T.sort((a,b)=>b.p_-a.p_)[0]||null}
function trackColor(t){try{return threatColor(t)}catch(e){return pal().muted}}
function offline(ctx,w,h,name){ctx.fillStyle="rgba(5,8,11,.72)";ctx.fillRect(0,0,w,h);ctx.fillStyle=pal().crit;ctx.font=F(16,700);ctx.textAlign="center";ctx.fillText(`${name} OFFLINE`,w/2,h/2);ctx.font=F(11);ctx.fillStyle=pal().muted;ctx.fillText("Coverage lost · fusion is running on the remaining sensors",w/2,h/2+18);ctx.textAlign="left"}
function chip(ctx,x,y,txt,col,align){ctx.font=F(10.5,600);const tw=ctx.measureText(txt).width+10;const X=align==="r"?x-tw:x;ctx.fillStyle="rgba(7,11,15,.82)";ctx.fillRect(X,y-12,tw,16);ctx.fillStyle=col;ctx.fillRect(X,y-12,2,16);ctx.fillText(txt,X+6,y);return tw}
function header(ctx,w,title,sub){ctx.fillStyle="rgba(7,11,15,.85)";ctx.fillRect(0,0,w,20);ctx.font=F(11,700);ctx.fillStyle=pal().ink;ctx.fillText(title,8,14);if(sub){ctx.font=F(10.5);ctx.fillStyle=pal().muted;const tw=ctx.measureText(sub).width;ctx.fillText(sub,w-tw-8,14)}}

// ======================================================================
// Built-in synthesizers (replaceable via VIEWS.setSynth)
// ======================================================================
const SYN={
  // Radar return power in dB for an entity (radar equation, normalised so a 0.01 m² drone at 1 km ≈ 30 dB over noise)
  radarPower(e,r){const rcs=e.rcs||0.001;return 30+10*Math.log10(rcs/0.01)-40*Math.log10(Math.max(r,30)/1000)},
  // Doppler spread for one entity in one coherent dwell: returns list of [vr, dB] components
  dopplerComponents(e,vr,P,t){
    const out=[[vr,P]];
    if(e.type==="drone"&&e.phase!=="wait"&&e.phase!=="done"){ // rotor micro-Doppler: blade flashes spread up to tip speed
      const tip=38;for(let k=0;k<10;k++){const v=vr+(rnd()*2-1)*tip;out.push([v,P-14-rnd()*10])}
      for(const h of[-1,1])out.push([vr+h*tip*0.92,P-9]);}
    else if(e.type==="bird"){const A=Math.PI*(e.flapHz||4)*(e.span||1)*0.5;const s=Math.sin(e.flap||0);out.push([vr+A*s,P-7]);out.push([vr-A*0.5*s,P-11])}
    else if(e.type==="car"){out.push([vr+(rnd()-.5)*2,P-6])}
    return out},
  // One micro-Doppler spectrogram column (Float32 dB, ny bins over [-vmax,vmax]) for a truth entity
  microDopplerColumn(col,ny,vmax,e,vr,t){
    for(let i=0;i<ny;i++)col[i]=-6+expn()*3; // noise floor
    if(!e)return;
    const bin=v=>Math.round((v+vmax)/(2*vmax)*(ny-1));
    const put=(v,db,w)=>{const b=bin(v);for(let j=-w;j<=w;j++){const k=b+j;if(k>=0&&k<ny)col[k]=Math.max(col[k],db-Math.abs(j)*4)}};
    put(vr,30,1);
    if(e.type==="drone"){ // blade flashes: brief full-spread vertical lines; rotor sidebands
      const flash=rnd()<0.45;const tip=34;
      for(let i=0;i<ny;i++){const v=-vmax+i*2*vmax/(ny-1);if(Math.abs(v-vr)<=tip)col[i]=Math.max(col[i],(flash?17:9)+gn()*2.5-Math.abs(v-vr)/tip*6)}
      put(vr+tip*0.95,16,0);put(vr-tip*0.95,16,0);put(vr+tip*0.5,12,0);put(vr-tip*0.5,12,0);
    }else if(e.type==="bird"){const A=Math.PI*(e.flapHz||4)*(e.span||1)*0.5;const ph=2*Math.PI*(e.flapHz||4)*t+(e.ph||0);put(vr+A*Math.sin(ph),22,1);put(vr+A*0.45*Math.sin(ph+0.6),15,0)}
    else if(e.type==="car"){put(vr,26,2)}
    else if(e.type==="payload"||e.type==="bag"){put(vr,16,0)}
  },
  // One RF waterfall row (dBm-ish) for a band. ctx gives emitters and helpers.
  rfRow(row,band,t,ctxi){
    const n=row.length,df=(band.f1-band.f0)/n;
    for(let i=0;i<n;i++)row[i]=-98+expn()*2.2;
    const blob=(fc,bw,db,soft)=>{const a=Math.floor((fc-bw/2-band.f0)/df),b=Math.ceil((fc+bw/2-band.f0)/df);for(let i=Math.max(0,a-2);i<Math.min(n,b+2);i++){const edge=(i<a||i>b)?(soft?8:14):0;row[i]=Math.max(row[i],db-edge+gn()*1.4)}};
    for(const wc of band.wifi){if(rnd()<wc.duty)blob(wc.f,wc.bw,wc.db+gn()*2,true)}
    for(const e of ctxi.emitters){
      const lv=e.level;
      if(e.kind==="link"||e.kind==="decoy"){
        if(band.id==="24"){const nb=2;for(let k=0;k<nb;k++){if(rnd()<0.7){const f=band.f0+2+rnd()*(band.f1-band.f0-4);blob(f,1.4,lv,false);e.lastHop=f}}}
        if(band.id==="58"&&e.kind==="link"&&e.airborne){blob(e.videoF,e.videoBW,lv-6+gn(),true)}
      }
      if(e.rid&&band.id==="24"&&((t*10)|0)%10===(e.ridSlot|0)){for(const f of[2402,2426,2480])blob(f,2,lv-4,false);blob(2437,20,lv-10,true)}
    }
  },
  // One acoustic spectrogram column over log-spaced bins (Hz list)
  acousticColumn(col,freqs,node,t,ctxi){
    const UIx=ctxi.UI||{};const wind=UIx.wind||0,rain=UIx.wx==="rain";
    for(let i=0;i<freqs.length;i++){const f=freqs[i];let db=8-6*Math.log10(f/50)+expn()*2.5; // pink-ish floor
      db=Math.max(db,wind*2.6-18*Math.log10(f/50)+gn()*2.5); // wind: low-frequency rumble rising with wind speed
      if(rain)db=Math.max(db,12+gn()*2);col[i]=db}
    const at=(f,db,w)=>{let lo=0,hi=freqs.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(freqs[m]<f)lo=m;else hi=m}const k=(f-freqs[lo]<freqs[hi]-f)?lo:hi;for(let j=-w;j<=w;j++){const q=k+j;if(q>=0&&q<freqs.length)col[q]=Math.max(col[q],db-Math.abs(j)*5+gn())}};
    for(const s of ctxi.sources){
      if(s.kind==="drone"){const bpf=s.bpf;for(let h=1;h<=12;h++){const f=bpf*h;if(f>freqs[freqs.length-1])break;const db=s.level-4*Math.log2(h)-f/900;for(let r=0;r<4;r++)at(f*(1+0.012*(r-1.5)),db-2*r,0)}}
      else if(s.kind==="truck"){for(let f=50;f<220;f*=1.06)at(f,s.level-(f-50)/12,1)}
      else if(s.kind==="aircraft"){for(let f=50;f<900;f*=1.08)at(f,s.level-(f/120),1)}
    }
  }
};
VIEWS_SYNTH_DEFAULT=SYN;

// ======================================================================
// 1. Radar PPI with phosphor persistence
// ======================================================================
const radarPPI={acc:null,last:null,sweepPeriod:2,face:null,faceKey:"",
  buildFace(ctx,w,h,cx,cy,R,sc,toS){const P=pal();ctx.fillStyle=P.bg;ctx.fillRect(0,0,w,h);
    const g=ctx.createRadialGradient(cx,cy,0,cx,cy,R);g.addColorStop(0,"#06140d");g.addColorStop(1,"#030806");ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.fill();
    ctx.strokeStyle="rgba(90,200,139,.22)";ctx.lineWidth=1;ctx.font=F(9.5);ctx.fillStyle="rgba(140,220,170,.6)";
    for(const km of[0.5,1,1.5,2]){const r=km*1000*sc;if(r>R+1)continue;ctx.beginPath();ctx.arc(cx,cy,r,0,7);ctx.stroke();ctx.fillText(`${km} km`,cx+3,cy-r+11)}
    for(let d=0;d<360;d+=10){const L=d%30===0?9:4,an=d*DEG;ctx.beginPath();ctx.moveTo(cx+Math.sin(an)*R,cy-Math.cos(an)*R);ctx.lineTo(cx+Math.sin(an)*(R-L),cy-Math.cos(an)*(R-L));ctx.stroke();
      if(d%30===0&&d!==0){const tx=cx+Math.sin(an)*(R+11),ty=cy-Math.cos(an)*(R+11)+3;ctx.textAlign="center";ctx.fillText(String(d).padStart(3,"0"),tx,ty);ctx.textAlign="left"}}
    ctx.textAlign="center";ctx.font=F(10,700);ctx.fillText("N",cx,cy-R-4);ctx.textAlign="left";
    ctx.strokeStyle="rgba(90,200,139,.12)";for(let d=0;d<360;d+=30){const an=d*DEG;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.sin(an)*R,cy-Math.cos(an)*R);ctx.stroke()}
    if(typeof FENCE!=="undefined"){const p0=toS(FENCE.x0,FENCE.y1),p1=toS(FENCE.x1,FENCE.y0);ctx.strokeStyle="rgba(213,225,234,.7)";ctx.lineWidth=1.2;ctx.strokeRect(p0[0],p0[1],p1[0]-p0[0],p1[1]-p0[1])}
    if(typeof ROADS!=="undefined"){ctx.save();ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.clip();ctx.strokeStyle="rgba(125,145,163,.25)";ctx.lineWidth=1;for(const r of ROADS){const p=toS(...r.pts[0]),q=toS(...r.pts[1]);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke()}ctx.restore()}},
  draw(ctx,w,h,dt){
    const{W,UI:U}=S();const P=pal();
    ctx.fillStyle=P.bg;ctx.fillRect(0,0,w,h);
    const R=Math.min(w,h)/2-30,cx=w/2,cy=h/2+12,range=(typeof RADAR!=="undefined"?RADAR.range:2200),sc=R/range,rx=RADAR.x,ry=RADAR.y;
    const toS=(x,y)=>[cx+(x-rx)*sc,cy-(y-ry)*sc];
    if(!this.acc||this.acc.width!==w||this.acc.height!==h){this.acc=document.createElement("canvas");this.acc.width=w;this.acc.height=h;this.last=null}
    const a=this.acc.getContext("2d");
    // static scope face is cached and only rebuilt on resize
    const key=w+"x"+h;if(this.faceKey!==key){this.faceKey=key;this.face=document.createElement("canvas");this.face.width=w;this.face.height=h;this.buildFace(this.face.getContext("2d"),w,h,cx,cy,R,sc,toS)}
    ctx.drawImage(this.face,0,0);
    if(!this.spr){this.spr=document.createElement("canvas");this.spr.width=this.spr.height=12;const sx=this.spr.getContext("2d"),gr=sx.createRadialGradient(6,6,0,6,6,6);gr.addColorStop(0,"rgba(190,255,210,.95)");gr.addColorStop(1,"rgba(40,200,100,0)");sx.fillStyle=gr;sx.fillRect(0,0,12,12)}
    if(false){
    const g=ctx.createRadialGradient(cx,cy,0,cx,cy,R);g.addColorStop(0,"#06140d");g.addColorStop(1,"#030806");ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.fill();
    ctx.strokeStyle="rgba(90,200,139,.22)";ctx.lineWidth=1;ctx.font=F(9.5);ctx.fillStyle="rgba(140,220,170,.6)";
    for(const km of[0.5,1,1.5,2]){const r=km*1000*sc;if(r>R+1)continue;ctx.beginPath();ctx.arc(cx,cy,r,0,7);ctx.stroke();ctx.fillText(`${km} km`,cx+3,cy-r+11)}
    for(let d=0;d<360;d+=10){const L=d%30===0?9:4,an=d*DEG;ctx.beginPath();ctx.moveTo(cx+Math.sin(an)*R,cy-Math.cos(an)*R);ctx.lineTo(cx+Math.sin(an)*(R-L),cy-Math.cos(an)*(R-L));ctx.stroke();
      if(d%30===0){const tx=cx+Math.sin(an)*(R+11),ty=cy-Math.cos(an)*(R+11)+3;ctx.textAlign="center";ctx.fillText(d===0?"N":String(d).padStart(3,"0"),tx,ty);ctx.textAlign="left"}}
    ctx.strokeStyle="rgba(90,200,139,.12)";for(let d=0;d<360;d+=30){const an=d*DEG;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.sin(an)*R,cy-Math.cos(an)*R);ctx.stroke()}
    // facility outline + roads
    if(typeof FENCE!=="undefined"){const p0=toS(FENCE.x0,FENCE.y1),p1=toS(FENCE.x1,FENCE.y0);ctx.strokeStyle="rgba(213,225,234,.7)";ctx.lineWidth=1.2;ctx.strokeRect(p0[0],p0[1],p1[0]-p0[0],p1[1]-p0[1])}
    if(typeof ROADS!=="undefined"){ctx.save();ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.clip();ctx.strokeStyle="rgba(125,145,163,.25)";ctx.lineWidth=1;for(const r of ROADS){const p=toS(...r.pts[0]),q=toS(...r.pts[1]);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke()}ctx.restore()}
    }
    if(!W){return}
    // phosphor: decay accumulated glow, paint swept wedge + plots under it
    const t=W.t,sw=((t/this.sweepPeriod)%1)*360,prev=this.last==null?sw-6:this.last;this.last=sw;
    a.globalCompositeOperation="destination-out";a.fillStyle=`rgba(0,0,0,${Math.min(0.5,1-Math.exp(-Math.max(dt,0.001)/1.6))})`;a.fillRect(0,0,w,h);a.globalCompositeOperation="lighter";
    let span=((sw-prev)+360)%360;if(span>120)span=6;
    if(U.radar!==false){
      a.save();a.beginPath();a.arc(cx,cy,R,0,7);a.clip();
      const a0=(prev-90)*DEG,a1=(prev+span-90)*DEG;a.fillStyle="rgba(30,140,70,.10)";a.beginPath();a.moveTo(cx,cy);a.arc(cx,cy,R,a0,a1);a.closePath();a.fill();
      for(const pl of W.radarPlots){const b=az([rx,ry],pl.p),inW=((b-prev+360)%360)<=span+0.5;if(!inW)continue;const q=toS(pl.p[0],pl.p[1]);a.drawImage(this.spr,q[0]-6,q[1]-6)}
      a.restore();
      ctx.globalCompositeOperation="lighter";ctx.drawImage(this.acc,0,0);ctx.globalCompositeOperation="source-over";
      // sweep line with trailing gradient
      ctx.save();ctx.beginPath();ctx.arc(cx,cy,R,0,7);ctx.clip();
      for(let k=0;k<7;k++){const an=(sw-k*3)*DEG;ctx.strokeStyle=`rgba(120,255,170,${0.55*(1-k/7)})`;ctx.lineWidth=k?2:1.6;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(cx+Math.sin(an)*R,cy-Math.cos(an)*R);ctx.stroke()}
      ctx.restore();
    }
    // fused tracks (2525-style frames)
    const sel=selTrack();
    for(const tr of W.tracks){if(!tr.confirmed||tr.hidden)continue;const q=toS(tr.p[0],tr.p[1]);if(Math.hypot(q[0]-cx,q[1]-cy)>R)continue;
      const col=trackColor(tr),hostile=tr.threat==="ALERT"||tr.cls==="Drone",bird=tr.cls==="Bird";ctx.strokeStyle=col;ctx.lineWidth=hostile?2:1.4;ctx.beginPath();
      if(hostile){ctx.moveTo(q[0],q[1]-7);ctx.lineTo(q[0]+7,q[1]);ctx.lineTo(q[0],q[1]+7);ctx.lineTo(q[0]-7,q[1]);ctx.closePath()}
      else if(bird){ctx.arc(q[0],q[1],3,0,7)}else{ctx.rect(q[0]-5,q[1]-5,10,10)}ctx.stroke();
      const v=toS(tr.p[0]+tr.v[0]*30,tr.p[1]+tr.v[1]*30);ctx.beginPath();ctx.moveTo(q[0],q[1]);ctx.lineTo(v[0],v[1]);ctx.stroke();
      if(tr===sel||tr.threat!=="TRACK"||tr.p_>0.3){ctx.font=F(9.5,600);ctx.fillStyle=col;ctx.fillText(tr.id,q[0]+8,q[1]-6)}
      if(tr===sel){ctx.strokeStyle="rgba(255,255,255,.8)";ctx.lineWidth=1;ctx.setLineDash([2,2]);ctx.strokeRect(q[0]-11,q[1]-11,22,22);ctx.setLineDash([])}}
    // pilot estimates (bearing fix)
    for(const em of Object.values(W.emitters||{})){if(!em.est)continue;const q=toS(em.est.x,em.est.y);if(Math.hypot(q[0]-cx,q[1]-cy)>R)continue;ctx.strokeStyle=em.spoof?P.warn:P.rf;ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(q[0]-5,q[1]-5);ctx.lineTo(q[0]+5,q[1]+5);ctx.moveTo(q[0]-5,q[1]+5);ctx.lineTo(q[0]+5,q[1]-5);ctx.stroke()}
    header(ctx,w,"RADAR PPI · X-band 3D · north-up",`sweep ${sw.toFixed(0).padStart(3,"0")}° · ${W.radarPlots.filter(p=>t-p.t<0.6).length} plots/scan`);
    if(sel){const rng=hyp(sel.p[0]-rx,sel.p[1]-ry),brg=az([rx,ry],sel.p),spd=hyp(sel.v[0],sel.v[1]);
      chip(ctx,8,h-30,`${sel.id} ${sel.cls}  RNG ${(rng/1000).toFixed(2)} km  BRG ${brg.toFixed(0).padStart(3,"0")}°`,trackColor(sel));chip(ctx,8,h-11,`ALT ${Math.round(sel.p[2])} m  SPD ${spd.toFixed(1)} m/s  P(drone) ${Math.round(sel.p_*100)}%`,trackColor(sel))}
    if(w>=520){ctx.font=F(9.5);ctx.fillStyle=P.muted;ctx.textAlign="right";ctx.fillText("◆ hostile  □ unknown  ○ bird  ✕ RF fix",w-8,h-8);ctx.textAlign="left"}
    if(U.radar===false)offline(ctx,w,h,"RADAR");
  }};

// ======================================================================
// 2. Range-Doppler map + micro-Doppler spectrogram
// ======================================================================
const rangeDoppler={nx:176,ny:96,vmax:30,grid:null,img:null,small:null,tAcc:0,
  md:{buf:null,ny:96,vmax:40,acc:0,trackId:null,col:null},
  update(dt){
    const{W,UI:U}=S();if(!W)return;
    // micro-Doppler columns at ~40 col/s regardless of active tab
    const m=this.md,tr=selTrack();
    if(!m.col)m.col=new Float32Array(m.ny);
    if(tr&&tr.id!==m.trackId){m.trackId=tr.id;if(m.buf){const c=m.buf.getContext("2d");c.fillStyle="#000004";c.fillRect(0,0,m.buf.width,m.buf.height)}}
    m.acc+=dt*40;const n=Math.min(12,Math.floor(m.acc));m.acc-=n;
    if(!m.buf){m.buf=document.createElement("canvas");m.buf.width=240;m.buf.height=m.ny;const c=m.buf.getContext("2d");c.fillStyle="#000004";c.fillRect(0,0,240,m.ny)}
    if(n>0){const c=m.buf.getContext("2d");c.drawImage(m.buf,-n,0);const img=c.createImageData(n,m.ny);
      const e=tr&&U.radar!==false?tr.truth:null;let vr=0;if(e){const R=p3(RADAR),d=[e.pos[0]-R[0],e.pos[1]-R[1],e.pos[2]-R[2]],L=Math.hypot(...d)||1;vr=(e.vel[0]*d[0]+e.vel[1]*d[1]+e.vel[2]*d[2])/L}
      const vals=new Float32Array(n*m.ny);
      for(let k=0;k<n;k++){SYN.microDopplerColumn(m.col,m.ny,m.vmax,e,vr,W.t-(n-k)/40);for(let i=0;i<m.ny;i++)vals[(m.ny-1-i)*n+k]=m.col[i]}
      paint(img,vals,INFERNO,2,32);c.putImageData(img,240-n,0)}
  },
  computeRD(){
    const{W}=S();const nx=this.nx,ny=this.ny,vmax=this.vmax,rmax=RADAR.range;if(!this.grid)this.grid=new Float32Array(nx*ny);const G2=this.grid;
    for(let i=0;i<G2.length;i++)G2[i]=expn()*1.6; // thermal noise
    for(let x=0;x<nx;x++){const r=(x+0.5)/nx*rmax,clut=34-26*Math.log10(1+r/180);for(let dy=-2;dy<=2;dy++){const y=Math.round((ny-1)/2)+dy;G2[y*nx+x]=Math.max(G2[y*nx+x],clut-Math.abs(dy)*9+gn()*2)}}
    const R=p3(RADAR);
    for(const e of W.ents){if(!e.rcs||!e.alive||(e.type==="aircraft"&&e.wait>0))continue;if(e.pos[2]<0.5&&e.type!=="car")continue;
      const d=[e.pos[0]-R[0],e.pos[1]-R[1],e.pos[2]-R[2]],r=Math.hypot(...d);if(r>rmax||r<25)continue;
      const vr=(e.vel[0]*d[0]+e.vel[1]*d[1]+e.vel[2]*d[2])/r,P=SYN.radarPower(e,r);if(P<3)continue;
      const x=Math.min(nx-1,Math.floor(r/rmax*nx));
      for(const[v,db]of SYN.dopplerComponents(e,vr,P,W.t)){const y=Math.round((v+vmax)/(2*vmax)*(ny-1));if(y<0||y>=ny)continue;
        for(let j=-2;j<=2;j++)for(let i=-2;i<=2;i++){const yy=y+j,xx=x+i;if(yy<0||yy>=ny||xx<0||xx>=nx)continue;const k=yy*nx+xx;G2[k]=Math.max(G2[k],db-(i*i*1.6+j*j*2.2))}}}
  },
  draw(ctx,w,h,dt){
    const{W,UI:U}=S();const P=pal();ctx.fillStyle=P.bg;ctx.fillRect(0,0,w,h);if(!W)return;
    this.tAcc+=dt;if(!this.small||this.tAcc>0.12){this.tAcc=0;this.computeRD();
      if(!this.small){this.small=document.createElement("canvas");this.small.width=this.nx;this.small.height=this.ny}
      const sc=this.small.getContext("2d"),img=sc.createImageData(this.nx,this.ny);
      const flip=new Float32Array(this.nx*this.ny);for(let y=0;y<this.ny;y++)for(let x=0;x<this.nx;x++)flip[(this.ny-1-y)*this.nx+x]=this.grid[y*this.nx+x];
      paint(img,flip,INFERNO,0,52);sc.putImageData(img,0,0)}
    const top=22,padL=40,padR=10,H1=Math.round((h-top)*0.55),pw=w-padL-padR,ph=H1-22;
    ctx.imageSmoothingEnabled=true;ctx.drawImage(this.small,padL,top,pw,ph);
    // axes
    ctx.strokeStyle="rgba(213,225,234,.25)";ctx.lineWidth=1;ctx.strokeRect(padL,top,pw,ph);ctx.font=F(9.5);ctx.fillStyle=P.muted;
    for(const km of[0,0.5,1,1.5,2]){const x=padL+km*1000/RADAR.range*pw;ctx.fillText(`${km}`,x-4,top+ph+11)}ctx.fillText("range km",padL+pw-48,top+ph+11);
    for(const v of[-30,-15,0,15,30]){const y=top+ph/2-v/this.vmax*ph/2;ctx.fillText(`${v>0?"+":""}${v}`,6,y+3);ctx.strokeStyle="rgba(213,225,234,.08)";ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(padL+pw,y);ctx.stroke()}
    ctx.save();ctx.translate(12,top+ph/2+40);ctx.rotate(-Math.PI/2);ctx.fillText("radial m/s",0,0);ctx.restore();
    ctx.fillStyle="rgba(213,225,234,.55)";ctx.textAlign="right";ctx.fillText("ground clutter (0 m/s)",padL+pw-6,top+ph/2-6);ctx.textAlign="left";
    // track markers
    const R=p3(RADAR);for(const tr of W.tracks){if(!tr.confirmed||tr.hidden)continue;const d=[tr.p[0]-R[0],tr.p[1]-R[1],tr.p[2]-R[2]],r=Math.hypot(...d);if(r>RADAR.range)continue;const vr=(tr.v[0]*d[0]+tr.v[1]*d[1]+tr.v[2]*d[2])/r;
      const x=padL+r/RADAR.range*pw,y=top+ph/2-vr/this.vmax*ph/2;ctx.strokeStyle=trackColor(tr);ctx.lineWidth=1.3;ctx.beginPath();ctx.arc(x,y,6,0,7);ctx.stroke();if(tr.threat!=="TRACK"||tr===selTrack()){ctx.font=F(9.5,600);ctx.fillStyle=trackColor(tr);ctx.fillText(tr.id,x+8,y-4)}}
    // micro-Doppler strip
    const y2=top+H1+4,h2=h-y2-8,m=this.md,tr=selTrack();
    ctx.drawImage(m.buf,padL,y2+14,pw,h2-14);ctx.strokeStyle="rgba(213,225,234,.25)";ctx.strokeRect(padL,y2+14,pw,h2-14);
    ctx.font=F(10.5,700);ctx.fillStyle=P.ink;ctx.fillText(`MICRO-DOPPLER · ${tr?tr.id:"no track"} · last 6 s`,padL,y2+10);
    ctx.font=F(9.5);ctx.fillStyle=P.muted;for(const v of[-40,0,40]){const y=y2+14+(h2-14)/2-v/m.vmax*(h2-14)/2;ctx.fillText(`${v>0?"+":""}${v}`,8,y+3)}
    if(tr&&tr.truth){const e=tr.truth;let txt,col;if(e.type==="drone"){txt="Blade-flash spread ±34 m/s · rotor sidebands → multirotor";col=P.crit}else if(e.type==="bird"){txt=`Periodic wing-tip trace ~${(e.flapHz||4).toFixed(1)} Hz → bird`;col=P.ok}else if(e.type==="car"){txt="Narrow body line, ground speed → vehicle";col=P.muted}else{txt="Narrow line, no micro-motion";col=P.warn}
      chip(ctx,padL+6,y2+30,txt,col)}
    else if(!tr){ctx.fillStyle=P.muted;ctx.fillText("Select or cue a track to see its micro-Doppler signature",padL+10,y2+40)}
    header(ctx,w,"RANGE-DOPPLER · coherent dwell 64 pulses",`${W.radarPlots.filter(p=>W.t-p.t<0.6).length} detections`);
    if(U.radar===false)offline(ctx,w,h,"RADAR");
  }};

// ======================================================================
// 3. RF spectrum + waterfall (2.4 GHz and 5.8 GHz ISM)
// ======================================================================
const BANDS=[{id:"24",name:"2.4 GHz ISM",f0:2400,f1:2483.5,n:167,wifi:[{f:2412,bw:20,db:-74,duty:0.07},{f:2437,bw:20,db:-70,duty:0.12},{f:2462,bw:20,db:-78,duty:0.05}]},
             {id:"58",name:"5.8 GHz",f0:5725,f1:5850,n:125,wifi:[{f:5745,bw:20,db:-72,duty:0.08},{f:5785,bw:40,db:-70,duty:0.06}]}];
const rfWaterfall={rows:null,img:null,acc:0,emState:{},last:null,
  emitters(){const{W}=S();const out=[];if(!W)return out;const R0=typeof RFS!=="undefined"?p3(RFS[0]):[0,0,10];
    for(const em of Object.values(W.emitters||{})){
      const heard=W.rfDets.some(r=>r.em===em.id&&W.t-r.t<1.6);if(!heard)continue;
      let st=this.emState[em.id];if(!st){st=this.emState[em.id]={videoF:5745+20+((em.id.charCodeAt(2)*37)%60),ridSlot:em.id.charCodeAt(2)%10}}
      let pos=null,airborne=false;
      if(em.kind==="link"&&em.drone){pos=em.drone.pos;airborne=em.drone.pos[2]>1}
      else{const sc=(W.sc&&(W.sc.rfDecoys||[]).concat(W.sc.spoofers||[]))||[];const s=sc.find(x=>x.id===em.id);if(s)pos=[s.tx[0],s.tx[1],2]}
      const r=pos?Math.max(50,d3(R0,pos)):1000;const level=-28-20*Math.log10(r/100);
      const brg=W.rfDets.filter(x=>x.em===em.id&&x.kind==="drone").slice(-1)[0]||W.rfDets.filter(x=>x.em===em.id).slice(-1)[0];
      out.push({id:em.id,kind:em.kind,level,airborne,videoF:st.videoF,videoBW:20,rid:!!(em.rid&&W.t-em.rid.t<3),ridSlot:st.ridSlot,spoof:!!em.spoof,unassoc:!!em.unassoc,brg:brg?brg.b:null})}
    return out},
  update(dt){
    const{W,UI:U}=S();if(!W)return;
    if(!this.rows){this.rows=BANDS.map(b=>{const c=document.createElement("canvas");c.width=b.n;c.height=120;const x=c.getContext("2d");x.fillStyle="#440154";x.fillRect(0,0,b.n,120);return{c,row:new Float32Array(b.n),avg:new Float32Array(b.n).fill(-98),peak:new Float32Array(b.n).fill(-98)}})}
    this.acc+=dt*24;const n=Math.min(8,Math.floor(this.acc));this.acc-=n;if(n<=0)return;
    const ems=U.rf===false?[]:this.emitters();this.last=ems;
    BANDS.forEach((b,bi)=>{const R=this.rows[bi],x=R.c.getContext("2d");x.drawImage(R.c,0,n);
      for(let k=0;k<n;k++){SYN.rfRow(R.row,b,W.t-k/24,{emitters:ems});const img=x.createImageData(b.n,1);paint(img,R.row,VIRIDIS,-100,-40);x.putImageData(img,0,n-1-k);
        for(let i=0;i<b.n;i++){R.avg[i]=R.avg[i]*0.85+R.row[i]*0.15;R.peak[i]=Math.max(R.peak[i]-0.6,R.row[i])}}})
  },
  draw(ctx,w,h,dt){
    const{W,UI:U}=S();const P=pal();ctx.fillStyle=P.bg;ctx.fillRect(0,0,w,h);if(!W||!this.rows)return;
    const top=26,gap=10,padL=34,bw=(w-padL-8-gap)/2,specH=Math.round((h-top)*0.32),wfY=top+specH+16,wfH=h-wfY-26;
    BANDS.forEach((b,bi)=>{const R=this.rows[bi],x0=padL+bi*(bw+gap);
      // spectrum trace (avg + peak hold)
      ctx.fillStyle="#0a1016";ctx.fillRect(x0,top,bw,specH);ctx.strokeStyle="rgba(213,225,234,.08)";for(let db=-100;db<=-40;db+=20){const y=top+specH*(1-(db+100)/60);ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x0+bw,y);ctx.stroke();if(bi===0){ctx.font=F(9);ctx.fillStyle=P.muted;ctx.fillText(db,4,y+3)}}
      const yv=v=>top+specH*(1-Math.max(0,Math.min(1,(v+100)/60)));
      const grd=ctx.createLinearGradient(0,top,0,top+specH);grd.addColorStop(0,"rgba(76,194,216,.45)");grd.addColorStop(1,"rgba(76,194,216,.02)");
      ctx.beginPath();ctx.moveTo(x0,top+specH);for(let i=0;i<b.n;i++)ctx.lineTo(x0+i/(b.n-1)*bw,yv(R.avg[i]));ctx.lineTo(x0+bw,top+specH);ctx.closePath();ctx.fillStyle=grd;ctx.fill();
      ctx.strokeStyle=P.accent;ctx.lineWidth=1.2;ctx.beginPath();for(let i=0;i<b.n;i++){const X=x0+i/(b.n-1)*bw;i?ctx.lineTo(X,yv(R.avg[i])):ctx.moveTo(X,yv(R.avg[i]))}ctx.stroke();
      ctx.strokeStyle="rgba(240,169,59,.55)";ctx.lineWidth=1;ctx.beginPath();for(let i=0;i<b.n;i++){const X=x0+i/(b.n-1)*bw;i?ctx.lineTo(X,yv(R.peak[i])):ctx.moveTo(X,yv(R.peak[i]))}ctx.stroke();
      ctx.font=F(10,700);ctx.fillStyle=P.ink;ctx.fillText(b.name,x0+4,top+12);
      // waterfall
      ctx.imageSmoothingEnabled=false;ctx.drawImage(R.c,x0,wfY,bw,wfH);ctx.imageSmoothingEnabled=true;ctx.strokeStyle="rgba(213,225,234,.2)";ctx.strokeRect(x0,wfY,bw,wfH);
      ctx.font=F(9);ctx.fillStyle=P.muted;const ticks=bi===0?[2405,2420,2440,2460]:[5740,5760,5800,5830];for(const f of ticks){const X=x0+(f-b.f0)/(b.f1-b.f0)*bw;ctx.fillText(f,X-12,h-14)}
      // annotations for emitters in this band
      let ay=wfY+14;for(const e of (this.last||[])){
        const col=e.spoof?P.warn:e.unassoc?P.warn:P.rf;
        if(bi===0&&(e.kind==="link"||e.kind==="decoy")){chip(ctx,x0+4,ay,`${e.id} FHSS ctrl${e.kind==="decoy"?" · no drone (decoy?)":bw>300?" · OcuSync-class":""}${e.brg!=null?` · ${Math.round(e.brg)}°`:""}`,col);ay+=19}
        if(bi===0&&e.rid){chip(ctx,x0+4,ay,`${e.id} Remote ID${bw>300?" beacon (BT adv + Wi-Fi NAN)":""}${e.spoof?" · SPOOF":""}`,col);ay+=19}
        if(bi===1&&e.kind==="link"&&e.airborne){const X=x0+(e.videoF-b.f0)/(b.f1-b.f0)*bw;ctx.strokeStyle=col;ctx.setLineDash([3,3]);ctx.strokeRect(X-10/(b.f1-b.f0)*bw*2,wfY+2,20/(b.f1-b.f0)*bw*2/2+8,wfH-4);ctx.setLineDash([]);chip(ctx,Math.min(X+8,x0+bw-(bw>300?150:96)),ay,bw>300?`${e.id} video downlink 20 MHz`:`${e.id} video`,col);ay+=19}}
    });
    ctx.font=F(9);ctx.fillStyle=P.muted;ctx.fillText("MHz · newest at top · avg (cyan) / peak hold (amber)",padL,h-3);
    header(ctx,w,"RF SPECTRUM · passive direction-finding receivers",`${(this.last||[]).length} emitter(s)`);
    if(U.rf===false)offline(ctx,w,h,"RF");
  }};

// ======================================================================
// 4. Acoustic spectrogram (per node)
// ======================================================================
const NF=96;const FREQS=Float32Array.from({length:NF},(_,i)=>50*Math.pow(4000/50,i/(NF-1)));
const acousticSpectrogram={node:null,auto:true,buf:null,col:new Float32Array(NF),acc:0,marks:[],lastNode:null,
  pickNode(){const{W}=S();if(typeof ACS==="undefined")return null;if(!this.auto&&this.node)return ACS.find(n=>n.id===this.node)||ACS[0];
    const tr=selTrack();if(!tr)return ACS[0];let b=ACS[0],bd=1e9;for(const n of ACS){const d=hyp(n.x-tr.p[0],n.y-tr.p[1]);if(d<bd){bd=d;b=n}}return b},
  sources(node){const{W}=S();const out=[];const np=p3(node);
    for(const e of W.ents){if(!e.alive)continue;
      if(e.type==="drone"&&e.pos[2]>0.8&&e.phase!=="done"){const r=d3(np,e.pos);if(r>450)continue;const sp=hyp(e.vel[0],e.vel[1]),climb=e.vel[2];
        const d=[e.pos[0]-np[0],e.pos[1]-np[1],e.pos[2]-np[2]],L=Math.hypot(...d)||1,vr=(e.vel[0]*d[0]+e.vel[1]*d[1]+e.vel[2]*d[2])/L;
        const rpm=5600+sp*90+climb*220+(e.hasPayload?500:0),bpf=rpm/60*2*(1-vr/343);out.push({kind:"drone",bpf,level:52-20*Math.log10(Math.max(r,8)/10),r,id:e.name})}
      else if(e.type==="car"){const r=d3(np,e.pos);if(r<700)out.push({kind:"truck",level:26-20*Math.log10(r/100)})}
      else if(e.type==="aircraft"&&!(e.wait>0)){const r=d3(np,e.pos);if(r<2500)out.push({kind:"aircraft",level:20-20*Math.log10(r/1000)})}}
    return out},
  update(dt){
    const{W,UI:U}=S();if(!W||typeof ACS==="undefined")return;
    if(!this.buf){this.buf=document.createElement("canvas");this.buf.width=220;this.buf.height=NF;const c=this.buf.getContext("2d");c.fillStyle="#000004";c.fillRect(0,0,220,NF)}
    const node=this.pickNode();if(node&&this.lastNode!==node.id){this.lastNode=node.id;const c=this.buf.getContext("2d");c.fillStyle="#000004";c.fillRect(0,0,220,NF);this.marks=[]}
    this.acc+=dt*28;const n=Math.min(10,Math.floor(this.acc));this.acc-=n;if(n<=0||!node)return;
    const src=U.ac===false?[]:this.sources(node);this.lastSrc=src;
    const c=this.buf.getContext("2d");c.drawImage(this.buf,-n,0);const img=c.createImageData(n,NF),vals=new Float32Array(n*NF);
    for(let k=0;k<n;k++){SYN.acousticColumn(this.col,FREQS,node,W.t,{UI:U,sources:src});for(let i=0;i<NF;i++)vals[(NF-1-i)*n+k]=this.col[i]}
    paint(img,vals,INFERNO,0,48);c.putImageData(img,220-n,0);
    this.marks=this.marks.map(m=>({...m,x:m.x-n})).filter(m=>m.x>-4);
    const np=p3(node);for(const a of (W.acDets||[])){if(a.n[0]!==np[0]||a.n[1]!==np[1]||a._seenBy===this)continue;if(W.t-a.t<0.15&&!this.marks.some(m=>m.t===a.t))this.marks.push({x:219,t:a.t,fa:a.fa})}
  },
  draw(ctx,w,h,dt){
    const{W,UI:U}=S();const P=pal();ctx.fillStyle=P.bg;ctx.fillRect(0,0,w,h);if(!W||!this.buf)return;
    const node=this.pickNode(),top=42,padL=42,pw=w-padL-10,ph=h-top-24;
    ctx.imageSmoothingEnabled=true;ctx.drawImage(this.buf,padL,top,pw,ph);ctx.strokeStyle="rgba(213,225,234,.25)";ctx.strokeRect(padL,top,pw,ph);
    ctx.font=F(9.5);ctx.fillStyle=P.muted;
    for(const f of[50,100,200,500,1000,2000,4000]){const y=top+ph-(Math.log(f/50)/Math.log(80))*ph;ctx.fillText(f>=1000?`${f/1000}k`:`${f}`,8,y+3);ctx.strokeStyle="rgba(213,225,234,.08)";ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(padL+pw,y);ctx.stroke()}
    ctx.fillText("Hz",w-26,top-4);ctx.fillText("−8 s",padL,h-8);ctx.fillText("now",padL+pw-22,h-8);
    for(const m of this.marks){const x=padL+m.x/220*pw;ctx.fillStyle=m.fa?P.warn:P.crit;ctx.fillRect(x-1,top,2,6);ctx.font=F(8.5,700);ctx.fillText(m.fa?"FA":"DET",x-8,top+15)}
    const dr=(this.lastSrc||[]).filter(s=>s.kind==="drone").sort((a,b)=>a.r-b.r)[0];
    if(dr){const y=top+ph-(Math.log(dr.bpf/50)/Math.log(80))*ph;ctx.strokeStyle="rgba(255,91,82,.8)";ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(padL+pw,y);ctx.stroke();ctx.setLineDash([]);
      chip(ctx,padL+6,Math.max(top+28,y-6),`Drone ${dr.id}: blade-pass ${Math.round(dr.bpf)} Hz + harmonics · ${Math.round(dr.r)} m → multirotor`,P.crit)}
    else{ctx.font=F(10);ctx.fillStyle=P.muted;ctx.fillText(U.wind>8?"Wind noise is masking the low band":"No rotor harmonics in range of this node",padL+8,top+18)}
    // node selector strip (click to choose; 'auto' follows the cued track)
    ctx.font=F(10,600);let x=8;const items=["AUTO"].concat((typeof ACS!=="undefined"?ACS:[]).map(n=>n.id));this._hit=[];
    for(const it of items){const on=it==="AUTO"?this.auto:(!this.auto&&node&&node.id===it);const tw=ctx.measureText(it).width+12;ctx.fillStyle=on?"#173640":"#111922";ctx.fillRect(x,22-1,tw,13);ctx.strokeStyle=on?P.accent:P.line;ctx.strokeRect(x,22-1,tw,13);ctx.fillStyle=on?"#e9fbff":P.muted;ctx.fillText(it,x+6,31);this._hit.push([x,21,tw,13,it]);x+=tw+4}
    header(ctx,w,`ACOUSTIC · ${node?node.id:"none"} · MEMS array`,`wind ${U.wind||0} m/s${U.wx==="rain"?" · rain":""}`);
    if(U.ac===false)offline(ctx,w,h,"ACOUSTIC");
  },
  click(x,y){for(const[a,b,c,d,id]of(this._hit||[])){if(x>=a&&x<=a+c&&y>=b&&y<=b+d){if(id==="AUTO"){this.auto=true}else{this.auto=false;this.node=id}return true}}return false}};

// ======================================================================
// mount + frame loop
// ======================================================================
const TABS=[["radarPPI","Radar scope"],["rangeDoppler","Range-Doppler"],["rfWaterfall","RF spectrum"],["acousticSpectrogram","Acoustic"]];
const VIEWS={radarPPI,rangeDoppler,rfWaterfall,acousticSpectrogram,active:"radarPPI",selectedTrack:null,timings:{},canvas:null,ctx:null,
  setSynth(o){Object.assign(SYN,o||{})},
  mount(container){
    const wrap=document.createElement("div");wrap.className="views-wrap";
    wrap.innerHTML=`<style>.views-wrap{display:flex;flex-direction:column;gap:6px}.views-tabs{display:flex;flex-wrap:wrap;gap:4px}.views-tabs button{font:600 12px var(--display,system-ui);background:#18232e;color:var(--ink,#d5e1ea);border:1px solid var(--line,#21303d);border-radius:4px;padding:4px 9px;cursor:pointer}.views-tabs button[aria-selected="true"]{background:#173640;border-color:var(--accent,#4cc2d8);color:#e9fbff}.views-tabs button:focus-visible{outline:2px solid var(--accent,#4cc2d8);outline-offset:1px}.views-wrap canvas{width:100%;display:block;border-radius:4px;background:#070b0f}</style>
      <div class="views-tabs" role="tablist" aria-label="Sensor views"></div><canvas role="tabpanel" aria-label="Sensor view"></canvas>`;
    const tl=wrap.querySelector(".views-tabs"),cv=wrap.querySelector("canvas");
    TABS.forEach(([k,label],i)=>{const b=document.createElement("button");b.type="button";b.setAttribute("role","tab");b.id="vt-"+k;b.textContent=label;b.setAttribute("aria-selected",String(k===this.active));b.tabIndex=k===this.active?0:-1;
      b.onclick=()=>this.select(k);b.onkeydown=e=>{if(e.key==="ArrowRight"||e.key==="ArrowLeft"){const j=(i+(e.key==="ArrowRight"?1:TABS.length-1))%TABS.length;this.select(TABS[j][0]);tl.children[j].focus();e.preventDefault()}};tl.appendChild(b)});
    container.appendChild(wrap);this.canvas=cv;this.ctx=cv.getContext("2d");this.tabs=tl;
    const fit=()=>{const r=cv.getBoundingClientRect(),dpr=Math.min(2,window.devicePixelRatio||1),W_=Math.max(240,Math.round(r.width)),H_=Math.round(W_*0.68);if(this.cw===W_&&this.ch===H_&&this.dpr===dpr)return;cv.style.height=H_+"px";cv.width=Math.round(W_*dpr);cv.height=Math.round(H_*dpr);this.cw=W_;this.ch=H_;this.dpr=dpr;if(this.ctx)try{this.frame(0)}catch(e){}};
    fit();if(window.ResizeObserver)new ResizeObserver(fit).observe(wrap);
    cv.addEventListener("click",e=>{const r=cv.getBoundingClientRect(),x=(e.clientX-r.left),y=(e.clientY-r.top);if(this.active==="acousticSpectrogram")acousticSpectrogram.click(x,y)});
    return wrap},
  select(k){this.active=k;if(this.tabs)[...this.tabs.children].forEach((b,i)=>{const on=TABS[i][0]===k;b.setAttribute("aria-selected",String(on));b.tabIndex=on?0:-1})},
  frame(dt){
    dt=Math.max(0,Math.min(0.25,dt||0));
    const t0=performance.now();rangeDoppler.update(dt);rfWaterfall.update(dt);acousticSpectrogram.update(dt);this.timings.update=performance.now()-t0;
    if(!this.ctx)return;const c=this.ctx;c.setTransform(this.dpr,0,0,this.dpr,0,0);const t1=performance.now();
    this[this.active].draw(c,this.cw,this.ch,dt);this.timings[this.active]=performance.now()-t1}
};
window.VIEWS=VIEWS;
})();
var VIEWS_SYNTH_DEFAULT;
