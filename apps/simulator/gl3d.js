// ================= GL3D: realistic WebGL2 renderer for the 3D view =================
// Raw WebGL2, no libraries. HDR scene (MSAA) -> bloom -> ACES tonemap. Directional shadow map,
// procedural terrain/materials, floodlight point lights, instanced trees, sky with sun/moon/stars/clouds,
// distance fog, rain. The 2D canvas on top keeps drawing all analytic overlays with the SAME camera (C3).
const GL3D=(function(){
"use strict";
const api={ok:false,err:null,frameMs:0};
let gl=null,cv=null;
const PR={},BUF={},TG={};
let SW=0,SH=0,MSAA=0,HDRFMT=null,shadowTex=null,shadowFB=null;const SHADOW=2048;
const NEARP=1.0,FARP=26000;
let STATIC=null,FENCEG=null,TREEG=[],GROUND=null,inited=false;

// ---------------- helpers ----------------
const lin=h=>h.match(/\w\w/g).map(x=>Math.pow(parseInt(x,16)/255,2.2));
function hash2(x,y){let h=Math.sin(x*127.1+y*311.7)*43758.5453;return h-Math.floor(h)}
function vnoise(x,y){const ix=Math.floor(x),iy=Math.floor(y),fx=x-ix,fy=y-iy,ux=fx*fx*(3-2*fx),uy=fy*fy*(3-2*fy);
  const a=hash2(ix,iy),b=hash2(ix+1,iy),c=hash2(ix,iy+1),d=hash2(ix+1,iy+1);return a+(b-a)*ux+(c-a)*uy+(a-b-c+d)*ux*uy}
function fbm2(x,y){let s=0,a=0.5;for(let i=0;i<5;i++){s+=a*vnoise(x,y);x=x*2.03+17.1;y=y*2.03+9.2;a*=0.5}return s}
const sstep=(a,b,x)=>{const t=Math.min(1,Math.max(0,(x-a)/(b-a)));return t*t*(3-2*t)};
// terrain height: flat over the whole operating area, rolling hills beyond it
function hgt(x,y){const dx=Math.max(Math.abs(x)-2750,0),dy=Math.max(Math.abs(y-50)-1850,0);const d=Math.hypot(dx,dy);if(d<=0)return 0;
  const m=sstep(0,2600,d);return m*(40+300*fbm2(x*0.00035,y*0.00035))+m*m*90*fbm2(x*0.0011+5,y*0.0011)}
api.hgt=hgt;

function Builder(stride){this.s=stride;this.a=new Float32Array(stride*4096);this.n=0}
Builder.prototype.push=function(arr){if(this.n+arr.length>this.a.length){let L=this.a.length*2;while(L<this.n+arr.length)L*=2;const t=new Float32Array(L);t.set(this.a);this.a=t}this.a.set(arr,this.n);this.n+=arr.length};
Builder.prototype.count=function(){return this.n/this.s};
// solid vertex: pos3 nrm3 col3 mat1
function S_tri(B,p0,p1,p2,c,m){const ux=p1[0]-p0[0],uy=p1[1]-p0[1],uz=p1[2]-p0[2],vx=p2[0]-p0[0],vy=p2[1]-p0[1],vz=p2[2]-p0[2];
  let nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;const l=Math.hypot(nx,ny,nz)||1;nx/=l;ny/=l;nz/=l;
  B.push([p0[0],p0[1],p0[2],nx,ny,nz,c[0],c[1],c[2],m,p1[0],p1[1],p1[2],nx,ny,nz,c[0],c[1],c[2],m,p2[0],p2[1],p2[2],nx,ny,nz,c[0],c[1],c[2],m])}
function S_quad(B,a,b,c_,d,c,m){S_tri(B,a,b,c_,c,m);S_tri(B,a,c_,d,c,m)}
function xf(cx,cy,cz,yaw,pitch){const cy_=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch||0),sp=Math.sin(pitch||0);
  return(x,y,z)=>{const y2=y*cp-z*sp,z2=y*sp+z*cp;return[cx+x*cy_+y2*sy,cy-x*sy+y2*cy_,cz+z2]}}
function S_box(B,cx,cy,cz,hx,hy,hz,yaw,pitch,c,m,noBottom){const T=xf(cx,cy,cz,yaw,pitch);
  const p=[T(-hx,-hy,-hz),T(hx,-hy,-hz),T(hx,hy,-hz),T(-hx,hy,-hz),T(-hx,-hy,hz),T(hx,-hy,hz),T(hx,hy,hz),T(-hx,hy,hz)];
  S_quad(B,p[4],p[5],p[6],p[7],c,m);S_quad(B,p[0],p[1],p[5],p[4],c,m);S_quad(B,p[1],p[2],p[6],p[5],c,m);S_quad(B,p[2],p[3],p[7],p[6],c,m);S_quad(B,p[3],p[0],p[4],p[7],c,m);if(!noBottom)S_quad(B,p[0],p[3],p[2],p[1],c,m)}
function S_aabb(B,x0,y0,z0,x1,y1,z1,c,m){S_box(B,(x0+x1)/2,(y0+y1)/2,(z0+z1)/2,(x1-x0)/2,(y1-y0)/2,(z1-z0)/2,0,0,c,m,true)}
function S_cyl(B,cx,cy,z0,z1,r,n,c,m,cap){for(let i=0;i<n;i++){const a0=i/n*6.2832,a1=(i+1)/n*6.2832;const p0=[cx+Math.cos(a0)*r,cy+Math.sin(a0)*r],p1=[cx+Math.cos(a1)*r,cy+Math.sin(a1)*r];
  S_quad(B,[p0[0],p0[1],z0],[p1[0],p1[1],z0],[p1[0],p1[1],z1],[p0[0],p0[1],z1],c,m);if(cap)S_tri(B,[cx,cy,z1],[p0[0],p0[1],z1],[p1[0],p1[1],z1],c,m)}}
function S_seg(B,a,b,r,c,m){const dx=b[0]-a[0],dy=b[1]-a[1],dz=b[2]-a[2],L=Math.hypot(dx,dy,dz)||1;const yaw=Math.atan2(dx,dy),pitch=Math.asin(dz/L);
  S_box(B,(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2,r,L/2,r,yaw,pitch,c,m)}

// ---------------- GLSL ----------------
const COMMON=`precision highp float;precision highp int;precision highp sampler2DShadow;
uniform vec3 uEye,uSunDir,uSunCol,uAmbSky,uAmbGnd,uWts,uFogCol;uniform float uTime,uLightsOn,uFogDist,uOvercast,uWet,uShadowOn;
uniform vec4 uLights[40];uniform int uNL;uniform sampler2DShadow uShadow;uniform mat4 uShadowVP;uniform vec2 uWind;
float hash12(vec2 p){vec3 p3=fract(vec3(p.xyx)*.1031);p3+=dot(p3,p3.yzx+33.33);return fract((p3.x+p3.y)*p3.z);}
float hash11(float p){p=fract(p*.1031);p*=p+33.33;p*=p+p;return fract(p);}
float vnoise(vec2 p){vec2 i=floor(p),f=fract(p);vec2 u=f*f*(3.-2.*f);return mix(mix(hash12(i),hash12(i+vec2(1,0)),u.x),mix(hash12(i+vec2(0,1)),hash12(i+vec2(1,1)),u.x),u.y);}
float fbm(vec2 p){float s=0.,a=.5;for(int i=0;i<5;i++){s+=a*vnoise(p);p=p*2.03+vec2(17.1,9.2);a*=.5;}return s;}
float fbm3(vec2 p){float s=0.,a=.5;for(int i=0;i<3;i++){s+=a*vnoise(p);p=p*2.03+vec2(17.1,9.2);a*=.5;}return s;}
vec3 skyCol(vec3 d){float mu=d.z;float m=max(mu,0.);float sd=max(dot(d,uSunDir),0.);
 vec3 day=mix(vec3(.60,.72,.86),vec3(.11,.29,.68),pow(m,.45))*1.7;day+=vec3(1.,.9,.72)*(pow(sd,6.)*.35+pow(sd,64.)*.8);
 vec3 dusk=mix(vec3(1.35,.58,.24),vec3(.13,.15,.38),pow(m,.33))*.95;dusk+=vec3(1.5,.62,.26)*(pow(sd,7.)*.9+pow(sd,90.)*2.);
 vec3 night=mix(vec3(.034,.04,.058),vec3(.004,.008,.02),pow(m,.4));night+=vec3(.06,.036,.02)*exp(-m*22.);night+=vec3(.45,.55,.75)*pow(sd,40.)*.06;
 vec3 c=day*uWts.x+dusk*uWts.y+night*uWts.z;
 vec3 oc=mix(vec3(.62,.65,.68),vec3(.48,.51,.56),m)*(uWts.x*1.25+uWts.y*.45+uWts.z*.028);
 c=mix(c,oc,uOvercast);if(mu<0.)c*=mix(1.,.75,clamp(-mu*5.,0.,1.));return c;}
float shadowF(vec3 wp,vec3 n){if(uShadowOn<.5)return 1.;vec4 s=uShadowVP*vec4(wp+n*.25,1.);vec3 c=s.xyz/s.w*.5+.5;
 if(c.x<0.||c.y<0.||c.x>1.||c.y>1.||c.z>1.)return 1.;vec2 ts=1./vec2(textureSize(uShadow,0));float sum=0.;
 for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)sum+=texture(uShadow,vec3(c.xy+vec2(x,y)*ts*1.25,c.z-.0012));return mix(1.,sum/9.,uShadowOn);}
vec3 sodium(vec3 p,vec3 n,vec3 lp,float I){vec3 L=lp-p;float d2=dot(L,L);vec3 l=L*inversesqrt(d2);float spot=smoothstep(.0,.55,l.z)*.85+.15;
 float fall=pow(clamp(1.-d2/7000.,0.,1.),2.);return vec3(1.,.55,.21)*I/(d2+25.)*fall*spot*max(dot(n,l),0.);}
vec3 pointLights(vec3 p,vec3 n){vec3 acc=vec3(0);if(uLightsOn<.01)return acc;
 if(abs(p.x)<330.&&abs(p.y)<300.){for(int i=0;i<40;i++){if(i>=uNL)break;vec3 L=uLights[i].xyz-p;if(dot(L,L)>7000.)continue;acc+=sodium(p,n,uLights[i].xyz,uLights[i].w);}}
 if(abs(p.y+1145.)<60.){float lx=floor(p.x/70.+.5)*70.;acc+=sodium(p,n,vec3(lx,-1140.,8.),260.);acc+=sodium(p,n,vec3(lx+70.*sign(p.x-lx),-1140.,8.),260.);}
 if(abs(p.x-1345.)<60.&&p.y>-1110.&&p.y<1620.){float ly=floor((p.y+1100.)/90.+.5)*90.-1100.;acc+=sodium(p,n,vec3(1340.,ly,8.),260.);}
 return acc*uLightsOn;}
vec3 shade(vec3 p,vec3 n,vec3 alb,float sh){vec3 d=uSunCol*max(dot(n,uSunDir),0.)*sh;vec3 amb=mix(uAmbGnd,uAmbSky,n.z*.5+.5);return alb*(d+amb+pointLights(p,n));}
vec3 applyFog(vec3 c,vec3 p){vec3 v=p-uEye;float dist=length(v);vec3 dir=v/dist;vec3 fc=skyCol(normalize(vec3(dir.xy,max(dir.z,.06))))*mix(1.,.55,uWts.z);
 float f=1.-exp(-dist/uFogDist);float hf=exp(-max(p.z,0.)/(uFogDist<1000.?120.:600.));return mix(c,fc,clamp(f*mix(.55,1.,hf),0.,1.));}
`;
const VS_SOLID=`#version 300 es
layout(location=0) in vec3 aPos;layout(location=1) in vec3 aNrm;layout(location=2) in vec3 aCol;layout(location=3) in float aMat;
uniform mat4 uVP;out vec3 vW;out vec3 vN;out vec3 vC;flat out float vM;
void main(){vW=aPos;vN=aNrm;vC=aCol;vM=aMat;gl_Position=uVP*vec4(aPos,1.);}`;
const FS_SOLID=`#version 300 es
${COMMON}
in vec3 vW;in vec3 vN;in vec3 vC;flat in float vM;uniform float uAlpha;out vec4 o;
void main(){vec3 n=normalize(vN);if(!gl_FrontFacing)n=-n;vec3 V=normalize(uEye-vW);vec3 alb=vC;vec3 emis=vec3(0);float spec=0.;int m=int(vM+.5);
 if(m==1){vec2 wc=vec2(abs(n.x)>.5?vW.y:vW.x,vW.z);float stain=fbm3(wc*vec2(.12,.35))*.3+vnoise(wc*3.)*.12;
  alb=vC*(.78+stain)*(1.-.3*smoothstep(1.2,0.,vW.z))*(1.-.18*smoothstep(.0,.6,fbm3(wc*vec2(.6,.05))-.45));
  float panel=1.-smoothstep(.0,.04,abs(fract(wc.x/3.)-.5)-.46);alb*=1.-.12*panel;
  float col=fract(wc.x/6.);float fl=-1.;if(wc.y>2.3&&wc.y<4.5)fl=0.;else if(wc.y>7.3&&wc.y<9.5)fl=1.;
  if(fl>=0.&&col>.33&&col<.67){vec2 cell=vec2(floor(wc.x/6.)+floor(vW.x*.013)*31.+floor(vW.y*.017)*57.,fl+(abs(n.x)>.5?7.:0.));float h=hash12(cell);
   float bars=step(.82,fract((col-.33)/.34*5.));vec3 refl=skyCol(reflect(-V,n));
   alb=vec3(.02);spec=.7;emis=(h>.33?vec3(1.,.70,.40)*(.55+.9*h):vec3(.12,.15,.22)*.25)*uLightsOn*2.4*(1.-bars*.8)+refl*.22*(1.-uLightsOn*.6);}}
 else if(m==2){alb=vec3(.16,.16,.165)*(.7+.55*vnoise(vW.xy*2.5))*(.85+.3*fbm3(vW.xy*.2));}
 else if(m==3){emis=vC*(.08+6.*uLightsOn);alb=vec3(.04);}
 else if(m==4){alb=vec3(.015);spec=.9;emis=vec3(.95,.78,.52)*uLightsOn*1.6+skyCol(reflect(-V,n))*.18;}
 else if(m==5){alb=vC*(.82+.3*vnoise(vW.xy*.9+vW.z*.6));spec=.25;}
 else if(m==6){spec=.85;}
 else if(m==7){emis=vC;alb=vec3(0.);}
 float sh=shadowF(vW,n);vec3 c=shade(vW,n,alb,sh)+emis;
 if(spec>0.){vec3 h=normalize(uSunDir+V);c+=uSunCol*pow(max(dot(n,h),0.),70.)*spec*sh*.6;c+=skyCol(reflect(-V,n))*spec*.06*pow(1.-max(dot(n,V),0.),3.);}
 o=vec4(applyFog(c,vW),uAlpha);}`;
const VS_GROUND=`#version 300 es
layout(location=0) in vec3 aPos;layout(location=1) in vec3 aNrm;uniform mat4 uVP;out vec3 vW;out vec3 vN;
void main(){vW=aPos;vN=aNrm;gl_Position=uVP*vec4(aPos,1.);}`;
function FS_GROUND(){
  const roads=ROADS.map(r=>`{float t;float d=sdSeg(p,vec2(${r.pts[0][0].toFixed(1)},${r.pts[0][1].toFixed(1)}),vec2(${r.pts[1][0].toFixed(1)},${r.pts[1][1].toFixed(1)}),t);float hw=${(r.w*2.2).toFixed(2)};
   if(d<hw+2.2){float sh=smoothstep(hw+2.2,hw+.4,d);alb=mix(alb,vec3(.10,.09,.075)*(.7+.6*vnoise(p*3.)),sh*.8);}
   if(d<hw){alb=vec3(.032,.033,.036)*(.75+.35*vnoise(p*2.3)+.3*d2)*(1.+.25*smoothstep(.2,.9,fbm3(p*.08)));spec=.2;
    if(d<.13&&fract(t/12.)<.5)alb=vec3(.55,.42,.08);if(abs(d-(hw-.45))<.1)alb=vec3(.5);}}`).join("\n");
  const F=FENCE,Y=YARD;
  return `#version 300 es
${COMMON}
in vec3 vW;in vec3 vN;out vec4 o;uniform vec4 uFA[48];uniform vec4 uFB[48];uniform vec3 uFC[48];uniform int uNF;
float sdSeg(vec2 p,vec2 a,vec2 b,out float t){vec2 pa=p-a,ba=b-a;float h=clamp(dot(pa,ba)/dot(ba,ba),0.,1.);t=h*length(ba);return length(pa-ba*h);}
float rectD(vec2 p,vec2 lo,vec2 hi){vec2 c=(lo+hi)*.5,h=(hi-lo)*.5;vec2 q=abs(p-c)-h;return length(max(q,0.))+min(max(q.x,q.y),0.);}
void main(){vec2 p=vW.xy;vec3 n=normalize(vN);float dist=length(vW-uEye);
 float d1=fbm(p*.0035),d2=fbm3(p*.045),d3=vnoise(p*.8);
 vec3 grass=mix(vec3(.040,.068,.022),vec3(.095,.115,.045),d1)*(.78+.4*d2)*(.9+.2*d3);
 vec3 alb=mix(grass,vec3(.12,.095,.065)*(.8+.4*d2),smoothstep(.56,.76,fbm3(p*.0022+3.)));
 alb=mix(alb,vec3(.10,.10,.09),smoothstep(30.,400.,vW.z)*.5);float spec=0.;
 for(int i=0;i<48;i++){if(i>=uNF)break;vec4 A=uFA[i];vec4 B=uFB[i];vec2 q=p-A.xy;vec2 l=vec2(dot(q,A.zw),dot(q,vec2(-A.w,A.z)));
  if(abs(l.x)<B.x&&abs(l.y)<B.y){float rows=.80+.20*sin(l.y*1.7+vnoise(p*.05)*2.)*(1.-smoothstep(300.,1500.,dist));float edge=smoothstep(0.,5.,min(B.x-abs(l.x),B.y-abs(l.y)));
   alb=mix(alb,uFC[i]*rows*(.85+.3*d2),edge*.92);}}
 ${roads}
 // facility: gravel apron, patrol road, concrete compound, yard court, parking lot
 float rp=rectD(p,vec2(${F.x0-26},${F.y0-26}),vec2(${F.x1+26},${F.y1+26}));
 if(rp<0.){alb=vec3(.13,.125,.115)*(.65+.7*vnoise(p*4.5))*(.85+.3*d2);
  float rr=rectD(p,vec2(${F.x0-17},${F.y0-17}),vec2(${F.x1+17},${F.y1+17}));if(rr<0.&&rr>-7.){alb=vec3(.036,.037,.04)*(.8+.4*vnoise(p*2.));spec=.2;}}
 float rf=rectD(p,vec2(${F.x0},${F.y0}),vec2(${F.x1},${F.y1}));
 if(rf<0.){alb=vec3(.19,.19,.18)*(.85+.25*d2)*(.92+.16*vnoise(p*1.7));vec2 j=abs(fract(p/8.)-.5);if(min(j.x,j.y)>.49)alb*=.7;spec=.08;}
 vec2 ylo=vec2(${Y.x0},${Y.y0}),yhi=vec2(${Y.x1},${Y.y1});float ry=rectD(p,ylo,yhi);
 if(ry<0.){alb=vec3(.07,.12,.09)*(.85+.25*vnoise(p*.6));float ln=0.;ln+=1.-smoothstep(.08,.16,abs(ry+.6));
  ln+=1.-smoothstep(.08,.16,abs(p.x));vec2 cc=(ylo+yhi)*.5;ln+=1.-smoothstep(.08,.16,abs(length(p-cc)-9.));
  ln+=(1.-smoothstep(.08,.16,abs(rectD(p,vec2(-6.,ylo.y),vec2(6.,ylo.y+11.)))))*1.;ln+=(1.-smoothstep(.08,.16,abs(rectD(p,vec2(-6.,yhi.y-11.),vec2(6.,yhi.y)))));
  alb=mix(alb,vec3(.62),clamp(ln,0.,1.));}
 float rl=rectD(p,vec2(-202.,-217.),vec2(-108.,-173.));
 if(rl<0.){alb=vec3(.036,.037,.04)*(.8+.4*vnoise(p*2.2));spec=.2;float sx=fract((p.x+195.)/6.2);if((sx<.03||sx>.97)&&abs(p.y+195.)>3.)alb=vec3(.5);}
 // wet surfaces in rain
 alb*=mix(1.,.62,uWet);spec=mix(spec,max(spec,.55),uWet);
 float sh=shadowF(vW,n);vec3 c=shade(vW,n,alb,sh);
 vec3 V=normalize(uEye-vW);if(spec>0.){vec3 h=normalize(uSunDir+V);c+=uSunCol*pow(max(dot(n,h),0.),90.)*spec*sh*.5;c+=skyCol(reflect(-V,n))*spec*.12*pow(1.-max(dot(n,V),0.),4.);
  if(uWet>0.)c+=pointLights(vW+vec3(0,0,-1.),reflect(-V,n))*spec*.15;}
 o=vec4(applyFog(c,vW),1.);}`;}
const VS_TREE=`#version 300 es
layout(location=0) in vec3 aPos;layout(location=1) in vec3 aNrm;layout(location=2) in float aPart;layout(location=3) in vec4 aInst;layout(location=4) in float aZ;
uniform mat4 uVP;uniform float uTime;uniform vec2 uWind;out vec3 vW;out vec3 vN;flat out float vP;out float vAO;flat out float vSeed;
void main(){float h=aInst.z,seed=aInst.w;float s=.75+.5*fract(seed*7.13);vec3 lp=aPos*vec3(h*s,h*s,h);
 float sway=sin(uTime*1.6+seed*40.)*.5+sin(uTime*2.7+seed*13.)*.25;lp.xy+=uWind*.004*sway*h*aPos.z*aPos.z*(.4+length(uWind)*.06);
 float a=seed*6.2832;mat2 R=mat2(cos(a),sin(a),-sin(a),cos(a));lp.xy=R*lp.xy;vec3 n=aNrm;n.xy=R*n.xy;
 vW=vec3(aInst.xy,aZ)+lp;vN=n;vP=aPart;vAO=aPos.z;vSeed=seed;gl_Position=uVP*vec4(vW,1.);}`;
const FS_TREE=`#version 300 es
${COMMON}
in vec3 vW;in vec3 vN;flat in float vP;in float vAO;flat in float vSeed;out vec4 o;
void main(){vec3 n=normalize(vN);if(!gl_FrontFacing)n=-n;vec3 alb;
 if(vP<.5)alb=vec3(.07,.05,.035);else{alb=mix(vec3(.022,.045,.018),vec3(.055,.085,.03),fract(vSeed*13.7))*(.75+.5*vnoise(vW.xy*1.3+vW.z));
  if(vP>1.5)alb=mix(vec3(.04,.065,.022),vec3(.09,.10,.035),fract(vSeed*5.3))*(.8+.4*vnoise(vW.xy*.9+vW.z*1.3));}
 float ao=mix(.55,1.,clamp(vAO,0.,1.));float sh=shadowF(vW,n);vec3 c=shade(vW,n,alb*ao,sh);
 c+=alb*uSunCol*pow(max(dot(-n,uSunDir),0.),2.)*.25*sh;o=vec4(applyFog(c,vW),1.);}`;
const VS_FENCE=`#version 300 es
layout(location=0) in vec3 aPos;layout(location=1) in vec3 aNrm;layout(location=2) in vec3 aUV;uniform mat4 uVP;out vec3 vW;out vec3 vN;out vec3 vUV;
void main(){vW=aPos;vN=aNrm;vUV=aUV;gl_Position=uVP*vec4(aPos,1.);}`;
const FS_FENCE=`#version 300 es
${COMMON}
in vec3 vW;in vec3 vN;in vec3 vUV;out vec4 o;
void main(){vec3 n=normalize(vN);if(!gl_FrontFacing)n=-n;float a;
 if(vUV.z<0.){vec2 q=vec2(fract(vUV.x/.55)-.5,(vUV.y-.42)/.42);float r=abs(length(q*vec2(1.,1.15))-.42);vec2 q2=vec2(fract(vUV.x/.55+.5)-.5,q.y);float r2=abs(length(q2*vec2(1.,1.15))-.42);
  float w=min(r,r2);float fw=fwidth(w)*1.5+1e-4;a=(1.-smoothstep(.035-fw,.035+fw,w));a=mix(a,.3,clamp(fw*5.,0.,1.));}
 else{vec2 g=vec2(vUV.x+vUV.y,vUV.x-vUV.y)/.075;vec2 f=abs(fract(g)-.5);float w=min(f.x,f.y);float fw=fwidth(w)*1.2+1e-4;float wire=1.-smoothstep(.07-fw,.07+fw,w);
  a=mix(wire,.2,clamp(fw*4.,0.,1.));if(vUV.y>vUV.z-.06||vUV.y<.06)a=1.;a*=.95;}
 if(a<.01)discard;vec3 alb=vec3(.33,.34,.35);float sh=shadowF(vW,n);vec3 c=shade(vW,n,alb,sh)+shade(vW,-n,alb,sh)*.3;
 vec3 V=normalize(uEye-vW);c+=uSunCol*pow(max(dot(n,normalize(uSunDir+V)),0.),30.)*.15*sh;o=vec4(applyFog(c,vW),a);}`;
const VS_SKY=`#version 300 es
out vec2 vN;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);vN=p*2.-1.;gl_Position=vec4(vN,.9999,1.);}`;
const FS_SKY=`#version 300 es
${COMMON}
in vec2 vN;out vec4 o;uniform vec3 uCR,uCU,uCF,uDisk;uniform vec2 uTan;
void main(){vec3 d=normalize(uCF+uCR*vN.x*uTan.x+uCU*vN.y*uTan.y);vec3 c=skyCol(d);float sd=dot(d,uSunDir);
 c+=uDisk*(smoothstep(.99982,.99992,sd)+.015*pow(max(sd,0.),600.))*(1.-uOvercast*.9);
 if(d.z>0.){vec3 q=d*420.;vec3 id=floor(q);float h=hash12(id.xy*.731+id.z*17.13);
  if(h>.9965){vec3 f=fract(q)-.5;c+=vec3(.8,.85,1.)*smoothstep(.38,0.,length(f))*uWts.z*(1.-uOvercast)*(.55+.45*sin(uTime*2.3+h*97.))*1.4*smoothstep(0.,.12,d.z);}
  float mw=exp(-pow(dot(d,normalize(vec3(.3,.8,.52))),2.)*18.)*fbm3(d.xy*14.);c+=vec3(.025,.026,.032)*mw*uWts.z*(1.-uOvercast);
  vec2 cp=d.xy/(d.z+.07)*.85+vec2(uTime*.0035,uTime*.001);float cn=fbm(cp*1.5);float cov=smoothstep(.55-.35*uOvercast,.92,cn+.25*uOvercast)*smoothstep(0.,.1,d.z);
  vec3 cl=vec3(1.45,1.47,1.5)*uWts.x*(.6+.4*fbm3(cp*4.))+vec3(1.25,.58,.36)*uWts.y*(.7+.5*fbm3(cp*3.))+vec3(.022,.026,.034)*uWts.z;
  cl*=mix(1.,.62,uOvercast);cl+=uSunCol*.05*pow(max(sd,0.),8.)*(1.-cov);c=mix(c,cl,cov*.88);}
 o=vec4(c,1.);}`;
const VS_PTS=`#version 300 es
layout(location=0) in vec3 aPos;layout(location=1) in vec4 aCol;uniform mat4 uVP;uniform float uPxM;uniform vec3 uEye;uniform float uFogDist;out vec3 vC;
void main(){gl_Position=uVP*vec4(aPos,1.);float w=max(gl_Position.w,.1);gl_PointSize=clamp(aCol.a*uPxM/w,1.6,140.);float sc=aCol.a*uPxM/w;
 vC=aCol.rgb*exp(-length(aPos-uEye)/uFogDist*.9)*min(1.,sc/1.6);}`;
const FS_PTS=`#version 300 es
precision highp float;in vec3 vC;out vec4 o;void main(){vec2 q=gl_PointCoord*2.-1.;float r2=dot(q,q);if(r2>1.)discard;float a=exp(-r2*6.)*.7+exp(-r2*70.)*1.6;o=vec4(vC*a,1.);}`;
const VS_RAIN=`#version 300 es
uniform mat4 uVP;uniform vec3 uEye;uniform float uTime;uniform vec2 uWind;out float vA;
float hash11(float p){p=fract(p*.1031);p*=p+33.33;p*=p+p;return fract(p);}
void main(){int id=gl_VertexID/2;int e=gl_VertexID-id*2;float fi=float(id);vec3 h=vec3(hash11(fi*1.71),hash11(fi*3.13+5.),hash11(fi*7.37+1.));
 float R=mix(6.,140.,h.x*h.x);float a=h.y*6.2832;vec3 p=uEye+vec3(cos(a)*R,sin(a)*R,0.);p.z=uEye.z-50.+mod(h.z*100.-uTime*13.,100.);
 vec3 dv=normalize(vec3(uWind*.07,-1.));vec3 q=p+dv*float(e)*1.4;vA=.25+.75*float(e);gl_Position=uVP*vec4(q,1.);}`;
const FS_RAIN=`#version 300 es
precision highp float;in float vA;uniform vec3 uRainCol;out vec4 o;void main(){o=vec4(uRainCol*vA,.32*vA);}`;
const VS_SHADOW=`#version 300 es
layout(location=0) in vec3 aPos;uniform mat4 uLVP;void main(){gl_Position=uLVP*vec4(aPos,1.);}`;
const VS_SHADOW_TREE=`#version 300 es
layout(location=0) in vec3 aPos;layout(location=3) in vec4 aInst;layout(location=4) in float aZ;uniform mat4 uLVP;
void main(){float h=aInst.z,seed=aInst.w;float s=.75+.5*fract(seed*7.13);vec3 lp=aPos*vec3(h*s,h*s,h);float a=seed*6.2832;lp.xy=mat2(cos(a),sin(a),-sin(a),cos(a))*lp.xy;gl_Position=uLVP*vec4(vec3(aInst.xy,aZ)+lp,1.);}`;
const FS_EMPTY=`#version 300 es
precision mediump float;out vec4 o;void main(){o=vec4(1);}`;
const VS_QUAD=`#version 300 es
out vec2 vUV;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);vUV=p;gl_Position=vec4(p*2.-1.,0.,1.);}`;
const FS_BRIGHT=`#version 300 es
precision highp float;in vec2 vUV;uniform sampler2D uTex;uniform vec2 uTexel;uniform float uThr;out vec4 o;
void main(){vec3 c=vec3(0);for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)c+=texture(uTex,vUV+vec2(x,y)*uTexel*1.5).rgb;c/=9.;float l=dot(c,vec3(.2126,.7152,.0722));o=vec4(c*max(l-uThr,0.)/max(l,1e-4),1.);}`;
const FS_BLUR=`#version 300 es
precision highp float;in vec2 vUV;uniform sampler2D uTex;uniform vec2 uDir;out vec4 o;
void main(){const float w[5]=float[](.227027,.1945946,.1216216,.054054,.016216);vec3 c=texture(uTex,vUV).rgb*w[0];for(int i=1;i<5;i++){c+=texture(uTex,vUV+uDir*float(i)).rgb*w[i];c+=texture(uTex,vUV-uDir*float(i)).rgb*w[i];}o=vec4(c,1.);}`;
const FS_COMP=`#version 300 es
precision highp float;in vec2 vUV;uniform sampler2D uScene,uBloom;uniform float uExposure,uBloomK,uNight;out vec4 o;
vec3 aces(vec3 x){return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);}
float h(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
void main(){vec3 c=texture(uScene,vUV).rgb*uExposure+texture(uBloom,vUV).rgb*uExposure*uBloomK;float lum=dot(c,vec3(.2126,.7152,.0722));
 float sc=uNight*.75*(1.-smoothstep(.08,.6,lum));c=mix(c,lum*vec3(.78,.9,1.15),sc);c=aces(c);c=pow(c,vec3(1./2.2));
 vec2 v=vUV-.5;c*=1.-.28*smoothstep(.25,.85,length(v*vec2(1.25,1.)));c+=(h(gl_FragCoord.xy)-.5)/255.;o=vec4(c,1.);}`;

function compile(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)){const log=gl.getShaderInfoLog(s);throw new Error("shader: "+log+"\n"+src.split("\n").slice(0,400).map((l,i)=>(i+1)+": "+l).join("\n").slice(0,6000))}return s}
function program(vs,fs){const p=gl.createProgram();gl.attachShader(p,compile(gl.VERTEX_SHADER,vs));gl.attachShader(p,compile(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error("link: "+gl.getProgramInfoLog(p));const u={};const n=gl.getProgramParameter(p,gl.ACTIVE_UNIFORMS);
  for(let i=0;i<n;i++){const inf=gl.getActiveUniform(p,i);u[inf.name.replace(/\[0\]$/,"")]=gl.getUniformLocation(p,inf.name)}return{p,u}}

// ---------------- geometry ----------------
function vaoFor(builderOrArr,layout,usage){const vao=gl.createVertexArray();gl.bindVertexArray(vao);const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
  if(builderOrArr)gl.bufferData(gl.ARRAY_BUFFER,builderOrArr,usage||gl.STATIC_DRAW);const stride=layout.reduce((s,l)=>s+l[1],0)*4;let off=0;
  for(const[loc,size]of layout){gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,stride,off);off+=size*4}gl.bindVertexArray(null);return{vao,buf:b,stride}}
const SOLID_LAYOUT=[[0,3],[1,3],[2,3],[3,1]];
function buildGround(){const N=240,E=15000,pos=[];const g=u=>Math.sign(u)*Math.pow(Math.abs(u),1.9)*E;
  for(let j=0;j<=N;j++)for(let i=0;i<=N;i++){const x=g(i/N*2-1),y=g(j/N*2-1)+50;pos.push([x,y,hgt(x,y)])}
  const V=new Float32Array((N+1)*(N+1)*6);for(let j=0;j<=N;j++)for(let i=0;i<=N;i++){const k=j*(N+1)+i,p=pos[k];const e=Math.max(2,Math.abs(p[0])*0.004+Math.abs(p[1])*0.004);
    const nx=hgt(p[0]-e,p[1])-hgt(p[0]+e,p[1]),ny=hgt(p[0],p[1]-e)-hgt(p[0],p[1]+e),nz=2*e,l=Math.hypot(nx,ny,nz);V.set([p[0],p[1],p[2],nx/l,ny/l,nz/l],k*6)}
  const I=new Uint32Array(N*N*6);let q=0;for(let j=0;j<N;j++)for(let i=0;i<N;i++){const a=j*(N+1)+i,b=a+1,c=a+N+1,d=c+1;I.set([a,b,d,a,d,c],q);q+=6}
  const o=vaoFor(V,[[0,3],[1,3]]);gl.bindVertexArray(o.vao);const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,I,gl.STATIC_DRAW);gl.bindVertexArray(null);o.count=I.length;return o}
const HOUSES=[];{const r=mulberry32(901);for(let i=0;i<22;i++){const side=r()<0.5;let x,y,yaw;
  if(i<14){x=-2200+i*330+r()*120;y=-1150+(side?1:-1)*(28+r()*25);yaw=0;if(Math.abs(x)<60)continue}else{x=1350+(side?1:-1)*(28+r()*20);y=-800+(i-14)*290+r()*80;yaw=Math.PI/2}
  HOUSES.push({x,y,yaw,w:9+r()*5,d:8+r()*4,h:4.5+r()*2.5,col:["c9c2b2","a9b0b4","d8d2c4","8e8577","b7a58c"][i%5],roof:["3a3330","2c3036","4a3a32"][i%3]})}}
function buildStatic(){const B=new Builder(10);
  const concrete=lin("8c8a84"),metal=lin("6d7175"),dark=lin("2b2e31"),sod=[1.0,0.58,0.22];
  for(const bd of BUILDINGS){S_aabb(B,bd.x,bd.y,0,bd.x+bd.w,bd.y+bd.h,bd.z,concrete,1);
    // roof surface (separate, gravel) + parapet
    S_quad(B,[bd.x+.5,bd.y+.5,bd.z+.02],[bd.x+bd.w-.5,bd.y+.5,bd.z+.02],[bd.x+bd.w-.5,bd.y+bd.h-.5,bd.z+.02],[bd.x+.5,bd.y+bd.h-.5,bd.z+.02],concrete,2);
    for(const[a,b]of[[[bd.x,bd.y],[bd.x+bd.w,bd.y]],[[bd.x+bd.w,bd.y],[bd.x+bd.w,bd.y+bd.h]],[[bd.x+bd.w,bd.y+bd.h],[bd.x,bd.y+bd.h]],[[bd.x,bd.y+bd.h],[bd.x,bd.y]]])S_seg(B,[a[0],a[1],bd.z+.4],[b[0],b[1],bd.z+.4],.3,lin("7d7b75"),5);
    for(let i=0;i<3;i++){const x=bd.x+bd.w*(0.2+0.3*i),y=bd.y+bd.h*0.5;S_aabb(B,x-3,y-2.5,bd.z,x+3,y+2.5,bd.z+2.2,lin("9ba2a6"),5);S_cyl(B,x+4.5,y,bd.z,bd.z+1.2,0.9,10,lin("a4abaf"),5,true)}
    S_aabb(B,bd.x+bd.w*0.45,bd.y-2,0,bd.x+bd.w*0.55,bd.y,3.2,lin("3d4247"),5)}
  for(const[x,y]of TOWERS){S_aabb(B,x-1.2,y-1.2,0,x+1.2,y+1.2,14,lin("85837d"),5);S_aabb(B,x-3,y-3,14,x+3,y+3,17,dark,4);S_aabb(B,x-3.6,y-3.6,17,x+3.6,y+3.6,17.5,lin("55585b"),5);
    S_aabb(B,x-3.4,y-3.4,13.6,x+3.4,y+3.4,14,lin("55585b"),5);S_box(B,x,y,17.9,0.4,0.6,0.3,Math.atan2(-x,-y),0,sod,3)}
  // fence posts (two fences)
  for(const off of[0,4]){const x0=FENCE.x0-off,x1=FENCE.x1+off,y0=FENCE.y0-off,y1=FENCE.y1+off,h=off?4.5:6;
    for(const[a,b]of[[[x0,y0],[x1,y0]],[[x1,y0],[x1,y1]],[[x1,y1],[x0,y1]],[[x0,y1],[x0,y0]]]){const L=Math.hypot(b[0]-a[0],b[1]-a[1]),n=Math.round(L/4);
      for(let i=0;i<n;i++){const k=i/n,x=a[0]+(b[0]-a[0])*k,y=a[1]+(b[1]-a[1])*k;S_aabb(B,x-.05,y-.05,0,x+.05,y+.05,h+(off?0:.9),metal,5)}}}
  // light poles with sodium heads
  for(const L of LIGHTS){S_aabb(B,L[0]-.13,L[1]-.13,0,L[0]+.13,L[1]+.13,L[2],lin("4c5054"),5);S_aabb(B,L[0]-.45,L[1]-.25,L[2]-.1,L[0]+.45,L[1]+.25,L[2]+.15,lin("2d2f31"),5);S_aabb(B,L[0]-.35,L[1]-.18,L[2]-.16,L[0]+.35,L[1]+.18,L[2]-.1,sod,3)}
  for(const L of ROADLAMPS){S_aabb(B,L[0]-.1,L[1]-.1,0,L[0]+.1,L[1]+.1,L[2],lin("4c5054"),5);const dy=L[1]<-1000?-1.6:0,dx=L[1]<-1000?0:1.6;S_seg(B,[L[0],L[1],L[2]],[L[0]+dx,L[1]+dy,L[2]+.2],.06,lin("4c5054"),5);S_aabb(B,L[0]+dx-.3,L[1]+dy-.15,L[2]+.05,L[0]+dx+.3,L[1]+dy+.15,L[2]+.15,sod,3)}
  // sensors: radar lattice tower, RF masts, acoustic posts, PTZ pole
  {const R=RADAR;for(const[dx,dy]of[[-1,-1],[1,-1],[1,1],[-1,1]])S_seg(B,[R.x+dx*2.2,R.y+dy*2.2,0],[R.x+dx*.7,R.y+dy*.7,R.z-1],.12,lin("9aa0a4"),5);
    for(let z=3;z<R.z-1;z+=3.2){const k=1-z/R.z*0.68;S_aabb(B,R.x-2.2*k,R.y-2.2*k,z,R.x+2.2*k,R.y+2.2*k,z+.12,lin("9aa0a4"),5)}S_aabb(B,R.x-1.6,R.y-1.6,R.z-1,R.x+1.6,R.y+1.6,R.z-.6,lin("7a8085"),5)}
  for(const s of RFS){S_cyl(B,s.x,s.y,0,s.z,.18,8,lin("9aa0a4"),5,true);for(let i=0;i<4;i++){const a=i*Math.PI/2;S_aabb(B,s.x+Math.cos(a)*.6-.05,s.y+Math.sin(a)*.6-.05,s.z-1.5,s.x+Math.cos(a)*.6+.05,s.y+Math.sin(a)*.6+.05,s.z+.6,lin("e4e6e8"),5)}S_aabb(B,s.x-.7,s.y-.7,s.z-1.6,s.x+.7,s.y+.7,s.z-1.5,lin("9aa0a4"),5)}
  for(const s of ACS){S_cyl(B,s.x,s.y,0,s.z,.07,6,lin("9aa0a4"),5,true);S_cyl(B,s.x,s.y,s.z,s.z+.35,.18,10,lin("30343a"),5,true)}
  S_cyl(B,PTZ.x,PTZ.y,0,PTZ.z-.5,.25,10,lin("9aa0a4"),5,true);
  // parked cars
  for(const pc of PARKED){const c=lin(pc[2]);S_aabb(B,pc[0]-2.2,pc[1]-.9,.35,pc[0]+2.2,pc[1]+.9,1.05,c,6);S_aabb(B,pc[0]-1.1,pc[1]-.8,1.05,pc[0]+1.0,pc[1]+.8,1.55,lin("15191c"),6);
    for(const sx of[-1.4,1.4])for(const sy of[-.85,.85])S_box(B,pc[0]+sx,pc[1]+sy,.33,.33,.12,.33,Math.PI/2,0,lin("0e0e0e"),0)}
  // houses along the public roads (neighbourhood context)
  for(const H of HOUSES){const T=xf(H.x,H.y,0,H.yaw,0);const hw=H.w/2,hd=H.d/2,z=H.h,c=lin(H.col),rc=lin(H.roof);
    const wall=[T(-hw,-hd,0),T(hw,-hd,0),T(hw,hd,0),T(-hw,hd,0),T(-hw,-hd,z),T(hw,-hd,z),T(hw,hd,z),T(-hw,hd,z)];
    S_quad(B,wall[0],wall[1],wall[5],wall[4],c,1);S_quad(B,wall[1],wall[2],wall[6],wall[5],c,1);S_quad(B,wall[2],wall[3],wall[7],wall[6],c,1);S_quad(B,wall[3],wall[0],wall[4],wall[7],c,1);
    const r0=T(-hw-.5,0,z+2.6),r1=T(hw+.5,0,z+2.6),e0=T(-hw-.5,-hd-.6,z-.3),e1=T(hw+.5,-hd-.6,z-.3),e2=T(hw+.5,hd+.6,z-.3),e3=T(-hw-.5,hd+.6,z-.3);
    S_quad(B,e0,e1,r1,r0,rc,0);S_quad(B,e3,e2,r1,r0,rc,0);S_tri(B,T(-hw,-hd,z),T(-hw,hd,z),T(-hw,0,z+2.6),c,5);S_tri(B,T(hw,-hd,z),T(hw,hd,z),T(hw,0,z+2.6),c,5)}
  STATIC=vaoFor(B.a.subarray(0,B.n),SOLID_LAYOUT);STATIC.count=B.count()}
function buildFence(){const A=[];const add=(a,b,c,d,n,uva,uvb,uvc,uvd)=>{A.push(...a,...n,...uva,...b,...n,...uvb,...c,...n,...uvc,...a,...n,...uva,...c,...n,...uvc,...d,...n,...uvd)};
  for(const off of[0,4]){const x0=FENCE.x0-off,x1=FENCE.x1+off,y0=FENCE.y0-off,y1=FENCE.y1+off,h=off?4.5:6;let s=0;
    for(const[a,b]of[[[x0,y0],[x1,y0]],[[x1,y0],[x1,y1]],[[x1,y1],[x0,y1]],[[x0,y1],[x0,y0]]]){const L=Math.hypot(b[0]-a[0],b[1]-a[1]);const n=[-(b[1]-a[1])/L,(b[0]-a[0])/L,0];
      add([a[0],a[1],0],[b[0],b[1],0],[b[0],b[1],h],[a[0],a[1],h],n,[s,0,h],[s+L,0,h],[s+L,h,h],[s,h,h]);
      if(!off){add([a[0],a[1],h],[b[0],b[1],h],[b[0],b[1],h+.85],[a[0],a[1],h+.85],n,[s,0,-1],[s+L,0,-1],[s+L,.85,-1],[s,.85,-1])}
      s+=L}}
  FENCEG=vaoFor(new Float32Array(A),[[0,3],[1,3],[2,3]]);FENCEG.count=A.length/9}
function buildTrees(){
  // conifer: trunk + 3 stacked cones. broadleaf: trunk + 2 lumpy blobs. Unit height = 1.
  const mk=(type)=>{const V=[];const pushV=(p,n,part)=>V.push(p[0],p[1],p[2],n[0],n[1],n[2],part);const tri=(a,b,c,part,ns)=>{pushV(a,ns?ns[0]:[0,0,1],part);pushV(b,ns?ns[1]:[0,0,1],part);pushV(c,ns?ns[2]:[0,0,1],part)};
    const S=7,tr=type?0.04:0.03;for(let i=0;i<S;i++){const a0=i/S*6.283,a1=(i+1)/S*6.283;const n0=[Math.cos(a0),Math.sin(a0),0],n1=[Math.cos(a1),Math.sin(a1),0];
      tri([n0[0]*tr,n0[1]*tr,0],[n1[0]*tr,n1[1]*tr,0],[n1[0]*tr,n1[1]*tr,0.45],0,[n0,n1,n1]);tri([n0[0]*tr,n0[1]*tr,0],[n1[0]*tr,n1[1]*tr,0.45],[n0[0]*tr,n0[1]*tr,0.45],0,[n0,n1,n0])}
    if(!type){for(const[zb,r,zt]of[[0.18,0.34,0.62],[0.40,0.27,0.84],[0.60,0.18,1.0]]){const N=9;for(let i=0;i<N;i++){const a0=i/N*6.283,a1=(i+1)/N*6.283,am=(a0+a1)/2,sl=r/(zt-zb);
        const nn=a=>{const l=Math.hypot(1,sl);return[Math.cos(a)/l,Math.sin(a)/l,sl/l]};tri([Math.cos(a0)*r,Math.sin(a0)*r,zb],[Math.cos(a1)*r,Math.sin(a1)*r,zb],[0,0,zt],1,[nn(a0),nn(a1),nn(am)]);
        tri([Math.cos(a0)*r,Math.sin(a0)*r,zb],[0,0,zb+.04],[Math.cos(a1)*r,Math.sin(a1)*r,zb],1,[[0,0,-1],[0,0,-1],[0,0,-1]])}}}
    else{const blob=(cx,cy,cz,rx,rz,seed)=>{const Nu=8,Nv=5;const P=(u,v)=>{const th=u/Nu*6.283,ph=v/Nv*Math.PI;const j=1+0.18*Math.sin(th*3+seed)*Math.sin(ph*2+seed*2);
          const n=[Math.sin(ph)*Math.cos(th),Math.sin(ph)*Math.sin(th),Math.cos(ph)];return[[cx+n[0]*rx*j,cy+n[1]*rx*j,cz+n[2]*rz*j],n]};
        for(let u=0;u<Nu;u++)for(let v=0;v<Nv;v++){const a=P(u,v),b=P(u+1,v),c=P(u+1,v+1),d=P(u,v+1);tri(a[0],b[0],c[0],2,[a[1],b[1],c[1]]);tri(a[0],c[0],d[0],2,[a[1],c[1],d[1]])}};
      blob(0,0,0.62,0.34,0.3,1.3);blob(0.12,-0.06,0.8,0.24,0.2,4.1);blob(-0.13,0.08,0.5,0.22,0.18,2.2)}
    return new Float32Array(V)};
  const R=mulberry32(5150);const inst=[[],[]],zz=[[],[]];
  const addT=(x,y,h)=>{const s=R();const t=s<0.55?0:1;inst[t].push(x,y,h*(t?0.85:1.15),R());zz[t].push(hgt(x,y)-0.2)};
  for(const t of TREES)addT(t[0],t[1],t[2]*1.1);
  // windbreak rows along fields, forest on the hills
  for(const f of FIELDS){if(R()<0.5)continue;const a=f.q[0],b=f.q[1];const L=Math.hypot(b[0]-a[0],b[1]-a[1]);for(let s=0;s<L;s+=9+R()*6)addT(a[0]+(b[0]-a[0])*s/L+R()*3,a[1]+(b[1]-a[1])*s/L+R()*3,10+R()*8)}
  for(let i=0;i<2600;i++){const a=R()*6.283,r=2900+Math.pow(R(),0.7)*7500,x=Math.cos(a)*r*1.1,y=Math.sin(a)*r+50;if(fbm2(x*0.0009,y*0.0009)<0.48)continue;addT(x,y,14+R()*12)}
  TREEG=[0,1].map(t=>{const geo=mk(t);const o=vaoFor(geo,[[0,3],[1,3],[2,1]]);o.count=geo.length/7;gl.bindVertexArray(o.vao);
    const ib=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,ib);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(inst[t]),gl.STATIC_DRAW);gl.enableVertexAttribArray(3);gl.vertexAttribPointer(3,4,gl.FLOAT,false,16,0);gl.vertexAttribDivisor(3,1);
    const zb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,zb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(zz[t]),gl.STATIC_DRAW);gl.enableVertexAttribArray(4);gl.vertexAttribPointer(4,1,gl.FLOAT,false,4,0);gl.vertexAttribDivisor(4,1);
    gl.bindVertexArray(null);o.inst=inst[t].length/4;return o})}

// ---------------- render targets ----------------
function makeTex(w,h,fmt,filter){const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);const f=fmt==="hdr"&&HDRFMT?HDRFMT:{i:gl.RGBA8,f:gl.RGBA,t:gl.UNSIGNED_BYTE};
  gl.texImage2D(gl.TEXTURE_2D,0,f.i,w,h,0,f.f,f.t,null);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,filter||gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,filter||gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);return t}
function fbo(tex){const f=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,f);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);return f}
function freeTargets(){for(const k of Object.keys(TG)){const v=TG[k];if(!v)continue;if(v instanceof WebGLTexture)gl.deleteTexture(v);else if(v instanceof WebGLFramebuffer)gl.deleteFramebuffer(v);else if(v instanceof WebGLRenderbuffer)gl.deleteRenderbuffer(v)}for(const k of Object.keys(TG))delete TG[k]}
function setupTargets(w,h){freeTargets();SW=w;SH=h;const fmt=HDRFMT?HDRFMT.i:gl.RGBA8;
  TG.sceneTex=makeTex(w,h,"hdr");TG.sceneFB=fbo(TG.sceneTex);TG.depthRB=gl.createRenderbuffer();gl.bindRenderbuffer(gl.RENDERBUFFER,TG.depthRB);gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT24,w,h);gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,TG.depthRB);
  if(MSAA>1){TG.msFB=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,TG.msFB);TG.msColor=gl.createRenderbuffer();gl.bindRenderbuffer(gl.RENDERBUFFER,TG.msColor);gl.renderbufferStorageMultisample(gl.RENDERBUFFER,MSAA,fmt,w,h);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.RENDERBUFFER,TG.msColor);TG.msDepth=gl.createRenderbuffer();gl.bindRenderbuffer(gl.RENDERBUFFER,TG.msDepth);gl.renderbufferStorageMultisample(gl.RENDERBUFFER,MSAA,gl.DEPTH_COMPONENT24,w,h);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,TG.msDepth);if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE){MSAA=0}}
  const bw=Math.max(1,w>>2),bh=Math.max(1,h>>2);TG.bw=bw;TG.bh=bh;TG.bA=makeTex(bw,bh,"hdr");TG.bAF=fbo(TG.bA);TG.bB=makeTex(bw,bh,"hdr");TG.bBF=fbo(TG.bB);gl.bindFramebuffer(gl.FRAMEBUFFER,null)}
function setupShadow(){shadowTex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,shadowTex);gl.texStorage2D(gl.TEXTURE_2D,1,gl.DEPTH_COMPONENT24,SHADOW,SHADOW);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_COMPARE_MODE,gl.COMPARE_REF_TO_TEXTURE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_COMPARE_FUNC,gl.LEQUAL);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  shadowFB=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,shadowFB);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.TEXTURE_2D,shadowTex,0);gl.drawBuffers([gl.NONE]);gl.readBuffer(gl.NONE);gl.bindFramebuffer(gl.FRAMEBUFFER,null)}

// ---------------- matrices ----------------
function mul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s}return o}
function viewM(eye,r,u,f){const d=(v)=>v[0]*eye[0]+v[1]*eye[1]+v[2]*eye[2];return new Float32Array([r[0],u[0],f[0],0,r[1],u[1],f[1],0,r[2],u[2],f[2],0,-d(r),-d(u),-d(f),1])}
function projM(foc,w,h,n,f){const m=new Float32Array(16);m[0]=2*foc/w;m[5]=2*foc/h;m[10]=(f+n)/(f-n);m[14]=-2*f*n/(f-n);m[11]=1;return m}
function orthoM(R,n,f){const m=new Float32Array(16);m[0]=1/R;m[5]=1/R;m[10]=2/(f-n);m[14]=-(f+n)/(f-n);m[15]=1;return m}
const norm=v=>{const l=Math.hypot(...v)||1;return v.map(x=>x/l)};
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];

// ---------------- init ----------------
api.init=function(canvas){cv=canvas;try{
  gl=cv.getContext("webgl2",{antialias:false,alpha:false,depth:true,premultipliedAlpha:false,powerPreference:"high-performance",preserveDrawingBuffer:!!window.__glPreserve});
  if(!gl)throw new Error("WebGL2 unavailable");
  cv.addEventListener("webglcontextlost",e=>{e.preventDefault();api.ok=false;api.err="WebGL context lost; using canvas renderer"});
  if(gl.getExtension("EXT_color_buffer_float")){HDRFMT={i:gl.RGBA16F,f:gl.RGBA,t:gl.HALF_FLOAT}}else if(gl.getExtension("EXT_color_buffer_half_float")){HDRFMT={i:gl.RGBA16F,f:gl.RGBA,t:gl.HALF_FLOAT}}
  MSAA=Math.min(4,gl.getParameter(gl.MAX_SAMPLES)||0);
  PR.solid=program(VS_SOLID,FS_SOLID);PR.ground=program(VS_GROUND,FS_GROUND());PR.tree=program(VS_TREE,FS_TREE);PR.fence=program(VS_FENCE,FS_FENCE);
  PR.sky=program(VS_SKY,FS_SKY);PR.pts=program(VS_PTS,FS_PTS);PR.rain=program(VS_RAIN,FS_RAIN);PR.sh=program(VS_SHADOW,FS_EMPTY);PR.shTree=program(VS_SHADOW_TREE,FS_EMPTY);
  PR.bright=program(VS_QUAD,FS_BRIGHT);PR.blur=program(VS_QUAD,FS_BLUR);PR.comp=program(VS_QUAD,FS_COMP);
  GROUND=buildGround();buildStatic();buildFence();buildTrees();setupShadow();
  BUF.dyn=vaoFor(null,SOLID_LAYOUT,gl.DYNAMIC_DRAW);BUF.dynT=vaoFor(null,SOLID_LAYOUT,gl.DYNAMIC_DRAW);BUF.pts=vaoFor(null,[[0,3],[1,4]],gl.DYNAMIC_DRAW);
  BUF.empty=gl.createVertexArray();
  api.ok=true;inited=true}catch(e){api.ok=false;api.err=String(e&&e.message||e);console.warn("GL3D disabled:",api.err)}
  return api.ok};

// ---------------- per-frame dynamic geometry ----------------
const DB=new Builder(10),DT=new Builder(10);let PTS=[];
function sprite(p,c,size){PTS.push(p[0],p[1],p[2],c[0],c[1],c[2],size)}
function buildDynamic(t,E){DB.n=0;DT.n=0;PTS=[];const k=v3.icon;const lights=E.lights;
  // sensor moving parts
  {const R=RADAR,a=t*2.4;S_box(DB,R.x,R.y,R.z+1.0,2.6,0.18,0.75,a,0.12,lin("dfe3e6"),5);S_aabb(DB,R.x-.25,R.y-.25,R.z-.6,R.x+.25,R.y+.25,R.z+.4,lin("8e9499"),5);
   sprite([R.x,R.y,R.z+2.2],[40,2,2],(t%1.5)<0.75?0.9:0.0001)}
  {const cm=W.cam;S_box(DB,PTZ.x,PTZ.y,PTZ.z,0.28,0.55,0.3,cm.az*D2R,cm.el*D2R,lin("e9ecee"),5);S_box(DB,PTZ.x,PTZ.y,PTZ.z-0.45,0.2,0.2,0.2,cm.az*D2R,0,lin("4a4e52"),5);
   const f=[Math.sin(cm.az*D2R)*Math.cos(cm.el*D2R),Math.cos(cm.az*D2R)*Math.cos(cm.el*D2R),Math.sin(cm.el*D2R)];if(lights>0.1)sprite([PTZ.x+f[0]*0.6,PTZ.y+f[1]*0.6,PTZ.z+f[2]*0.6],[6,0.4,0.4],0.25)}
  // lamp glows + tower cabins
  if(lights>0.02){for(const L of LIGHTS){sprite([L[0],L[1],L[2]-0.25],[30*lights,17*lights,6*lights],1.1);sprite([L[0],L[1],L[2]-0.3],[1.3*lights,.7*lights,.25*lights],9)}
    for(const L of ROADLAMPS){const dy=L[1]<-1000?-1.6:0,dx=L[1]<-1000?0:1.6;sprite([L[0]+dx,L[1]+dy,L[2]],[26*lights,14*lights,5*lights],0.9);sprite([L[0]+dx,L[1]+dy,L[2]],[1.0*lights,.55*lights,.2*lights],7)}
    for(const[x,y]of TOWERS)sprite([x,y,17.9],[30*lights,24*lights,15*lights],1.0)}
  // aircraft warning light on the water tower-less horizon: none. Cars on the highway.
  for(const e of W.ents){if(e.type!=="car")continue;const dir=Math.sign(e.vel[0])||1,yaw=dir>0?Math.PI/2:-Math.PI/2,c=lin(["6b1f1f","233a55","9aa1a6","2c3034","c8c8c0"][Math.abs(Math.round(e.pos[1]*7+e.rcs))%5]);
    S_box(DB,e.pos[0],e.pos[1],0.72,0.9,2.25,0.38,yaw,0,c,6);S_box(DB,e.pos[0]-dir*0.2,e.pos[1],1.3,0.82,1.1,0.27,yaw,0,lin("15191c"),6);
    const hl=Math.max(lights,0.15);sprite([e.pos[0]+dir*2.3,e.pos[1]-0.6,0.75],[30*hl,28*hl,22*hl],0.5);sprite([e.pos[0]+dir*2.3,e.pos[1]+0.6,0.75],[30*hl,28*hl,22*hl],0.5);
    sprite([e.pos[0]-dir*2.3,e.pos[1]-0.6,0.8],[12*hl,0.5,0.4],0.35);sprite([e.pos[0]-dir*2.3,e.pos[1]+0.6,0.8],[12*hl,0.5,0.4],0.35)}
  if(!UI.truth)return;
  for(const e of W.ents){if(!e.alive)continue;
    if(e.type==="drone"){if((e.phase==="wait"||e.phase==="done")&&dist3(e.pos,C3.eye)>2500)continue;const T=xf(e.pos[0],e.pos[1],e.pos[2]+0.12*k,e.yaw,0);const s=k;
      const body=lin("8f959b"),armc=lin("2a2e32"),mot=lin("2a2c2e");
      S_box(DB,e.pos[0],e.pos[1],e.pos[2]+0.12*k,0.08*s,0.13*s,0.04*s,e.yaw,0,body,6);S_box(DB,...T(0,0.02*s,0.045*s),0.06*s,0.08*s,0.012*s,e.yaw,0,lin("c3c8cc"),6);
      S_box(DB,...T(0,0.14*s,-0.035*s),0.025*s,0.025*s,0.025*s,e.yaw,0.3,lin("101214"),6);
      for(let i=0;i<4;i++){const a=Math.PI/4+i*Math.PI/2,ex=Math.sin(a)*e.span/2*s,ey=Math.cos(a)*e.span/2*s;const A=T(0,0,0),Bp=T(ex,ey,0.01*s);S_seg(DB,A,Bp,0.012*s,armc,5);
        S_cyl(DB,Bp[0],Bp[1],Bp[2]-0.01*s,Bp[2]+0.025*s,0.022*s,8,mot,5,true);
        // prop disc (translucent) + blade
        const pz=Bp[2]+0.03*s,R=0.12*s;for(let j=0;j<16;j++){const a0=j/16*6.283,a1=(j+1)/16*6.283;S_tri(DT,[Bp[0],Bp[1],pz],[Bp[0]+Math.cos(a0)*R,Bp[1]+Math.sin(a0)*R,pz],[Bp[0]+Math.cos(a1)*R,Bp[1]+Math.sin(a1)*R,pz],[0.55,0.58,0.6],0)}
        const ba=t*41+i;S_box(DB,Bp[0],Bp[1],pz+0.002*s,0.006*s,R,0.002*s,ba,0,lin("2a2c2e"),5);
        if(e.lights){const on=(t*2)%1<0.5;sprite([Bp[0],Bp[1],Bp[2]-0.01*s],i<2?[on?60:6,2,2]:[2,50,8],0.05*s)}}
      for(const[x,y]of[[-0.07,0.07],[0.07,0.07],[-0.07,-0.07],[0.07,-0.07]])S_seg(DB,T(x*s,y*s,-0.03*s),T(x*1.35*s,y*s,-0.11*s),0.006*s,lin("3a3d40"),5);
      if(e.hasPayload){const top=[e.pos[0],e.pos[1],e.pos[2]+0.08*k],bot=[e.pos[0]+Math.sin(t*1.3)*0.05*k,e.pos[1],e.pos[2]-0.75*k];S_seg(DB,top,bot,0.004*k,lin("c9b88a"),0);
        S_box(DB,bot[0],bot[1],bot[2]-0.07*k,0.09*k,0.07*k,0.07*k,e.yaw,0,lin("8a6a3c"),0);S_box(DB,bot[0],bot[1],bot[2]-0.07*k,0.092*k,0.015*k,0.072*k,e.yaw,0,lin("c7b48b"),0)}}
    else if(e.type==="bird"){const kb=k*0.75,sp=e.span*kb,ph=Math.sin(e.flap),fl=(e.glide>0&&!(e.hover>0)?0.08:ph)*0.36*sp;const T=xf(e.pos[0],e.pos[1],e.pos[2],e.h,0);
      const col=e.kind==="gull"?lin("e8ecee"):e.kind==="crow"?lin("17191c"):lin("8a5a32"),tip=e.kind==="gull"?lin("1a1c1e"):e.kind==="crow"?lin("0d0e10"):lin("4a2e18"),under=e.kind==="gull"?lin("cfd5d8"):col;
      S_box(DB,e.pos[0],e.pos[1],e.pos[2],0.06*sp,0.16*sp,0.055*sp,e.h,0,under,0);S_box(DB,...T(0,0.19*sp,0.02*sp),0.04*sp,0.05*sp,0.04*sp,e.h,0,col,0);
      S_box(DB,...T(0,-0.2*sp,0),0.05*sp,0.06*sp,0.008*sp,e.h,-0.15,col,0);
      for(const sd of[-1,1]){const r0=T(sd*0.05*sp,0.06*sp,0.02*sp),r1=T(sd*0.05*sp,-0.08*sp,0.02*sp),m0=T(sd*0.25*sp,0.05*sp,fl*0.45),m1=T(sd*0.25*sp,-0.07*sp,fl*0.45),t0=T(sd*0.5*sp,-0.02*sp,fl-0.04*sp),t1=T(sd*0.47*sp,-0.07*sp,fl-0.03*sp);
        S_quad(DB,r0,m0,m1,r1,col,0);S_quad(DB,m0,t0,t1,m1,tip,0)}}
    else if(e.type==="payload"){const s=k*0.6;S_box(DB,e.pos[0],e.pos[1],e.pos[2]+0.07*s,0.12*s,0.1*s,0.08*s,0.3,0,lin("8a6a3c"),0)}
    else if(e.type==="bag"){const s=k*0.5;S_box(DB,e.pos[0],e.pos[1],e.pos[2],0.25*s,0.2*s,0.05*s,t*0.7+e.ph,Math.sin(t+e.ph)*0.6,lin("dfe3e5"),0)}
    else if(e.type==="aircraft"&&!(e.wait>0)){const yaw=Math.atan2(e.vel[0],e.vel[1]);S_box(DB,e.pos[0],e.pos[1],e.pos[2],1.8,16,1.9,yaw,0,lin("c9cdd0"),6);S_box(DB,e.pos[0],e.pos[1],e.pos[2],15,2.4,0.3,yaw,0,lin("b7bbbe"),6);
      const T=xf(e.pos[0],e.pos[1],e.pos[2],yaw,0);sprite(T(-15,0,0),[60,3,3],3);sprite(T(15,0,0),[3,60,10],3);if((t%1.2)<0.1)sprite(T(0,-15,1),[250,250,250],7)}}
  for(const d of W.drones){const px=d.pilot[0],py=d.pilot[1],z=hgt(px,py),s=3;
    S_box(DB,px,py,z+0.95*s/2,0.22*s/2,0.14*s/2,0.62*s/2,0,0,lin("2b3440"),0);S_box(DB,px,py,z+0.95*s+0.12*s,0.1*s,0.1*s,0.12*s,0,0,lin("b08a6e"),0);
    S_box(DB,px+0.11*s*2,py+0.12*s,z+0.95*s*0.85,0.08*s,0.1*s,0.03*s,0.4,0.5,lin("111111"),0);
    S_box(DB,px+6,py+2.5,0.75,0.9,2.3,0.4,0.25,0,lin("1c1f22"),6);S_box(DB,px+6,py+2.5,1.35,0.85,1.2,0.3,0.25,0,lin("0f1113"),6)}}
function upload(o,B){gl.bindBuffer(gl.ARRAY_BUFFER,o.buf);gl.bufferData(gl.ARRAY_BUFFER,B.a.subarray(0,B.n),gl.DYNAMIC_DRAW);o.count=B.count()}

// ---------------- frame ----------------
function lightingState(E){const d=UI.tod,wN=Math.max(0,1-d/0.45),wD=Math.max(0,(d-0.45)/0.55),wK=Math.max(0,1-wN-wD);
  const sunDay=norm([Math.sin(215*D2R)*Math.cos(50*D2R),Math.cos(215*D2R)*Math.cos(50*D2R),Math.sin(50*D2R)]),sunDusk=norm([Math.sin(258*D2R)*Math.cos(5*D2R),Math.cos(258*D2R)*Math.cos(5*D2R),Math.sin(5*D2R)]),moon=norm([Math.sin(145*D2R)*Math.cos(38*D2R),Math.cos(145*D2R)*Math.cos(38*D2R),Math.sin(38*D2R)]);
  const L=norm(sunDay.map((v,i)=>v*wD+sunDusk[i]*wK+moon[i]*wN));const oc=UI.wx==="clear"?0:UI.wx==="fog"?0.9:0.8;
  const sun=[3.4*wD+1.9*wK+0.13*wN,3.2*wD+1.05*wK+0.17*wN,2.95*wD+0.55*wK+0.26*wN].map(v=>v*(1-oc*0.85));
  const amb=[0.33*wD+0.19*wK+0.010*wN,0.45*wD+0.18*wK+0.014*wN,0.66*wD+0.25*wK+0.030*wN].map(v=>v*(1+oc*0.5));
  const gnd=[0.11*wD+0.07*wK+0.006*wN,0.10*wD+0.05*wK+0.007*wN,0.08*wD+0.05*wK+0.01*wN];
  const disk=wN>0.5?[1.6,1.7,1.9]:[60,52,40].map(v=>v*(wD+wK*0.7));
  const exposure=(0.55*wD+1.25*wK+3.4*wN)*(1+oc*0.25);
  return{L,sun,amb,gnd,disk,exposure,w:[wD,wK,wN],oc}}
function setCommon(P,S,E,t,vp,lvp,shadowOn){const u=P.u;gl.useProgram(P.p);
  const set3=(n,v)=>{if(u[n])gl.uniform3fv(u[n],v)},set1=(n,v)=>{if(u[n])gl.uniform1f(u[n],v)};
  if(u.uVP)gl.uniformMatrix4fv(u.uVP,false,vp);set3("uEye",C3.eye);set3("uSunDir",S.L);set3("uSunCol",S.sun);set3("uAmbSky",S.amb);set3("uAmbGnd",S.gnd);set3("uWts",S.w);
  set1("uTime",t);set1("uLightsOn",E.lights);set1("uFogDist",UI.wx==="clear"?16000:E.fog);set1("uOvercast",S.oc);set1("uWet",UI.wx==="rain"?1:0);set1("uShadowOn",shadowOn?1:0);
  if(u.uWind)gl.uniform2fv(u.uWind,windVec());if(u.uLights){gl.uniform4fv(u.uLights,LIGHTU);gl.uniform1i(u.uNL,LIGHTU.length/4)}
  if(u.uShadowVP)gl.uniformMatrix4fv(u.uShadowVP,false,lvp);if(u.uShadow){gl.activeTexture(gl.TEXTURE5);gl.bindTexture(gl.TEXTURE_2D,shadowTex);gl.uniform1i(u.uShadow,5)}}
let LIGHTU=null;
function fieldUniforms(P){const A=[],B=[],C=[];for(const f of FIELDS.slice(0,48)){const q=f.q;const cx=(q[0][0]+q[2][0])/2,cy=(q[0][1]+q[2][1])/2;const ax=q[1][0]-q[0][0],ay=q[1][1]-q[0][1],L1=Math.hypot(ax,ay),L2=Math.hypot(q[3][0]-q[0][0],q[3][1]-q[0][1]);
  A.push(cx,cy,ax/L1,ay/L1);B.push(L1/2,L2/2,0,0);C.push(...lin(f.col).map(v=>v*0.55))}
  gl.uniform4fv(P.u.uFA,new Float32Array(A));gl.uniform4fv(P.u.uFB,new Float32Array(B));gl.uniform3fv(P.u.uFC,new Float32Array(C));gl.uniform1i(P.u.uNF,A.length/4)}
let Q=1,lastT=0,avgDt=16,nF=0;
function resize(){const dpr=Math.min(window.devicePixelRatio||1,1.5)*Q;const w=Math.max(2,Math.round(MW*dpr)),h=Math.max(2,Math.round(MH*dpr));
  if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h}if(SW!==w||SH!==h)setupTargets(w,h);return dpr}
api.lastVP=null;api.render=function(){if(!api.ok||!C3)return;const t0=performance.now();
  if(lastT&&t0-lastT<500){avgDt=avgDt*0.95+(t0-lastT)*0.05;if(++nF>90){if(avgDt>26&&Q>0.55){Q*=0.85;if(Q<0.8&&MSAA>1){MSAA=0;SW=0}nF=0}else if(avgDt<16.5&&Q<1){Q=Math.min(1,Q*1.08);nF=0}}}lastT=t0;api.quality=Q;const t=W.t,E=ENV();const dpr=resize();const S=lightingState(E);
  if(!LIGHTU){const a=[];for(const L of LIGHTS.slice(0,36))a.push(L[0],L[1],L[2]-0.3,520);for(const[x,y]of TOWERS)a.push(x,y,17.6,700);LIGHTU=new Float32Array(a.slice(0,160))}
  const vp=mul(projM(C3.foc,MW,MH,NEARP,FARP),viewM(C3.eye,C3.r,C3.u,C3.f));api.lastVP=vp;
  // shadow light frustum around what the camera is looking at
  const c0=[v3.target[0],v3.target[1],0],R=clamp(v3.dist*0.85,260,1500);const Ld=S.L;const le=[c0[0]+Ld[0]*4000,c0[1]+Ld[1]*4000,c0[2]+Ld[2]*4000];const lf=norm([-Ld[0],-Ld[1],-Ld[2]]);
  let lr=norm(cross(lf,[0,0,1]));if(!isFinite(lr[0]))lr=[1,0,0];const lu=cross(lr,lf);const lvp=mul(orthoM(R,100,8000),viewM(le,lr,lu,lf));
  const shadowOn=S.L[2]>0.05&&S.oc<0.7?1:0;
  buildDynamic(t,E);upload(BUF.dyn,DB);upload(BUF.dynT,DT);
  gl.bindBuffer(gl.ARRAY_BUFFER,BUF.pts.buf);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(PTS),gl.DYNAMIC_DRAW);BUF.pts.count=PTS.length/7;
  gl.disable(gl.CULL_FACE);
  // ---- shadow pass
  if(shadowOn){gl.bindFramebuffer(gl.FRAMEBUFFER,shadowFB);gl.viewport(0,0,SHADOW,SHADOW);gl.clear(gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.depthMask(true);gl.enable(gl.POLYGON_OFFSET_FILL);gl.polygonOffset(1.5,3);
    gl.useProgram(PR.sh.p);gl.uniformMatrix4fv(PR.sh.u.uLVP,false,lvp);for(const o of[STATIC,BUF.dyn]){if(!o.count)continue;gl.bindVertexArray(o.vao);gl.drawArrays(gl.TRIANGLES,0,o.count)}
    gl.useProgram(PR.shTree.p);gl.uniformMatrix4fv(PR.shTree.u.uLVP,false,lvp);for(const o of TREEG){gl.bindVertexArray(o.vao);gl.drawArraysInstanced(gl.TRIANGLES,0,o.count,o.inst)}gl.disable(gl.POLYGON_OFFSET_FILL)}
  // ---- scene pass (HDR, MSAA)
  gl.bindFramebuffer(gl.FRAMEBUFFER,MSAA>1?TG.msFB:TG.sceneFB);gl.viewport(0,0,SW,SH);gl.clearColor(0,0,0,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  // sky
  gl.disable(gl.DEPTH_TEST);gl.depthMask(false);setCommon(PR.sky,S,E,t,vp,lvp,0);gl.uniform3fv(PR.sky.u.uCR,C3.r);gl.uniform3fv(PR.sky.u.uCU,C3.u);gl.uniform3fv(PR.sky.u.uCF,C3.f);
  gl.uniform2f(PR.sky.u.uTan,MW/(2*C3.foc),MH/(2*C3.foc));gl.uniform3fv(PR.sky.u.uDisk,S.disk);gl.bindVertexArray(BUF.empty);gl.drawArrays(gl.TRIANGLES,0,3);
  gl.enable(gl.DEPTH_TEST);gl.depthMask(true);gl.depthFunc(gl.LESS);
  setCommon(PR.ground,S,E,t,vp,lvp,shadowOn);fieldUniforms(PR.ground);gl.bindVertexArray(GROUND.vao);gl.drawElements(gl.TRIANGLES,GROUND.count,gl.UNSIGNED_INT,0);
  setCommon(PR.solid,S,E,t,vp,lvp,shadowOn);gl.uniform1f(PR.solid.u.uAlpha,1);for(const o of[STATIC,BUF.dyn]){if(!o.count)continue;gl.bindVertexArray(o.vao);gl.drawArrays(gl.TRIANGLES,0,o.count)}
  setCommon(PR.tree,S,E,t,vp,lvp,shadowOn);for(const o of TREEG){gl.bindVertexArray(o.vao);gl.drawArraysInstanced(gl.TRIANGLES,0,o.count,o.inst)}
  // transparent: fence, prop discs, rain, glows
  gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
  setCommon(PR.fence,S,E,t,vp,lvp,shadowOn);gl.bindVertexArray(FENCEG.vao);gl.drawArrays(gl.TRIANGLES,0,FENCEG.count);
  if(BUF.dynT.count){setCommon(PR.solid,S,E,t,vp,lvp,shadowOn);gl.uniform1f(PR.solid.u.uAlpha,0.22);gl.bindVertexArray(BUF.dynT.vao);gl.drawArrays(gl.TRIANGLES,0,BUF.dynT.count)}
  if(UI.wx==="rain"){gl.blendFunc(gl.SRC_ALPHA,gl.ONE);gl.useProgram(PR.rain.p);const u=PR.rain.u;gl.uniformMatrix4fv(u.uVP,false,vp);gl.uniform3fv(u.uEye,C3.eye);gl.uniform1f(u.uTime,t);gl.uniform2fv(u.uWind,windVec());
    const rb=0.25*S.w[0]+0.12*S.w[1]+0.05*S.w[2]+0.05*E.lights;gl.uniform3f(u.uRainCol,rb,rb*1.05,rb*1.12);gl.bindVertexArray(BUF.empty);gl.drawArrays(gl.LINES,0,9000)}
  if(BUF.pts.count){gl.blendFunc(gl.ONE,gl.ONE);gl.useProgram(PR.pts.p);const u=PR.pts.u;gl.uniformMatrix4fv(u.uVP,false,vp);gl.uniform1f(u.uPxM,C3.foc*dpr);gl.uniform3fv(u.uEye,C3.eye);gl.uniform1f(u.uFogDist,UI.wx==="clear"?16000:E.fog);
    gl.bindVertexArray(BUF.pts.vao);gl.drawArrays(gl.POINTS,0,BUF.pts.count)}
  gl.disable(gl.BLEND);gl.depthMask(true);
  // ---- resolve, bloom, tonemap
  if(MSAA>1){gl.bindFramebuffer(gl.READ_FRAMEBUFFER,TG.msFB);gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,TG.sceneFB);gl.blitFramebuffer(0,0,SW,SH,0,0,SW,SH,gl.COLOR_BUFFER_BIT,gl.NEAREST)}
  gl.disable(gl.DEPTH_TEST);gl.bindVertexArray(BUF.empty);
  gl.bindFramebuffer(gl.FRAMEBUFFER,TG.bAF);gl.viewport(0,0,TG.bw,TG.bh);gl.useProgram(PR.bright.p);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,TG.sceneTex);gl.uniform1i(PR.bright.u.uTex,0);
  gl.uniform2f(PR.bright.u.uTexel,1/SW,1/SH);gl.uniform1f(PR.bright.u.uThr,1.0/S.exposure);gl.drawArrays(gl.TRIANGLES,0,3);
  for(let it=0;it<2;it++){gl.useProgram(PR.blur.p);gl.uniform1i(PR.blur.u.uTex,0);gl.bindFramebuffer(gl.FRAMEBUFFER,TG.bBF);gl.bindTexture(gl.TEXTURE_2D,TG.bA);gl.uniform2f(PR.blur.u.uDir,(1+it)/TG.bw,0);gl.drawArrays(gl.TRIANGLES,0,3);
    gl.bindFramebuffer(gl.FRAMEBUFFER,TG.bAF);gl.bindTexture(gl.TEXTURE_2D,TG.bB);gl.uniform2f(PR.blur.u.uDir,0,(1+it)/TG.bh);gl.drawArrays(gl.TRIANGLES,0,3)}
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,cv.width,cv.height);gl.useProgram(PR.comp.p);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,TG.sceneTex);gl.uniform1i(PR.comp.u.uScene,0);
  gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,TG.bA);gl.uniform1i(PR.comp.u.uBloom,1);gl.uniform1f(PR.comp.u.uExposure,S.exposure);gl.uniform1f(PR.comp.u.uBloomK,0.35+0.5*S.w[2]);gl.uniform1f(PR.comp.u.uNight,S.w[2]);gl.drawArrays(gl.TRIANGLES,0,3);
  gl.activeTexture(gl.TEXTURE0);gl.bindVertexArray(null);
  const dt=performance.now()-t0;api.frameMs=api.frameMs?api.frameMs*0.9+dt*0.1:dt};
api.finish=function(){if(!gl)return;const px=new Uint8Array(4);gl.readPixels(0,0,1,1,gl.RGBA,gl.UNSIGNED_BYTE,px)};
return api})();
