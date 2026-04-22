import{i as le,t as ie,r as T,g as ae}from"./styles-CDRzXZyU.js";import{f as ce,a as pe,c as de}from"./api-BlpZUjmm.js";function me(t,o){const s=t.split(/\r?\n/);if(s.length<3)return t;const r=parseInt(s[0].trim(),10);if(isNaN(r)||r<1)return t;const l=[s[0],s[1]];for(let a=2;a<2+r&&a<s.length;a++){const g=s[a].trim().split(/\s+/);if(g.length<4){l.push(s[a]);continue}const f=parseFloat(g[1])+(Math.random()-.5)*2*o,h=parseFloat(g[2])+(Math.random()-.5)*2*o,b=parseFloat(g[3])+(Math.random()-.5)*2*o;l.push(`${g[0]}  ${f.toFixed(6)}  ${h.toFixed(6)}  ${b.toFixed(6)}`)}return l.join(`
`)}const C=.529177249;function oe(t){const o=[],s=/--- Geometry Optimization Step (\d+) ---[\s\S]*?Energy:\s*([-+]?\d+\.\d+)\s+Hartree[\s\S]*?Max gradient:\s*([-+]?\d+\.\d+[eE][-+]?\d+)[\s\S]*?RMS gradient:\s*([-+]?\d+\.\d+[eE][-+]?\d+)/g;let r;for(;(r=s.exec(t))!==null;)o.push({step:parseInt(r[1]),energy:parseFloat(r[2]),maxGrad:parseFloat(r[3]),rmsGrad:parseFloat(r[4])});return o}function ne(t,o){const s=new Map,r=/\[Geometry Step (\d+)\]\s*\n([\s\S]*?)(?=\n\s*\n|\n---|\n\[|\nWARNING|\n=)/g;let l;for(;(l=r.exec(t))!==null;){const a=parseInt(l[1]),g=l[2].trim().split(`
`),f=[],h=[];for(const b of g){const m=b.trim().split(/\s+/);m.length>=7&&(f.push({element:m[0],x:parseFloat(m[1])*C,y:parseFloat(m[2])*C,z:parseFloat(m[3])*C}),h.push({x:parseFloat(m[4])*C,y:parseFloat(m[5])*C,z:parseFloat(m[6])*C}))}o>0&&f.length!==o||f.length>0&&s.set(a,{atoms:f,forces:h})}return s}function ue(t){const o=[],s=t.match(/Optimized Geometry \(Bohr\):\s*\n([\s\S]*?)(?:\n\s*\n|\n\[|$)/);if(!s)return o;const r=s[1].trim().split(`
`);for(const l of r){const a=l.trim().split(/\s+/);a.length>=4&&o.push({element:a[0],x:parseFloat(a[1])*C,y:parseFloat(a[2])*C,z:parseFloat(a[3])*C})}return o}function D(t){const o=[`${t.length}`,"geometry"];for(const s of t)o.push(`${s.element}  ${s.x.toFixed(6)}  ${s.y.toFixed(6)}  ${s.z.toFixed(6)}`);return o.join(`
`)}function j(t,o,s,r,l,a,g=!1,f){const h=ae(),b=500,m=260,x=72,A=16,u=20,z=36,v=b-x-A,$=m-u-z;if(o.length===0){t.innerHTML="";return}const G=Math.min(...o),X=Math.max(...o);let y=s;g&&(y=s.map(e=>e>0?Math.log10(e):-12));const E=Math.min(...y)-.05*(Math.max(...y)-Math.min(...y)||1),B=Math.max(...y)+.05*(Math.max(...y)-Math.min(...y)||1),R=e=>x+(e-G)/(X-G||1)*v,F=e=>u+$-(e-E)/(B-E||1)*$;let i=`<svg width="${b}" height="${m}" xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;max-width:${b}px;height:auto" viewBox="0 0 ${b} ${m}">`;for(let e=0;e<=4;e++){const c=E+(B-E)*e/4,M=F(c);i+=`<line x1="${x}" y1="${M}" x2="${x+v}" y2="${M}" stroke="${h.grid}" stroke-width="0.5"/>`;const J=g?`1e${c.toFixed(0)}`:c.toFixed(6);i+=`<text x="${x-4}" y="${M+3}" text-anchor="end" font-size="8" fill="${h.dim}">${J}</text>`}i+=`<line x1="${x}" y1="${u}" x2="${x}" y2="${u+$}" stroke="${h.axis}" stroke-width="1"/>`,i+=`<line x1="${x}" y1="${u+$}" x2="${x+v}" y2="${u+$}" stroke="${h.axis}" stroke-width="1"/>`;const U=Math.min(o.length,10),W=Math.max(1,Math.floor(o.length/U));for(let e=0;e<o.length;e+=W)i+=`<text x="${R(o[e])}" y="${u+$+14}" text-anchor="middle" font-size="8" fill="${h.dim}">${o[e]}</text>`;let q="";for(let e=0;e<o.length;e++){const c=R(o[e]),M=F(y[e]);q+=q?` L${c},${M}`:`M${c},${M}`}i+=`<path d="${q}" fill="none" stroke="${a}" stroke-width="2"/>`;for(let e=0;e<o.length;e++)i+=`<circle cx="${R(o[e])}" cy="${F(y[e])}" r="3" fill="${a}"/>`;if(f!==void 0&&f>=0&&f<o.length){const e=R(o[f]),c=F(y[f]);i+=`<line x1="${e}" y1="${u}" x2="${e}" y2="${u+$}" stroke="#ff6600" stroke-width="1" stroke-dasharray="4,3" opacity="0.6"/>`,i+=`<circle cx="${e}" cy="${c}" r="6" fill="none" stroke="#ff6600" stroke-width="2"/>`,i+=`<circle cx="${e}" cy="${c}" r="3" fill="#ff6600"/>`}i+=`<text x="${x+v/2}" y="${m-4}" text-anchor="middle" font-size="10" fill="${h.dim}">${r}</text>`,i+=`<text x="10" y="${u+$/2}" text-anchor="middle" font-size="10" fill="${h.dim}" transform="rotate(-90,10,${u+$/2})">${l}</text>`,i+="</svg>",t.innerHTML=i}function ge(t){return t.replace(/\x1b\[[0-9;]*m/g,"")}async function fe(){le();const t=document.getElementById("app");t.innerHTML=`
    <header class="header-top">
      <h1>GANSU</h1>
      <span class="subtitle">Geometry Optimization</span>
      <nav class="demo-nav">
        <a href="./" class="demo-tab">Calculation</a>
        <a href="./pes.html" class="demo-tab">PES</a>
        <a class="demo-tab active">Geometry Opt</a>
      </nav>
      <button id="theme-btn" class="icon-btn" title="Toggle theme"><span id="theme-icon">&#9790;</span></button>
    </header>

    <div class="geomopt-layout">
      <div class="geomopt-left">
        <div class="panel">
          <h2>Molecule</h2>
          <div class="pes-form">
            <label>Sample molecules</label>
            <select id="go-sample"><option value="">-- Select --</option></select>
            <label>XYZ</label>
            <textarea id="go-xyz" rows="8" style="font-family:monospace;font-size:0.75rem;width:100%;resize:vertical;background:var(--color-input);color:var(--color-text);border:1px solid var(--color-border-input);border-radius:5px;padding:6px"></textarea>
            <label>Distortion: <span id="go-distort-label">0.15 &#197;</span></label>
            <input type="range" id="go-distort-mag" min="1" max="50" value="15" style="width:100%" />
            <button id="go-distort" class="secondary-btn" style="margin-top:4px;width:100%">Distort</button>
          </div>
        </div>

        <div class="panel">
          <h2>Settings</h2>
          <div class="pes-form">
            <label>Method</label>
            <select id="go-method">
              <option value="RHF" selected>RHF</option>
              <option value="UHF">UHF</option>
            </select>
            <label>Basis</label>
            <select id="go-basis"></select>
            <label>Optimizer</label>
            <select id="go-optimizer">
              <option value="bfgs" selected>BFGS</option>
              <option value="dfp">DFP</option>
              <option value="sr1">SR1</option>
              <option value="gdiis">GDIIS</option>
              <option value="cg-pr">CG (Polak-Ribi&egrave;re)</option>
              <option value="cg-fr">CG (Fletcher-Reeves)</option>
              <option value="sd">Steepest Descent</option>
              <option value="newton">Newton</option>
            </select>
            <label>Initial guess</label>
            <select id="go-guess">
              <option value="sad" selected>SAD</option>
              <option value="core">Core</option>
              <option value="gwh">GWH</option>
            </select>
            <div class="pes-actions">
              <button id="go-run" class="primary-btn">Optimize</button>
              <button id="go-cancel" class="secondary-btn hidden">Cancel</button>
            </div>
            <div class="pes-progress" id="go-progress"></div>
          </div>
        </div>
      </div>

      <div class="geomopt-right">
        <div class="panel">
          <div id="go-mol-preview" style="width:100%;max-width:360px;aspect-ratio:1;margin:0 auto;"></div>
          <div class="pes-slider-row" id="go-slider-row" style="display:none">
            <input type="range" id="go-slider" min="0" max="0" value="0" />
            <span id="go-slider-label" class="pes-slider-label"></span>
          </div>
        </div>
        <div class="panel geomopt-plots">
          <div id="go-energy-plot"></div>
          <div id="go-gradient-plot"></div>
        </div>
        <div class="panel" id="go-result-panel" style="display:none">
          <h2>Result</h2>
          <pre id="go-result-text" style="font-size:0.75rem;overflow-x:auto;max-height:200px;background:var(--color-surface-alt);padding:8px;border-radius:6px"></pre>
        </div>
        <details class="panel" id="go-log-details">
          <summary>Raw Output</summary>
          <pre id="go-log" style="font-size:0.7rem;overflow-x:auto;max-height:300px;background:var(--color-surface-alt);padding:8px;border-radius:6px;white-space:pre-wrap"></pre>
        </details>
      </div>
    </div>
  `;const o=t.querySelector("#theme-btn"),s=t.querySelector("#theme-icon");o.addEventListener("click",()=>{const n=ie();s.textContent=n==="dark"?"☀":"☾"});const r=t.querySelector("#go-sample"),l=t.querySelector("#go-xyz"),a=t.querySelector("#go-distort"),g=t.querySelector("#go-distort-mag"),f=t.querySelector("#go-distort-label"),h=t.querySelector("#go-method"),b=t.querySelector("#go-basis"),m=t.querySelector("#go-optimizer"),x=t.querySelector("#go-guess"),A=t.querySelector("#go-run"),u=t.querySelector("#go-cancel"),z=t.querySelector("#go-progress"),v=t.querySelector("#go-mol-preview"),$=t.querySelector("#go-slider-row"),G=t.querySelector("#go-slider"),X=t.querySelector("#go-slider-label"),y=t.querySelector("#go-energy-plot"),E=t.querySelector("#go-gradient-plot"),B=t.querySelector("#go-result-panel"),R=t.querySelector("#go-result-text"),F=t.querySelector("#go-log");g.addEventListener("input",()=>{f.innerHTML=`${(parseInt(g.value)/100).toFixed(2)} &#197;`});let i="";a.addEventListener("click",()=>{const n=i||l.value.trim();i||(i=n);const p=parseInt(g.value)/100,d=me(n,p);l.value=d,T(v,d,[])});const U=await ce(".");for(const n of U){const p=document.createElement("option");p.value=n.filename,p.textContent=n.name,r.appendChild(p)}r.addEventListener("change",async()=>{if(!r.value)return;const n=await pe(r.value);l.value=n,i=n,T(v,n,[])}),l.addEventListener("input",()=>{l.value.trim()&&T(v,l.value,[])});const W=await de();for(const n of W){const p=document.createElement("option");p.value=n,p.textContent=n,n==="sto-3g"&&(p.selected=!0),b.appendChild(p)}const q=`3
H2O
O  0.000000  0.000000  0.117300
H  0.000000  0.756950 -0.469200
H  0.000000 -0.756950 -0.469200`;l.value=q,i=q,T(v,q,[]);let e=[],c=new Map,M=!1;function J(n){if(n<0||n>=e.length)return;const p=e[n].step,d=c.get(p);if(d){const S=D(d.atoms),K=[{type:"forces",forces:d.forces}];T(v,S,K)}X.textContent=`Step ${p}: E = ${e[n].energy.toFixed(8)} Ha`;const L=e.map(S=>S.step),Z=e.map(S=>S.energy),V=e.map(S=>S.maxGrad);j(y,L,Z,"Step","Energy (Hartree)","#3b82f6",!1,n),j(E,L,V,"Step","Max Gradient","#ef4444",!0,n)}G.addEventListener("input",()=>{if(!M)return;const n=parseInt(G.value);J(n)});let Y=null;A.addEventListener("click",async()=>{const n=l.value.trim();if(!n){z.textContent="No molecule specified";return}A.disabled=!0,u.classList.remove("hidden"),z.textContent="Starting optimization...",y.innerHTML="",E.innerHTML="",B.style.display="none",F.textContent="",$.style.display="none",e=[],c=new Map,M=!1;const p=parseInt(n.split(/\r?\n/)[0].trim(),10)||0;let d="";Y=new AbortController;try{const Z=(await fetch("/api/run/stream",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({xyz_text:n,basis:b.value,method:h.value,run_type:"optimize",optimizer:m.value,initial_guess:x.value,timeout:1200}),signal:Y.signal})).body?.getReader();if(!Z)return;const V=new TextDecoder;let S="";for(;;){const{done:K,value:se}=await Z.read();if(K)break;S+=V.decode(se,{stream:!0});const ee=S.split(`

`);S=ee.pop()||"";for(const re of ee){const te=re.trim();if(te.startsWith("data: "))try{const H=JSON.parse(te.slice(6));if(H.type==="line"){const N=ge(H.text);d+=N+`
`,F.textContent=d,F.scrollTop=F.scrollHeight;const _=oe(d);if(_.length>e.length){e=_;const O=e[e.length-1];z.textContent=`Step ${O.step}: E = ${O.energy.toFixed(8)} Ha, max grad = ${O.maxGrad.toExponential(2)}`;const w=e.map(P=>P.step),k=e.map(P=>P.energy),Q=e.map(P=>P.maxGrad);j(y,w,k,"Step","Energy (Hartree)","#3b82f6",!1,e.length-1),j(E,w,Q,"Step","Max Gradient","#ef4444",!0,e.length-1)}const I=ne(d,p);if(I.size>c.size){c=I;const O=Math.max(...c.keys()),w=c.get(O);if(w){const k=D(w.atoms),Q=[{type:"forces",forces:w.forces}];T(v,k,Q)}}N.includes("Geometry Optimization Converged")&&(z.textContent="Converged!")}else if(H.type==="error")z.textContent=`Error: ${H.error}`,d+=`
--- ERROR ---
`+(H.raw_output||H.error),F.textContent=d;else if(H.type==="result"){e=oe(d),c=ne(d,p);const N=ue(d);if(N.length>0){const _=D(N),I=e.length>0?c.get(e[e.length-1].step):void 0,O=I?[{type:"forces",forces:I.forces}]:[];T(v,_,O),B.style.display="";let w=`Optimized Geometry (Angstrom):
`;for(const k of N)w+=`  ${k.element.padEnd(4)} ${k.x.toFixed(8)}  ${k.y.toFixed(8)}  ${k.z.toFixed(8)}
`;e.length>0&&(w+=`
Final energy: ${e[e.length-1].energy.toFixed(12)} Hartree
`,w+=`Steps: ${e[e.length-1].step}
`),R.textContent=w}}}catch{}}}}catch(L){L instanceof Error&&L.name==="AbortError"?z.textContent="Cancelled":z.textContent=`Error: ${L}`}M=!0,A.disabled=!1,u.classList.add("hidden"),Y=null,e.length>1&&($.style.display="",G.min="0",G.max=String(e.length-1),G.value=String(e.length-1),X.textContent=`Step ${e[e.length-1].step}`)}),u.addEventListener("click",()=>{Y?.abort()})}fe();
