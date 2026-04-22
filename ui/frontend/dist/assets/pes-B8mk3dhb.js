import{i as V,t as W,u as J,l as K,r as Q,g as ee}from"./styles-CDRzXZyU.js";const z=[{id:"h2",label:"H₂",description:"Hydrogen dissociation",paramType:"bond",paramLabel:"H-H distance",paramUnit:"Å",defaultMin:.4,defaultMax:5,defaultSteps:24,defaultCharge:0,defaultBasis:"cc-pvdz",generateXYZ:e=>`2
H2 R=${e.toFixed(3)}
H  0.0  0.0  0.0
H  0.0  0.0  ${e.toFixed(6)}`},{id:"hf",label:"HF",description:"Hydrogen fluoride dissociation",paramType:"bond",paramLabel:"H-F distance",paramUnit:"Å",defaultMin:.5,defaultMax:3,defaultSteps:20,defaultCharge:0,defaultBasis:"sto-3g",generateXYZ:e=>`2
HF R=${e.toFixed(3)}
H  0.0  0.0  0.0
F  0.0  0.0  ${e.toFixed(6)}`},{id:"lih",label:"LiH",description:"Lithium hydride dissociation",paramType:"bond",paramLabel:"Li-H distance",paramUnit:"Å",defaultMin:.8,defaultMax:4,defaultSteps:20,defaultCharge:0,defaultBasis:"sto-3g",generateXYZ:e=>`2
LiH R=${e.toFixed(3)}
Li  0.0  0.0  0.0
H   0.0  0.0  ${e.toFixed(6)}`},{id:"n2",label:"N₂",description:"Nitrogen dissociation (triple bond)",paramType:"bond",paramLabel:"N-N distance",paramUnit:"Å",defaultMin:.8,defaultMax:3,defaultSteps:20,defaultCharge:0,defaultBasis:"sto-3g",generateXYZ:e=>`2
N2 R=${e.toFixed(3)}
N  0.0  0.0  0.0
N  0.0  0.0  ${e.toFixed(6)}`},{id:"h2o_angle",label:"H₂O (angle)",description:"Water bond angle scan",paramType:"angle",paramLabel:"H-O-H angle",paramUnit:"°",defaultMin:80,defaultMax:180,defaultSteps:20,defaultCharge:0,defaultBasis:"sto-3g",annotations:[{type:"angle",atoms:[0,1,2]}],generateXYZ:e=>{const c=e*Math.PI/180,g=c/2,o=.96*Math.sin(g),M=.96*Math.cos(g);return`3
H2O angle=${e.toFixed(1)}
O  0.0  0.0  0.0
H  ${o.toFixed(6)}  0.0  ${M.toFixed(6)}
H  ${(-o).toFixed(6)}  0.0  ${M.toFixed(6)}`}},{id:"h2o_bond",label:"H₂O (OH stretch)",description:"Water O-H bond stretch",paramType:"bond",paramLabel:"O-H distance",paramUnit:"Å",defaultMin:.5,defaultMax:3,defaultSteps:20,defaultCharge:0,defaultBasis:"sto-3g",annotations:[{type:"distance",atoms:[0,2]}],generateXYZ:e=>{const s=Math.sin(52*Math.PI/180),c=Math.cos(52*Math.PI/180);return`3
H2O R=${e.toFixed(3)}
O  0.0  0.0  0.0
H  ${(.96*s).toFixed(6)}  0.0  ${(.96*c).toFixed(6)}
H  ${(-e*s).toFixed(6)}  0.0  ${(e*c).toFixed(6)}`}},{id:"nh3_inversion",label:"NH₃ (inversion)",description:"Ammonia umbrella inversion",paramType:"height",paramLabel:"N height above H₃ plane",paramUnit:"Å",defaultMin:-.5,defaultMax:.5,defaultSteps:20,defaultCharge:0,defaultBasis:"sto-3g",annotations:[{type:"height",atom:0,planeAtoms:[1,2,3]}],generateXYZ:e=>{const c=Math.sqrt(Math.max(0,1.024144-e*e)),g=Math.sqrt(3)/2;return`4
NH3 h=${e.toFixed(3)}
N  0.0  0.0  ${e.toFixed(6)}
H  ${c.toFixed(6)}  0.0  0.0
H  ${(-c/2).toFixed(6)}  ${(c*g).toFixed(6)}  0.0
H  ${(-c/2).toFixed(6)}  ${(-c*g).toFixed(6)}  0.0`}}],G="";async function Z(){await fetch(`${G}/api/pes/reset`,{method:"POST"})}async function _(e,s,c,g,o,M){try{const p=await(await fetch(`${G}/api/pes/point`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({xyz_text:e,basis:s,method:c,charge:g,post_hf_method:o,use_prev_density:M,timeout:120})})).json();return p.ok?{energy:p.energy,postHfEnergy:p.correction,converged:p.converged??!0}:null}catch{return null}}function D(e,s,c,g,o,M,S){const p=ee(),q=600,P=360,m=72,Y=20,u=30,R=40,k=q-m-Y,h=P-u-R,b=[];for(const a of c)for(const l of a.data)l!==null&&b.push(l);if(b.length===0){e.innerHTML="<p>No data</p>";return}const B=s[0],X=s[s.length-1],C=Math.min(...b)-.02*(Math.max(...b)-Math.min(...b)||1),E=Math.max(...b)+.02*(Math.max(...b)-Math.min(...b)||1),N=a=>m+(a-B)/(X-B||1)*k,H=a=>u+h-(a-C)/(E-C||1)*h;let r=`<svg width="${q}" height="${P}" xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;max-width:${q}px;height:auto" viewBox="0 0 ${q} ${P}">`;const U=5;for(let a=0;a<=U;a++){const l=C+(E-C)*a/U,n=H(l);r+=`<line x1="${m}" y1="${n}" x2="${m+k}" y2="${n}" stroke="${p.grid}" stroke-width="0.5"/>`,r+=`<text x="${m-4}" y="${n+3}" text-anchor="end" font-size="9" fill="${p.dim}">${l.toFixed(4)}</text>`}r+=`<line x1="${m}" y1="${u}" x2="${m}" y2="${u+h}" stroke="${p.axis}" stroke-width="1"/>`,r+=`<line x1="${m}" y1="${u+h}" x2="${m+k}" y2="${u+h}" stroke="${p.axis}" stroke-width="1"/>`;const F=Math.min(s.length-1,8),x=Math.max(1,Math.floor(s.length/F));for(let a=0;a<s.length;a+=x){const l=N(s[a]);r+=`<text x="${l}" y="${u+h+16}" text-anchor="middle" font-size="9" fill="${p.dim}">${s[a].toFixed(2)}</text>`}for(const a of c){let l="";for(let n=0;n<s.length;n++){if(a.data[n]===null)continue;const d=N(s[n]),t=H(a.data[n]);l+=l?` L${d},${t}`:`M${d},${t}`}r+=`<path d="${l}" fill="none" stroke="${a.color}" stroke-width="2"/>`;for(let n=0;n<s.length;n++){if(a.data[n]===null)continue;const d=N(s[n]),t=H(a.data[n]);r+=`<circle cx="${d}" cy="${t}" r="3" fill="${a.color}"/>`,M&&(r+=`<circle cx="${d}" cy="${t}" r="8" fill="transparent" style="cursor:pointer" data-point-idx="${n}"/>`)}}if(S!==void 0&&S>=0&&S<s.length){const a=N(s[S]);r+=`<line x1="${a}" y1="${u}" x2="${a}" y2="${u+h}" stroke="#ff6600" stroke-width="1" stroke-dasharray="4,3" opacity="0.6"/>`;for(const l of c){const n=l.data[S];if(n!==null){const d=H(n);r+=`<circle cx="${a}" cy="${d}" r="6" fill="none" stroke="#ff6600" stroke-width="2"/>`,r+=`<circle cx="${a}" cy="${d}" r="3" fill="#ff6600"/>`}}}for(const a of c){let l=-1,n=1/0;for(let d=0;d<a.data.length;d++)a.data[d]!==null&&a.data[d]<n&&(n=a.data[d],l=d);if(l>=0){const d=N(s[l]),t=H(n);r+=`<circle cx="${d}" cy="${t}" r="6" fill="none" stroke="#00cc44" stroke-width="2"/>`,r+=`<circle cx="${d}" cy="${t}" r="3" fill="#00cc44"/>`,r+=`<text x="${d+8}" y="${t-6}" font-size="9" fill="#00cc44">min</text>`}}let T=m+8;for(const a of c)r+=`<line x1="${T}" y1="${u+8}" x2="${T+16}" y2="${u+8}" stroke="${a.color}" stroke-width="2"/>`,r+=`<text x="${T+20}" y="${u+12}" font-size="10" fill="${p.dim}">${a.label}</text>`,T+=20+a.label.length*6+16;r+=`<text x="${m+k/2}" y="${P-4}" text-anchor="middle" font-size="11" fill="${p.dim}">${g} (${o})</text>`,r+=`<text x="12" y="${u+h/2}" text-anchor="middle" font-size="11" fill="${p.dim}" transform="rotate(-90,12,${u+h/2})">Energy (Hartree)</text>`,r+="</svg>",e.innerHTML=r,M&&e.querySelectorAll("circle[data-point-idx]").forEach(a=>{a.addEventListener("click",()=>{const l=parseInt(a.getAttribute("data-point-idx"),10);M(l)})})}function te(){V();const e=document.getElementById("app");e.innerHTML=`
    <header class="header-top">
      <h1>GANSU</h1>
      <span class="subtitle">Potential Energy Scan</span>
      <nav class="demo-nav">
        <a href="./" class="demo-tab">Calculation</a>
        <a class="demo-tab active">PES</a>
        <a href="./geomopt.html" class="demo-tab">Geometry Opt</a>
      </nav>
      <button id="theme-btn" class="icon-btn" title="Toggle theme"><span id="theme-icon">&#9790;</span></button>
    </header>

    <div class="pes-layout">
      <div class="pes-left">
        <div class="panel pes-scenarios" id="scenarios">
          <h2>Scenarios</h2>
          <div class="scenario-grid" id="scenario-grid"></div>
        </div>

        <div class="panel pes-settings" id="settings-panel">
          <h2>Settings</h2>
          <div class="pes-form">
            <label>Method</label>
            <select id="pes-method">
              <option value="RHF" selected>RHF</option>
              <option value="UHF">UHF</option>
            </select>
            <label>Post-HF</label>
            <select id="pes-posthf">
              <option value="none" selected>None</option>
              <option value="mp2">MP2</option>
              <option value="ccsd">CCSD</option>
              <option value="fci">FCI</option>
            </select>
            <label>Basis</label>
            <select id="pes-basis">
              <option value="sto-3g">STO-3G</option>
              <option value="3-21g">3-21G</option>
              <option value="6-31g">6-31G</option>
              <option value="cc-pvdz" selected>cc-pVDZ</option>
            </select>
            <label id="param-label">Range</label>
            <div class="pes-range">
              <input type="number" id="pes-min" step="0.1" />
              <span>to</span>
              <input type="number" id="pes-max" step="0.1" />
            </div>
            <label>Steps</label>
            <input type="number" id="pes-steps" value="20" min="3" max="100" />
            <div class="pes-actions">
              <button id="pes-run" class="primary-btn">Run Scan</button>
              <button id="pes-cancel" class="secondary-btn hidden">Cancel</button>
            </div>
            <div class="pes-progress" id="pes-progress"></div>
          </div>
        </div>
      </div>

      <div class="pes-right">
        <div class="panel pes-results" id="pes-results">
          <div id="pes-mol-preview" style="width:100%;max-width:360px;aspect-ratio:1;margin:0 auto;"></div>
          <div class="pes-slider-row">
            <input type="range" id="pes-slider" min="0" max="100" value="0" />
            <span id="pes-slider-label" class="pes-slider-label"></span>
          </div>
          <div id="pes-chart"></div>
          <details id="pes-table-details" class="hidden">
            <summary>Data Table</summary>
            <table class="result-table" id="pes-data-table"></table>
          </details>
        </div>
      </div>
    </div>
  `;const s=e.querySelector("#theme-btn"),c=e.querySelector("#theme-icon");s.addEventListener("click",()=>{const t=W();c.textContent=t==="dark"?"☀":"☾"});const g=e.querySelector("#scenario-grid");let o=z[0];function M(){g.innerHTML=z.map(t=>`
      <button class="scenario-card ${t.id===o.id?"active":""}" data-id="${t.id}">
        <strong>${t.label}</strong>
        <span>${t.description}</span>
      </button>
    `).join("")}M();const S=e.querySelector("#pes-min"),p=e.querySelector("#pes-max"),q=e.querySelector("#pes-steps"),P=e.querySelector("#pes-basis"),m=e.querySelector("#pes-method"),Y=e.querySelector("#pes-posthf"),u=e.querySelector("#param-label"),R=e.querySelector("#pes-run"),k=e.querySelector("#pes-cancel"),h=e.querySelector("#pes-progress"),b=e.querySelector("#pes-chart"),B=e.querySelector("#pes-table-details"),X=e.querySelector("#pes-data-table"),C=e.querySelector("#pes-mol-preview"),E=e.querySelector("#pes-slider"),N=e.querySelector("#pes-slider-label");function H(t){const v=o.generateXYZ(t);Q(C,v,o.annotations),N.textContent=`${o.paramLabel} = ${t.toFixed(3)} ${o.paramUnit}`}E.addEventListener("input",()=>{const t=parseFloat(S.value),v=parseFloat(p.value),f=parseInt(E.value)/100,L=t+(v-t)*f;if(H(L),F.length>0&&l){let w=0,O=1/0;for(let $=0;$<F.length;$++){const I=Math.abs(F[$]-L);I<O&&(O=I,w=$)}n(w)}});function r(t){o=t,S.value=String(t.defaultMin),p.value=String(t.defaultMax),q.value=String(t.defaultSteps),P.value=t.defaultBasis,u.textContent=`${t.paramLabel} (${t.paramUnit})`,M(),H(t.defaultMin),E.value="0"}r(z[0]),g.addEventListener("click",t=>{const v=t.target.closest(".scenario-card");if(!v)return;const f=z.find(L=>L.id===v.dataset.id);f&&r(f)});let U=!1,F=[],x=[],T="",a="#3b82f6",l=!1;function n(t){F.length!==0&&D(b,F,[{label:T,color:a,data:x}],o.paramLabel,o.paramUnit,l?d:void 0,t)}function d(t){if(t<0||t>=F.length)return;const v=F[t],f=parseFloat(S.value),L=parseFloat(p.value),w=(v-f)/(L-f)*100;E.value=String(Math.round(w)),H(v),n(t)}R.addEventListener("click",async()=>{const t=parseFloat(S.value),v=parseFloat(p.value),f=parseInt(q.value),L=P.value,w=m.value,O=Y.value,$=[];for(let i=0;i<f;i++)$.push(t+(v-t)*i/(f-1));F=$,x=new Array(f).fill(null),T=O!=="none"?`${w}/${O.toUpperCase()}`:w,a="#3b82f6",l=!1,U=!1,R.disabled=!0,k.classList.remove("hidden"),b.innerHTML="",B.classList.add("hidden"),J(C),await Z();for(let i=0;i<f&&!U;i++){h.textContent=`Point ${i+1}/${f}: ${o.paramLabel} = ${$[i].toFixed(3)} ${o.paramUnit}${i>0?" (density reuse)":""}`;const A=o.generateXYZ($[i]),j=($[i]-t)/(v-t)*100;E.value=String(Math.round(j)),H($[i]);let y=await _(A,L,w,o.defaultCharge,O,i>0);y&&(y.converged?i>0&&x[i-1]!==null&&Math.abs(y.energy+y.postHfEnergy-x[i-1])>.5?(h.textContent+=" (jump, retrying)",await Z(),y=await _(A,L,w,o.defaultCharge,O,!1),y&&y.converged&&(x[i]=y.energy+y.postHfEnergy)):x[i]=y.energy+y.postHfEnergy:(h.textContent+=" (not converged, skipped)",x[i]=null)),D(b,$,[{label:T,color:a,data:x}],o.paramLabel,o.paramUnit,void 0,i)}h.textContent=U?"Cancelled":"Done",R.disabled=!1,k.classList.add("hidden"),l=!0,K(C),n(f-1),B.classList.remove("hidden");let I=`<tr><th>${o.paramLabel} (${o.paramUnit})</th><th>Energy (Hartree)</th></tr>`;for(let i=0;i<f;i++)I+=`<tr><td>${$[i].toFixed(4)}</td><td>${x[i]!==null?x[i].toFixed(10):"FAILED"}</td></tr>`;X.innerHTML=I}),k.addEventListener("click",()=>{U=!0})}te();
