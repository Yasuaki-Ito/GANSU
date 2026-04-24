import{r as de,g as le,t as Re,i as Pe}from"./styles-CDRzXZyU.js";import{f as Se,a as qe,b as De,c as Ne,d as Ie}from"./api-BlpZUjmm.js";const q={xyz_text:"",xyz_file:"",xyz_dir:".",basis:"sto-3g",method:"RHF",charge:0,beta_to_alpha:0,convergence_method:"diis",diis_size:8,diis_include_transform:!1,damping_factor:.9,rohf_parameter_name:"Roothaan",maxiter:100,convergence_energy_threshold:1e-6,schwarz_screening_threshold:1e-12,initial_guess:"core",post_hf_method:"none",n_excited_states:5,spin_type:"singlet",excited_solver:"auto",frozen_core:"none",eri_method:"stored",auxiliary_basis:"",auxiliary_basis_dir:"auxiliary_basis",mulliken:!1,mayer:!1,wiberg:!1,export_molden:!1,verbose:!1,timeout:600};function Ae(s,t,a,o){let e="",i=t[0]||".";s.innerHTML=`
    <div class="panel">
      <h2>Molecule</h2>
      ${t.length>1?`
      <div class="form-group">
        <label>Directory</label>
        <div class="toggle-group" id="dir-group">
          ${t.map(d=>`<button class="toggle ${d===i?"active":""}" data-value="${d}">${d==="."?"small":d}</button>`).join("")}
        </div>
      </div>
      `:""}
      <div class="form-group">
        <label for="sample-select">Sample</label>
        <select id="sample-select">
          <option value="">-- Custom input --</option>
          ${a.map(d=>`<option value="${d.filename}">${d.name} (${d.filename})</option>`).join("")}
        </select>
      </div>
      <div class="form-group">
        <label for="xyz-input">XYZ</label>
        <textarea id="xyz-input" rows="8" placeholder="2

H  0.0 0.0 0.0
H  0.0 0.0 0.74" spellcheck="false"></textarea>
        <p class="hint">Drag & drop .xyz file or paste text</p>
      </div>
      <div id="mol-preview"></div>
    </div>
  `;const r=s.querySelector("#sample-select"),l=s.querySelector("#xyz-input"),m=s.querySelector("#mol-preview"),f=s.querySelector("#dir-group");f&&f.addEventListener("click",async d=>{const n=d.target.closest(".toggle");if(!(!n||!n.dataset.value)){f.querySelectorAll(".toggle").forEach(h=>h.classList.remove("active")),n.classList.add("active"),i=n.dataset.value,r.innerHTML='<option value="">-- Loading... --</option>';try{const h=await Se(i);r.innerHTML='<option value="">-- Custom input --</option>'+h.map(v=>`<option value="${v.filename}">${v.name} (${v.filename})</option>`).join("")}catch{r.innerHTML='<option value="">-- Custom input --</option>'}e="",m.innerHTML=""}});let b;function x(){clearTimeout(b),b=window.setTimeout(()=>{const d=l.value.trim();d?de(m,d,[]):m.innerHTML=""},300)}return r.addEventListener("change",()=>{e=r.value,e?(l.value="",l.placeholder=`Using sample: ${e}`,qe(e,i).then(d=>{d&&r.value===e&&de(m,d,[])}).catch(()=>{m.innerHTML='<p style="color:var(--color-text-dim)">Sample file selected</p>'})):(l.placeholder=`2

H  0.0 0.0 0.0
H  0.0 0.0 0.74`,m.innerHTML=""),o(l.value)}),l.addEventListener("input",()=>{l.value.trim()&&(r.value="",e=""),x()}),l.addEventListener("dragover",d=>{d.preventDefault(),l.classList.add("drag-over")}),l.addEventListener("dragleave",()=>l.classList.remove("drag-over")),l.addEventListener("drop",d=>{d.preventDefault(),l.classList.remove("drag-over");const n=d.dataTransfer?.files[0];if(n&&n.name.endsWith(".xyz")){const h=new FileReader;h.onload=()=>{l.value=h.result,r.value="",e="",x()},h.readAsText(n)}}),{getXyz:()=>l.value.trim(),getXyzFile:()=>e,getXyzDir:()=>i,setXyz:d=>{l.value=d,e="";const n=s.querySelector("#mol-preview");n&&de(n,d,[])}}}function Be(s,t,a=[]){s.innerHTML=`
    <div class="panel">
      <h2>Settings</h2>

      <div class="setting-row">
        <label>Method</label>
        <div class="toggle-group" id="method-group">
          <button class="toggle active" data-value="RHF">RHF</button>
          <button class="toggle" data-value="UHF">UHF</button>
          <button class="toggle" data-value="ROHF">ROHF</button>
        </div>
      </div>

      <div class="setting-row">
        <label>Basis Set</label>
        <select id="basis-select">
          ${t.map(c=>`<option value="${c}" ${c==="sto-3g"?"selected":""}>${c}</option>`).join("")}
        </select>
      </div>

      <div class="setting-row">
        <label>Charge</label>
        <div class="toggle-group" id="charge-group">
          <button class="toggle" data-value="-2">-2</button>
          <button class="toggle" data-value="-1">-1</button>
          <button class="toggle active" data-value="0">0</button>
          <button class="toggle" data-value="1">+1</button>
          <button class="toggle" data-value="2">+2</button>
        </div>
      </div>

      <div class="setting-row hidden" id="rohf-param-row">
        <label>ROHF Param</label>
        <select id="rohf-param-select">
          <option value="Roothaan" selected>Roothaan</option>
          <option value="McWeeny-Diercksen">McWeeny-Diercksen</option>
          <option value="Davidson">Davidson</option>
          <option value="Guest-Saunders">Guest-Saunders</option>
          <option value="Binkley-Pople-Dobosh">Binkley-Pople-Dobosh</option>
          <option value="Faegri-Manne">Faegri-Manne</option>
          <option value="Goddard">Goddard</option>
          <option value="Plakhutin-Gorelik-Breslavskaya">Plakhutin-Gorelik-Breslavskaya</option>
        </select>
      </div>

      <div class="setting-row">
        <label>Multiplicity</label>
        <div class="toggle-group" id="mult-group">
          <button class="toggle active" data-value="0">Singlet</button>
          <button class="toggle" data-value="1">Doublet</button>
          <button class="toggle" data-value="2">Triplet</button>
        </div>
      </div>

      <div class="setting-row">
        <label>Initial Guess</label>
        <div class="toggle-group" id="guess-group">
          <button class="toggle active" data-value="core">Core H</button>
          <button class="toggle" data-value="sad">SAD</button>
          <button class="toggle" data-value="gwh">GWH</button>
        </div>
      </div>

      <div class="setting-row">
        <label>Convergence</label>
        <div class="toggle-group" id="conv-group">
          <button class="toggle active" data-value="diis">DIIS</button>
          <button class="toggle" data-value="damping">Damping</button>
          <button class="toggle" data-value="optimal_damping">Optimal Damping</button>
        </div>
      </div>

      <div class="setting-row" id="diis-params">
        <label>DIIS History</label>
        <input type="number" id="diis-size" value="8" min="2" max="20" class="num-input">
        <label class="checkbox-label" style="margin-top:4px"><input type="checkbox" id="chk-diis-transform"> Include transform</label>
      </div>

      <div class="setting-row hidden" id="damping-params">
        <label>Damping Factor</label>
        <input type="number" id="damping-factor" value="0.9" min="0.05" max="0.95" step="0.05" class="num-input">
      </div>

      <div class="setting-row">
        <label>ERI Method</label>
        <div class="toggle-group" id="eri-group">
          <button class="toggle active" data-value="stored">Stored</button>
          <button class="toggle" data-value="direct">Direct</button>
          <button class="toggle" data-value="ri">RI</button>
          <button class="toggle" data-value="direct_ri">Direct RI</button>
        </div>
      </div>

      <div class="setting-row hidden" id="aux-basis-row">
        <label>Auxiliary Basis</label>
        <select id="aux-basis-select">
          <option value="">-- Select --</option>
          ${a.map(c=>`<option value="${c.name}" data-dir="${c.dir}">${c.dir==="auxiliary_basis"?"aux_basis":"basis"}/${c.name}</option>`).join("")}
        </select>
      </div>

      <div class="setting-row">
        <label>Post-HF</label>
        <div class="toggle-group" id="posthf-group">
          <button class="toggle active" data-value="none">None</button>
          <button class="toggle" data-value="mp2">MP2</button>
          <button class="toggle" data-value="mp3">MP3</button>
          <button class="toggle" data-value="ccsd">CCSD</button>
          <button class="toggle" data-value="ccsd_t">CCSD(T)</button>
          <button class="toggle" data-value="fci">FCI</button>
        </div>
      </div>

      <div class="setting-row">
        <label>Excited States</label>
        <div class="toggle-group" id="excited-group">
          <button class="toggle active" data-value="none">None</button>
          <button class="toggle" data-value="cis">CIS</button>
          <button class="toggle" data-value="adc2">ADC(2)</button>
          <button class="toggle" data-value="adc2x">ADC(2)-x</button>
          <button class="toggle" data-value="eom_mp2">EOM-MP2</button>
          <button class="toggle" data-value="eom_cc2">EOM-CC2</button>
          <button class="toggle" data-value="eom_ccsd">EOM-CCSD</button>
        </div>
      </div>

      <div class="setting-row hidden" id="excited-params">
        <label>Spin Type</label>
        <div class="toggle-group" id="spin-type-group">
          <button class="toggle active" data-value="singlet">Singlet</button>
          <button class="toggle" data-value="triplet">Triplet</button>
        </div>
        <label style="margin-top:6px">Number of States</label>
        <input type="number" id="n-excited-states" value="5" min="1" max="100" class="num-input">
        <div id="solver-row" class="hidden" style="margin-top:6px">
          <label>Solver</label>
          <div class="toggle-group" id="solver-group">
            <button class="toggle active" data-value="auto">Auto</button>
            <button class="toggle" data-value="schur_static">Schur Static</button>
            <button class="toggle" data-value="schur_omega">Schur Omega</button>
            <button class="toggle" data-value="full">Full</button>
          </div>
        </div>
      </div>

      <div class="setting-row">
        <label>Frozen Core</label>
        <div class="toggle-group" id="frozen-core-group">
          <button class="toggle active" data-value="none">None</button>
          <button class="toggle" data-value="auto">Auto</button>
        </div>
      </div>

      <div class="setting-row">
        <label>Analysis</label>
        <div class="checkbox-group">
          <label class="checkbox-label"><input type="checkbox" id="chk-mulliken"> Mulliken</label>
          <label class="checkbox-label"><input type="checkbox" id="chk-mayer"> Mayer</label>
          <label class="checkbox-label"><input type="checkbox" id="chk-wiberg"> Wiberg</label>
          <label class="checkbox-label"><input type="checkbox" id="chk-molden"> Molden</label>
        </div>
      </div>


      <details class="advanced-settings">
        <summary>Advanced</summary>
        <div class="setting-row">
          <label>Max Iterations</label>
          <input type="number" id="maxiter" value="100" min="1" max="1000" class="num-input">
        </div>
        <div class="setting-row">
          <label>Energy Threshold</label>
          <input type="text" id="conv-thresh" value="1e-6" class="num-input">
        </div>
        <div class="setting-row">
          <label>Schwarz Threshold</label>
          <input type="text" id="schwarz-thresh" value="1e-12" class="num-input">
        </div>
        <div class="setting-row">
          <label>Timeout (s)</label>
          <input type="number" id="timeout" value="600" min="10" max="3600" class="num-input">
        </div>
      </details>
    </div>
  `,s.querySelectorAll(".toggle-group").forEach(c=>{c.addEventListener("click",S=>{const T=S.target.closest(".toggle");T&&(c.querySelectorAll(".toggle").forEach(H=>H.classList.remove("active")),T.classList.add("active"))})});const o=s.querySelector("#conv-group"),e=s.querySelector("#diis-params"),i=s.querySelector("#damping-params");o.addEventListener("click",()=>{const S=o.querySelector(".active")?.dataset.value;e.classList.toggle("hidden",S!=="diis"),i.classList.toggle("hidden",S!=="damping")});const r=s.querySelector("#method-group"),l=s.querySelector("#rohf-param-row");r.addEventListener("click",()=>{const c=r.querySelector(".active");l.classList.toggle("hidden",c?.dataset.value!=="ROHF")});const m=s.querySelector("#eri-group"),f=s.querySelector("#aux-basis-row");m.addEventListener("click",()=>{const S=m.querySelector(".active")?.dataset.value;f.classList.toggle("hidden",S!=="ri"&&S!=="direct_ri")});const b=s.querySelector("#posthf-group"),x=s.querySelector("#excited-group"),d=s.querySelector("#excited-params");b.addEventListener("click",()=>{const c=b.querySelector(".active");c?.dataset.value&&c.dataset.value!=="none"&&(x.querySelectorAll(".toggle").forEach(S=>S.classList.remove("active")),x.querySelector('[data-value="none"]')?.classList.add("active"),d.classList.add("hidden"))});const n=s.querySelector("#solver-row"),h=new Set(["adc2","adc2x","eom_mp2","eom_cc2"]);x.addEventListener("click",()=>{const S=x.querySelector(".active")?.dataset.value||"none";S!=="none"&&(b.querySelectorAll(".toggle").forEach(T=>T.classList.remove("active")),b.querySelector('[data-value="none"]')?.classList.add("active")),d.classList.toggle("hidden",S==="none"),n.classList.toggle("hidden",!h.has(S))});function v(c){return s.querySelector(`#${c} .toggle.active`)?.dataset.value||""}function u(c){return s.querySelector(`#${c}`)?.checked||!1}function C(c,S){const T=parseFloat(s.querySelector(`#${c}`)?.value||"");return isNaN(T)?S:T}return{getParams:()=>({method:v("method-group")||q.method,basis:s.querySelector("#basis-select")?.value||q.basis,charge:parseInt(v("charge-group")||"0",10),beta_to_alpha:parseInt(v("mult-group")||"0",10),initial_guess:v("guess-group")||q.initial_guess,convergence_method:v("conv-group")||q.convergence_method,diis_size:C("diis-size",q.diis_size),diis_include_transform:u("chk-diis-transform"),damping_factor:C("damping-factor",q.damping_factor),rohf_parameter_name:s.querySelector("#rohf-param-select")?.value||"Roothaan",eri_method:v("eri-group")||q.eri_method,auxiliary_basis:s.querySelector("#aux-basis-select")?.value||"",auxiliary_basis_dir:s.querySelector("#aux-basis-select")?.selectedOptions[0]?.dataset.dir||"auxiliary_basis",post_hf_method:v("excited-group")!=="none"?v("excited-group"):v("posthf-group")||q.post_hf_method,n_excited_states:C("n-excited-states",q.n_excited_states),spin_type:v("spin-type-group")||q.spin_type,excited_solver:v("solver-group")||"auto",frozen_core:v("frozen-core-group")||"none",mulliken:u("chk-mulliken"),mayer:u("chk-mayer"),wiberg:u("chk-wiberg"),export_molden:u("chk-molden"),maxiter:C("maxiter",q.maxiter),convergence_energy_threshold:C("conv-thresh",q.convergence_energy_threshold),schwarz_screening_threshold:C("schwarz-thresh",q.schwarz_screening_threshold),timeout:C("timeout",q.timeout)}),setParams:c=>{if(c.method&&M("method-group",c.method),c.basis){const S=s.querySelector("#basis-select");S&&(S.value=c.basis)}if(c.post_hf_method&&(["cis","adc2","adc2x","eom_mp2","eom_cc2","eom_ccsd"].includes(c.post_hf_method)?(M("posthf-group","none"),M("excited-group",c.post_hf_method)):(M("posthf-group",c.post_hf_method),M("excited-group","none"))),c.initial_guess&&M("guess-group",c.initial_guess),c.eri_method&&M("eri-group",c.eri_method),c.n_excited_states){const S=s.querySelector("#n-excited-states");S&&(S.value=c.n_excited_states)}}};function M(c,S){s.querySelectorAll(`[data-group="${c}"]`).forEach(H=>{H.classList.toggle("active",H.dataset.value===S)})}}function Ge(s,t,a){if(t.length<2){s.innerHTML='<p style="color:var(--color-text-dim);font-size:0.85rem;">Not enough iterations</p>';return}const o=t.filter(w=>w.iter>=1&&w.deltaE!==0).map(w=>({x:w.iter,y:Math.log10(Math.abs(w.deltaE))}));if(o.length<2){s.innerHTML='<p style="color:var(--color-text-dim);font-size:0.85rem;">No data</p>';return}const e=le(),i=320,r=200,l=52,m=12,f=24,b=32,x=i-l-m,d=r-f-b,n=o[0].x,h=o[o.length-1].x,v=Math.min(...o.map(w=>w.y),Math.log10(a))-1,u=Math.max(o[0].y,0)+.5,C=w=>l+(w-n)/(h-n||1)*x,M=w=>f+d-(w-v)/(u-v||1)*d;let c=`<svg width="${i}" height="${r}" xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;height:auto;" viewBox="0 0 ${i} ${r}">`;const S=2,T=Math.ceil(v/S)*S;for(let w=T;w<=u;w+=S){const E=M(w);E<f||E>f+d||(c+=`<line x1="${l}" y1="${E}" x2="${l+x}" y2="${E}" stroke="${e.grid}" stroke-width="0.5"/>`,c+=`<text x="${l-4}" y="${E+3}" text-anchor="end" font-size="9" fill="${e.dim}">1e${w}</text>`)}c+=`<line x1="${l}" y1="${f}" x2="${l}" y2="${f+d}" stroke="${e.axis}" stroke-width="1"/>`,c+=`<line x1="${l}" y1="${f+d}" x2="${l+x}" y2="${f+d}" stroke="${e.axis}" stroke-width="1"/>`;const H=Math.max(1,Math.round((h-n)/5));for(let w=n;w<=h;w+=H){const E=C(w);c+=`<text x="${E}" y="${f+d+14}" text-anchor="middle" font-size="9" fill="${e.dim}">${w}</text>`}const L=M(Math.log10(a));L>=f&&L<=f+d&&(c+=`<line x1="${l}" y1="${L}" x2="${l+x}" y2="${L}" stroke="${e.error}" stroke-width="1" stroke-dasharray="4,3"/>`,c+=`<text x="${l+x+2}" y="${L+3}" font-size="8" fill="${e.error}">${a.toExponential(0)}</text>`);let z="";for(let w=0;w<o.length;w++){const E=C(o[w].x),I=M(o[w].y);z+=w===0?`M${E},${I}`:` L${E},${I}`}c+=`<path d="${z}" fill="none" stroke="${e.accent}" stroke-width="1.5"/>`;for(const w of o)c+=`<circle cx="${C(w.x)}" cy="${M(w.y)}" r="2.5" fill="${e.accent}"/>`;c+=`<text x="${i/2}" y="14" text-anchor="middle" font-size="11" fill="${e.titleSvg}">SCF Convergence</text>`,c+=`<text x="${l+x/2}" y="${r-2}" text-anchor="middle" font-size="9" fill="${e.dim}">Iteration</text>`,c+=`<text x="12" y="${f+d/2}" text-anchor="middle" font-size="9" fill="${e.dim}" transform="rotate(-90,12,${f+d/2})">log10(|deltaE|)</text>`,c+="</svg>",s.innerHTML=`<h3>SCF Convergence</h3>${c}`}const we=12,je=1e-4,fe=6;function me(s){const t=[];let a=0;for(;a<s.length;){const o=[a],e=[s[a].occupation],i=s[a].energy;for(;a+1<s.length&&Math.abs(s[a+1].energy-i)<je;)a++,o.push(a),e.push(s[a].occupation);const r=o.reduce((l,m)=>l+s[m].energy,0)/o.length;t.push({indices:o,energy:r,occupations:e}),a++}return t}function ae(s,t,a){return s+t*a+(t-1)*fe}function Ee(s,t){const a=s.slice(),o=a.map((e,i)=>i);o.sort((e,i)=>a[e]-a[i]);for(let e=1;e<o.length;e++){const i=o[e-1],r=o[e];a[r]-a[i]<t&&(a[r]=a[i]+t)}return a}function Me(s,t,a){let o=2;const e=12,i=.5,r=Math.min(a,520),l=document.createElement("div");l.style.position="relative";const m=document.createElement("div");m.style.cssText="display:flex;align-items:center;justify-content:center;gap:6px;padding:2px 0 4px;font-size:11px;color:var(--color-text-dim,#888)";const f="width:22px;height:22px;border:1px solid var(--color-border,#ccc);border-radius:4px;background:var(--color-surface,#fff);color:var(--color-text,#333);cursor:pointer;font-size:14px;line-height:1;display:flex;align-items:center;justify-content:center",b=document.createElement("button");b.style.cssText=f,b.textContent="−",b.title="Zoom out";const x=document.createElement("button");x.style.cssText=f,x.textContent="+",x.title="Zoom in";const d=document.createElement("span");d.style.cssText="min-width:40px;text-align:center;font-size:10px",m.appendChild(b),m.appendChild(d),m.appendChild(x),l.appendChild(m);const n=document.createElement("div");n.style.overflowY="auto",n.style.maxHeight=`${r}px`,n.style.position="relative",l.appendChild(n);function h(u){if(u=Math.max(i,Math.min(e,u)),u===o)return;const C=n.scrollHeight>r?n.scrollTop/(n.scrollHeight-r):0;o=u,v();const M=n.scrollHeight-r;M>0&&(n.scrollTop=C*M)}function v(){n.innerHTML=t(o),d.textContent=`${Math.round(o*100)}%`,b.disabled=o<=i,x.disabled=o>=e}n.addEventListener("wheel",u=>{if(!u.ctrlKey&&!u.metaKey)return;u.preventDefault();const C=u.deltaY<0?1.2:1/1.2;h(o*C)},{passive:!1}),x.addEventListener("click",()=>h(o*1.3)),b.addEventListener("click",()=>h(o/1.3)),v(),s.appendChild(l)}function Ue(s,t,a){let i=`<polygon points="${s-3.5-3},${t-2} ${s-3.5},${t-10} ${s-3.5+3},${t-2}" fill="${a}"/>`;return i+=`<polygon points="${s+3.5-3},${t-10} ${s+3.5},${t-2} ${s+3.5+3},${t-10}" fill="${a}"/>`,i}function Ce(s,t,a){return`<polygon points="${s-3},${t-2} ${s},${t-10} ${s+3},${t-2}" fill="${a}"/>`}function Ye(s,t,a){return`<polygon points="${s-3},${t-10} ${s},${t-2} ${s+3},${t-10}" fill="${a}"/>`}function Ve(s,t,a,o,e){const i=(t-s)*27.2114;if(Math.abs(i)<.01)return"";const r=a(s),l=a(t);let m=`<line x1="${o}" y1="${r}" x2="${o}" y2="${l}" stroke="${e.gap}" stroke-width="1.5" stroke-dasharray="3,2"/>`;m+=`<polygon points="${o-3},${r-4} ${o},${r} ${o+3},${r-4}" fill="${e.gap}"/>`,m+=`<polygon points="${o-3},${l+4} ${o},${l} ${o+3},${l+4}" fill="${e.gap}"/>`;const f=(r+l)/2;return m+=`<text x="${o+6}" y="${f+4}" font-size="10" fill="${e.gap}">${i.toFixed(2)} eV</text>`,m}function ve(s){return s==="occ"||s==="closed"||s==="open"}function he(s){for(let t=s.length-1;t>=0;t--)if(ve(s[t].occupation))return t;return-1}function We(s,t){const a=t.length;if(a===0)return;const o=me(t),e=he(t),i=e+1<a?e+1:-1,r=28,l=24,m=70,f=40,b=Math.max(...o.map(H=>H.indices.length)),x=ae(m,b,f),d=Math.max(300,x+100),n=t[0].energy,h=t[a-1].energy,u=(h-n||1)*.1,C=n-u,M=h+u,c=Math.max(180,Math.min(480,o.length*40));function S(H){const L=le(),z=c*H,w=z+r+l,E=$=>r+z-($-C)/(M-C)*z,I=o.map($=>E($.energy)),R=Ee(I,we);let g=`<svg width="${d}" height="${w}" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0 auto;">`;g+=`<line x1="${m-8}" y1="${r}" x2="${m-8}" y2="${r+z}" stroke="${L.axis}" stroke-width="1"/>`;for(let $=0;$<o.length;$++){const y=o[$],_=E(y.energy),p=R[$],k=y.indices.length,D=ae(m,k,f),G=e>=0&&y.indices.includes(e),F=i>=0&&y.indices.includes(i);for(let N=0;N<k;N++){const Q=y.indices[N],Y=t[Q].occupation,ee=ve(Y),A=Q===e||Q===i,te=ee?L.occupied:L.virtual,se=A?2.5:1.5,j=m+N*(f+fe);g+=`<line x1="${j}" y1="${_}" x2="${j+f}" y2="${_}" stroke="${te}" stroke-width="${se}"/>`,Y==="occ"||Y==="closed"?g+=Ue(j+f/2,_,te):Y==="open"&&(g+=Ce(j+f/2,_,te))}Math.abs(p-_)>3&&(g+=`<line x1="${m-10}" y1="${_}" x2="${m-14}" y2="${p}" stroke="${L.leader}" stroke-width="0.5"/>`);const P=k>1?` (×${k})`:"";g+=`<text x="${m-16}" y="${p+4}" text-anchor="end" font-size="10" fill="${L.label}">${y.energy.toFixed(3)}${P}</text>`;const J=y.occupations.some(N=>N==="open");if(G){const N=J?"SOMO":"HOMO";g+=`<text x="${D+6}" y="${_+4}" font-size="11" fill="${L.occupied}" font-weight="bold">${N}</text>`}else J&&(g+=`<text x="${D+6}" y="${_+4}" font-size="10" fill="${L.alpha}" font-weight="bold">SOMO</text>`);F&&!G&&(g+=`<text x="${D+6}" y="${_+4}" font-size="11" fill="${L.dim}" font-weight="bold">LUMO</text>`)}return e>=0&&i>=0&&(g+=Ve(t[e].energy,t[i].energy,E,x+50,L)),g+=`<text x="${d/2}" y="14" text-anchor="middle" font-size="10" fill="${L.hint}">Ctrl+Scroll or +/− to zoom</text>`,g+="</svg>",g}s.innerHTML="";const T=c+r+l;Me(s,S,T)}function Ze(s,t,a){const o=t.length,e=a.length;if(o===0&&e===0)return;const i=me(t),r=me(a),l=he(t),m=l+1<o?l+1:-1,f=he(a),b=f+1<e?f+1:-1,x=28,d=24,n=60,h=36,v=50,u=Math.max(1,...i.map(p=>p.indices.length)),C=Math.max(1,...r.map(p=>p.indices.length)),M=ae(n,u,h),c=M+v,S=ae(c,C,h),T=Math.max(420,S+80),H=[];for(const p of t)H.push(p.energy);for(const p of a)H.push(p.energy);H.sort((p,k)=>p-k);const L=H[0],z=H[H.length-1],E=(z-L||1)*.1,I=L-E,R=z+E,g=Math.max(i.length,r.length),$=Math.max(180,Math.min(480,g*40));function y(p){const k=le(),D=$*p,G=D+x+d,F=N=>x+D-(N-I)/(R-I)*D;let P=`<svg width="${T}" height="${G}" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0 auto;">`;P+=`<text x="${T/2}" y="14" text-anchor="middle" font-size="10" fill="${k.hint}">Ctrl+Scroll or +/− to zoom</text>`;function J(N,Q,Y,ee,A,te,se,j,ke){const Le=(A+te)/2;P+=`<text x="${Le}" y="${x-4}" text-anchor="middle" font-size="10" fill="${se}" font-weight="bold">${ke}</text>`,P+=`<line x1="${A-8}" y1="${x}" x2="${A-8}" y2="${x+D}" stroke="${k.axis}" stroke-width="1"/>`;const ze=N.map(Z=>F(Z.energy)),He=Ee(ze,we);for(let Z=0;Z<N.length;Z++){const V=N[Z],U=F(V.energy),K=He[Z],X=V.indices.length,oe=ae(A,X,h),be=Y>=0&&V.indices.includes(Y),Te=ee>=0&&V.indices.includes(ee);for(let B=0;B<X;B++){const W=V.indices[B],xe=ve(Q[W].occupation),Fe=W===Y||W===ee,re=xe?se:k.virtual,Oe=Fe?2.5:1.5,ce=A+B*(h+fe);if(P+=`<line x1="${ce}" y1="${U}" x2="${ce+h}" y2="${U}" stroke="${re}" stroke-width="${Oe}"/>`,xe){const $e=ce+h/2;P+=j?Ce($e,U,re):Ye($e,U,re)}}if(j){Math.abs(K-U)>3&&(P+=`<line x1="${A-10}" y1="${U}" x2="${A-14}" y2="${K}" stroke="${k.leader}" stroke-width="0.5"/>`);const B=X>1?` (×${X})`:"";P+=`<text x="${A-16}" y="${K+4}" text-anchor="end" font-size="9" fill="${k.label}">${V.energy.toFixed(3)}${B}</text>`}else{Math.abs(K-U)>3&&(P+=`<line x1="${oe+2}" y1="${U}" x2="${oe+6}" y2="${K}" stroke="${k.leader}" stroke-width="0.5"/>`);const B=X>1?` (×${X})`:"";P+=`<text x="${oe+8}" y="${K+4}" text-anchor="start" font-size="9" fill="${k.label}">${V.energy.toFixed(3)}${B}</text>`}if(be){const B=j?oe+4:A-12,W=j?"start":"end";P+=`<text x="${B}" y="${U+4}" text-anchor="${W}" font-size="9" fill="${se}" font-weight="bold">HOMO</text>`}if(Te&&!be){const B=j?oe+4:A-12,W=j?"start":"end";P+=`<text x="${B}" y="${U+4}" text-anchor="${W}" font-size="9" fill="${k.dim}" font-weight="bold">LUMO</text>`}}}return J(i,t,l,m,n,M,k.alpha,!0,"α"),J(r,a,f,b,c,S,k.beta,!1,"β"),P+="</svg>",P}s.innerHTML="";const _=$+x+d;Me(s,y,_)}function Ke(s,t,a,o){if(!t||t.length===0){s.innerHTML='<p style="color:var(--color-text-dim);font-size:0.85rem;">No excited state data</p>';return}const e=le(),i=o==="triplet",r=700,l=280,m=48,f=16,b=28,x=38,d=r-m-f,n=l-b-x,h=t.map(g=>g.energy_ev),v=Math.max(0,Math.min(...h)-2),u=Math.max(...h)+2,C=.4,M=200,c=(u-v)/M,S=new Float64Array(M);for(let g=0;g<M;g++){const $=v+g*c;let y=0;for(const _ of t){const p=_.osc_strength;if(p>0){const k=$-_.energy_ev;y+=p*Math.exp(-k*k/(2*C*C))}}S[g]=y}const T=Math.max(...S)*1.15||1,H=g=>m+(g-v)/(u-v||1)*d,L=g=>b+n-g/T*n;let z=`<svg width="${r}" height="${l}" xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;height:auto;" viewBox="0 0 ${r} ${l}">`;const w=Math.ceil((u-v)/6);for(let g=Math.ceil(v);g<=u;g+=Math.max(1,w)){const $=H(g);$<m||$>m+d||(z+=`<line x1="${$}" y1="${b}" x2="${$}" y2="${b+n}" stroke="${e.grid}" stroke-width="0.5"/>`,z+=`<text x="${$}" y="${b+n+14}" text-anchor="middle" font-size="9" fill="${e.dim}">${g}</text>`)}if(z+=`<line x1="${m}" y1="${b}" x2="${m}" y2="${b+n}" stroke="${e.axis}" stroke-width="1"/>`,z+=`<line x1="${m}" y1="${b+n}" x2="${m+d}" y2="${b+n}" stroke="${e.axis}" stroke-width="1"/>`,!i){let g=`M${m},${b+n}`;for(let y=0;y<M;y++)g+=` L${H(v+y*c).toFixed(1)},${L(S[y]).toFixed(1)}`;g+=` L${H(u).toFixed(1)},${b+n} Z`,z+=`<path d="${g}" fill="${e.accent}" fill-opacity="0.15" stroke="none"/>`;let $="";for(let y=0;y<M;y++){const _=H(v+y*c),p=L(S[y]);$+=y===0?`M${_.toFixed(1)},${p.toFixed(1)}`:` L${_.toFixed(1)},${p.toFixed(1)}`}z+=`<path d="${$}" fill="none" stroke="${e.accent}" stroke-width="1.5"/>`}for(const g of t){const $=H(g.energy_ev);if($<m||$>m+d||!i&&g.osc_strength<=0)continue;const y=i?n*.5:g.osc_strength/T*n,_=b+n-y,p=i?e.error:e.accent;z+=`<line x1="${$.toFixed(1)}" y1="${(b+n).toFixed(1)}" x2="${$.toFixed(1)}" y2="${_.toFixed(1)}" stroke="${p}" stroke-width="1.5" stroke-opacity="0.7"/>`,z+=`<circle cx="${$.toFixed(1)}" cy="${_.toFixed(1)}" r="2" fill="${p}"/>`}const E=i?"Triplet":"Singlet",I=`${a} ${E} Absorption Spectrum`;z+=`<text x="${r/2}" y="16" text-anchor="middle" font-size="11" fill="${e.titleSvg}">${I}</text>`,z+=`<text x="${m+d/2}" y="${l-4}" text-anchor="middle" font-size="9" fill="${e.dim}">Energy (eV)</text>`,i||(z+=`<text x="12" y="${b+n/2}" text-anchor="middle" font-size="9" fill="${e.dim}" transform="rotate(-90,12,${b+n/2})">Osc. Strength (arb.)</text>`),z+="</svg>";let R='<table class="result-table"><tr><th>State</th><th>Energy (Ha)</th><th>Energy (eV)</th><th>f</th><th>Transitions</th></tr>';for(const g of t)R+=`<tr>
      <td>${g.state}</td>
      <td>${g.energy_ha.toFixed(6)}</td>
      <td>${g.energy_ev.toFixed(3)}</td>
      <td>${g.osc_strength.toFixed(4)}</td>
      <td style="font-size:0.8rem">${Xe(g.transitions)}</td>
    </tr>`;R+="</table>",s.innerHTML=`<h3>Excited States (${a} ${E})</h3>${z}${R}`}function Xe(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}function Je(s){s.innerHTML='<div id="results-container"></div>';const t=s.querySelector("#results-container");function a(e){let i='<div class="results">';const r=e.molecule,l=e.basis_set;if((r.num_atoms!==void 0||l.num_basis!==void 0)&&(i+='<div class="summary-row">',r.num_atoms!==void 0&&(i+=`
          <div class="panel result-card">
            <h3>Molecule</h3>
            <table class="result-table">
              ${O("Atoms",String(r.num_atoms))}
              ${r.num_electrons!==void 0?O("Electrons",String(r.num_electrons)):""}
              ${r.alpha_electrons!==void 0?ue("Alpha",String(r.alpha_electrons)):""}
              ${r.beta_electrons!==void 0?ue("Beta",String(r.beta_electrons)):""}
            </table>
          </div>`),l.num_basis!==void 0&&(i+=`
          <div class="panel result-card">
            <h3>Basis Set</h3>
            <table class="result-table">
              ${O("Basis Functions",String(l.num_basis))}
              ${l.num_primitives!==void 0?ue("Primitives",String(l.num_primitives)):""}
              ${l.num_auxiliary!==void 0?O("Auxiliary",String(l.num_auxiliary)):""}
            </table>
          </div>`),i+="</div>"),e.summary.total_energy!==void 0&&(i+=`
        <div class="panel result-card">
          <h3>Energy Summary</h3>
          <table class="result-table">
            ${O("Method",e.summary.method||"-")}
            ${O("Total Energy",ie(e.summary.total_energy)+" Hartree")}
            ${e.summary.electronic_energy!==void 0?O("Electronic Energy",ie(e.summary.electronic_energy)+" Hartree"):""}
            ${e.summary.iterations!==void 0?O("SCF Iterations",String(e.summary.iterations)):""}
            ${e.summary.convergence_algorithm?O("Convergence",e.summary.convergence_algorithm):""}
            ${e.summary.initial_guess?O("Initial Guess",e.summary.initial_guess):""}
            ${e.summary.energy_difference!==void 0?O("Final |deltaE|",e.summary.energy_difference.toExponential(2)):""}
            ${e.summary.computing_time_ms!==void 0?O("Computing Time",e.summary.computing_time_ms.toFixed(1)+" ms"):""}
          </table>
        </div>`),e.post_hf&&(e.post_hf.correction!==void 0||e.post_hf.total_energy!==void 0)){const n=e.molecule,h=n?.num_frozen||0,v=n?.num_occ,u=n?.num_vir,C=h>0?v-h:v;i+=`
        <div class="panel result-card">
          <h3>Post-HF: ${e.post_hf.method||""}${h>0?" (frozen core)":""}</h3>
          <table class="result-table">
            ${v!==void 0?O("Occupied / Virtual",`${v} / ${u}`):""}
            ${h>0?O("Frozen Core",`${h} orbitals frozen, ${C} active occupied`):""}
            ${e.post_hf.correction!==void 0?O("Correlation Energy",ie(e.post_hf.correction)+" Hartree"):""}
            ${e.post_hf.total_energy!==void 0?O("Total Energy",ie(e.post_hf.total_energy)+" Hartree"):""}
          </table>
        </div>`}if(e.excited_states&&e.excited_states.length>0&&(i+='<div class="panel result-card full-width" id="spectrum-chart-container"></div>'),e.orbital_energies.length>0){const n=e.orbital_energies_beta.length>0;i+='<div class="panel result-card orbital-card" id="orbital-diagram-container"><h3>Orbital Energies</h3></div>',n?i+=`
          <details class="panel result-card">
            <summary>Orbital Energies (Alpha) — Table</summary>
            ${pe(e.orbital_energies)}
          </details>
          <details class="panel result-card">
            <summary>Orbital Energies (Beta) — Table</summary>
            ${pe(e.orbital_energies_beta)}
          </details>`:i+=`
          <details class="panel result-card">
            <summary>Orbital Energies — Table</summary>
            ${pe(e.orbital_energies)}
          </details>`}const m=e.molecule.atoms||[],f=n=>{const h=m[n];return h?`${h.element}${n+1}`:String(n)};if(e.mulliken.length>0&&(i+=`
        <div class="panel result-card">
          <h3>Mulliken Population</h3>
          <table class="result-table">
            <tr><th>Atom</th><th>Charge</th></tr>
            ${e.mulliken.map((n,h)=>`<tr><td>${f(h)}</td><td>${n.charge.toFixed(6)}</td></tr>`).join("")}
          </table>
        </div>`),e.mayer_bond_order.length>0&&(i+=`
        <div class="panel result-card">
          <h3>Mayer Bond Order</h3>
          ${ye(e.mayer_bond_order,f)}
        </div>`),e.wiberg_bond_order.length>0&&(i+=`
        <div class="panel result-card">
          <h3>Wiberg Bond Order</h3>
          ${ye(e.wiberg_bond_order,f)}
        </div>`),i+='<div class="panel result-card" id="conv-graph-container"></div>',e.molden_content&&(i+=`
        <div class="panel result-card">
          <h3>Molden</h3>
          <button class="secondary-btn" id="download-molden">Download output.molden</button>
        </div>`),i+=`
      <details class="panel result-card">
        <summary>Raw Output</summary>
        <pre class="raw-output">${ge(e.raw_output)}</pre>
      </details>`,i+="</div>",t.innerHTML=i,e.molden_content){const n=()=>{const v=new Blob([e.molden_content],{type:"chemical/x-molden"}),u=URL.createObjectURL(v),C=document.createElement("a");C.href=u,C.download="output.molden",C.click(),URL.revokeObjectURL(u)};n(),t.querySelector("#download-molden")?.addEventListener("click",n)}const b=t.querySelector("#orbital-diagram-container");b&&e.orbital_energies.length>0&&(e.orbital_energies_beta.length>0?Ze(b,e.orbital_energies,e.orbital_energies_beta):We(b,e.orbital_energies));const x=t.querySelector("#conv-graph-container");if(x&&e.scf_iterations.length>0){const n=e.scf_iterations.filter(v=>v.delta_e!==void 0).map(v=>({iter:v.iteration,deltaE:v.delta_e})),h=e.summary.convergence_criterion||1e-6;Ge(x,n,h)}const d=t.querySelector("#spectrum-chart-container");d&&e.excited_states&&e.excited_states.length>0&&Ke(d,e.excited_states,e.excited_states_method||"",e.excited_states_spin||"singlet")}function o(e,i){let r='<div class="results">';r+=`
      <div class="panel result-card error-card">
        <h3>Error</h3>
        <pre class="error-output">${ge(e)}</pre>
      </div>`,i&&(r+=`
        <details class="panel result-card" open>
          <summary>Output</summary>
          <pre class="raw-output">${ge(i)}</pre>
        </details>`),r+="</div>",t.innerHTML=r}return{show:a,showError:o,hide:()=>{t.innerHTML=""}}}function O(s,t){return`<tr><td class="label">${s}</td><td class="value">${t}</td></tr>`}function ue(s,t){return`<tr><td class="label sub-label">${s}</td><td class="value">${t}</td></tr>`}function ie(s){return s.toFixed(10)}function ye(s,t){if(s.length===0)return"";const a=s.length;let o='<table class="result-table bond-table"><tr><th></th>';for(let e=0;e<a;e++)o+=`<th>${t(e)}</th>`;o+="</tr>";for(let e=0;e<a;e++){o+=`<tr><th>${t(e)}</th>`;for(let i=0;i<s[e].length;i++){const r=s[e][i],l=r>.5?"bond-strong":"";o+=`<td class="${l}">${r.toFixed(3)}</td>`}o+="</tr>"}return o+="</table>",o}function pe(s){const t={occ:"Occupied",vir:"Virtual",closed:"Closed",open:"Open","?":"?"};let a='<table class="result-table"><tr><th>#</th><th>Occupation</th><th>Energy (Hartree)</th><th>Energy (eV)</th></tr>';for(const o of s)a+=`<tr><td>${o.index}</td><td>${t[o.occupation]||o.occupation}</td><td>${o.energy.toFixed(6)}</td><td>${(o.energy*27.2114).toFixed(4)}</td></tr>`;return a+="</table>",a}function Qe(s){return s.replace(/\x1b\[[0-9;]*m/g,"")}function ge(s){return Qe(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}function ne(s){if(s<.01)return"<0.01s";if(s<10)return s.toFixed(2)+"s";if(s<60)return s.toFixed(1)+"s";const t=Math.floor(s/60),a=s%60;return`${t}m ${a.toFixed(0)}s`}class et{overlay;totalTimeEl;titleEl;cancelBtn;t0;totalTimer;steps=new Map;onCancel;constructor(t,a){this.onCancel=a,this.t0=performance.now(),this.overlay=document.createElement("div"),this.overlay.className="pt-overlay";const o=document.createElement("div");o.className="pt-card";const e=document.createElement("div");e.className="pt-title-row",this.titleEl=document.createElement("div"),this.titleEl.className="pt-title",this.titleEl.innerHTML='<span class="pt-title-icon"></span> RUNNING';const i=document.createElement("span");i.className="pt-badge active",i.textContent="GPU",e.appendChild(this.titleEl),e.appendChild(i),o.appendChild(e);const r=document.createElement("div");r.className="pt-steps";for(const m of t){const f=document.createElement("div");f.className="pt-step pending";const b=document.createElement("span");b.className="pt-icon";const x=document.createElement("span");x.className="pt-label",x.textContent=m.label;const d=document.createElement("span");d.className="pt-right";const n=document.createElement("span");n.className="pt-detail";const h=document.createElement("span");h.className="pt-time",d.appendChild(n),d.appendChild(h),f.appendChild(b),f.appendChild(x),f.appendChild(d),r.appendChild(f),this.steps.set(m.id,{el:f,iconEl:b,labelEl:x,detailEl:n,timeEl:h,status:"pending",startTime:0,timerHandle:0})}o.appendChild(r);const l=document.createElement("div");l.className="pt-footer",this.cancelBtn=document.createElement("button"),this.cancelBtn.className="pt-cancel",this.cancelBtn.textContent="Cancel",this.cancelBtn.addEventListener("click",()=>this.onCancel?.()),l.appendChild(this.cancelBtn),this.totalTimeEl=document.createElement("span"),this.totalTimeEl.className="pt-total",l.appendChild(this.totalTimeEl),o.appendChild(l),this.overlay.appendChild(o),document.body.appendChild(this.overlay),requestAnimationFrame(()=>this.overlay.classList.add("visible")),this.totalTimer=window.setInterval(()=>{this.totalTimeEl.textContent=ne((performance.now()-this.t0)/1e3)},100)}startStep(t,a){const o=this.steps.get(t);if(o){if(o.status==="active"){a&&(o.detailEl.textContent=a);return}o.status="active",o.el.className="pt-step active",o.startTime=performance.now(),a&&(o.detailEl.textContent=a),o.timerHandle=window.setInterval(()=>{o.timeEl.textContent=ne((performance.now()-o.startTime)/1e3)},100)}}updateStep(t,a){const o=this.steps.get(t);o&&(o.status==="pending"?this.startStep(t,a):o.detailEl.textContent=a)}completeStep(t,a){const o=this.steps.get(t);!o||o.status==="done"||(clearInterval(o.timerHandle),o.status="done",o.el.className="pt-step done",a&&(o.detailEl.textContent=a),o.startTime>0&&(o.timeEl.textContent=ne((performance.now()-o.startTime)/1e3)))}handleProgress(t){if(t.stage==="setup"){t.iteration===0?this.startStep("setup","Initializing..."):this.completeStep("setup");return}if(t.stage==="integrals"){t.iteration===0?this.startStep("integrals","Computing ERI..."):this.completeStep("integrals");return}if(t.stage==="scf"){const a=t.delta_e??0;t.iteration===0?(this.steps.get("setup")?.status==="active"&&this.completeStep("setup"),this.steps.get("integrals")?.status==="active"&&this.completeStep("integrals"),this.startStep("scf","iter 0")):this.updateStep("scf",`iter ${t.iteration}  ΔE=${a.toExponential(2)}`)}else if(t.stage==="posthf")t.iteration===0?(this.steps.get("scf")?.status==="active"&&this.completeStep("scf"),this.startStep("posthf","Starting...")):this.completeStep("posthf");else if(t.stage==="ccsd")this.steps.get("scf")?.status==="active"&&this.completeStep("scf"),this.updateStep("posthf",`CCSD iter ${t.iteration}  ΔE=${(t.delta_e??0).toExponential(2)}`);else if(t.stage==="ccsd_lambda")this.updateStep("posthf",`Λ iter ${t.iteration}  ‖Δλ‖=${(t.residual??0).toExponential(2)}`);else if(t.stage==="excited"){this.steps.get("scf")?.status==="active"&&this.completeStep("scf"),this.steps.get("posthf")?.status==="active"&&this.completeStep("posthf");const a={0:"MO transform...",1:"Building operator...",2:"Solving eigenstates..."};this.startStep("excited",a[t.iteration]||"Computing...")}else if(t.stage==="davidson")this.steps.get("posthf")?.status==="active"&&this.completeStep("posthf"),this.updateStep("excited",`Davidson iter ${t.iteration}  max|r|=${(t.max_residual??0).toExponential(2)}`);else if(t.stage==="schur")this.updateStep("excited",t.iteration===0?"Schur diagonalization...":"Schur done");else if(t.stage==="schur_omega"){const a=[t.total_energy,t.delta_e,t.correlation_energy].filter(r=>r!==void 0),o=a[0]!==void 0?Math.floor(a[0]):"?",e=a[1]!==void 0?Number(a[1]).toFixed(6):"",i=a[2]!==void 0?Number(a[2]).toExponential(2):"";this.updateStep("excited",`Root ${o} ω=${e} Δω=${i}`)}}complete(){for(const[,t]of this.steps)(t.status==="active"||t.status==="pending")&&(clearInterval(t.timerHandle),t.status="done",t.el.className="pt-step done",t.startTime>0&&(t.timeEl.textContent=ne((performance.now()-t.startTime)/1e3)));this.titleEl.innerHTML='<span class="pt-title-icon" style="background:var(--color-converged,#10b981)"></span> COMPLETE',this.cancelBtn.textContent="Close",this.cancelBtn.onclick=()=>this.close(),setTimeout(()=>this.close(),1500)}fail(t){for(const[,a]of this.steps)(a.status==="active"||a.status==="pending")&&(clearInterval(a.timerHandle),a.status="error",a.el.className="pt-step error");this.titleEl.innerHTML='<span class="pt-title-icon" style="background:var(--color-error,#e74c3c)"></span> ERROR',this.cancelBtn.textContent="Close",this.cancelBtn.onclick=()=>this.close(),setTimeout(()=>this.close(),3e3)}close(){clearInterval(this.totalTimer);for(const t of this.steps.values())clearInterval(t.timerHandle);this.overlay.classList.remove("visible"),this.overlay.addEventListener("transitionend",()=>this.overlay.remove(),{once:!0}),setTimeout(()=>{this.overlay.parentNode&&this.overlay.remove()},500)}}function tt(s,t="stored",a="energy"){const o=[{id:"setup",label:"Setup"}];t==="ri"||t==="RI"?(o.push({id:"ri-2c",label:"RI: 2-Center Integrals"}),o.push({id:"ri-3c",label:"RI: 3-Center Integrals"}),o.push({id:"ri-b",label:"RI: B Matrix"})):o.push({id:"integrals",label:"Integrals"}),o.push({id:"scf",label:"SCF Loop"});const e={mp2:"MP2 Correlation",mp3:"MP3 Correlation",mp4:"MP4 Correlation",cc2:"CC2 Correlation",ccsd:"CCSD Correlation",ccsd_t:"CCSD(T) Correlation",ccsd_density:"CCSD + Lambda",fci:"Full CI"},i={cis:"CIS Excited States",adc2:"ADC(2) Excited States",adc2x:"ADC(2)-x Excited States",eom_mp2:"EOM-MP2 Excited States",eom_cc2:"EOM-CC2 Excited States",eom_ccsd:"EOM-CCSD Excited States"};return s in e&&o.push({id:"posthf",label:e[s]}),s in i&&(s==="eom_ccsd"&&o.push({id:"posthf",label:"CCSD Ground State"}),o.push({id:"excited",label:i[s]})),a==="gradient"?o.push({id:"gradient",label:"Nuclear Gradient"}):a==="hessian"?o.push({id:"hessian",label:"Hessian / Frequencies"}):a==="optimize"&&o.push({id:"optimize",label:"Geometry Optimization"}),o.push({id:"properties",label:"Properties"}),o}async function st(s){let t=["."],a=[{filename:"H2.xyz",name:"H2"},{filename:"H2O.xyz",name:"H2O"}],o=["sto-3g","3-21g","6-31g","cc-pvdz","cc-pvtz"],e=[];try{const[n,h,v,u]=await Promise.all([De(),Se(),Ne(),Ie()]);n.length>0&&(t=n),h.length>0&&(a=h),v.length>0&&(o=v),u.length>0&&(e=u)}catch(n){console.warn("API not available, using defaults:",n)}s.innerHTML=`
    <header>
      <div class="header-top">
        <h1>GANSU</h1>
        <span class="subtitle">GPU Accelerated Numerical Simulation Utility</span>
        <button id="theme-btn" class="icon-btn" title="Toggle theme">
          <span id="theme-icon">&#9790;</span>
        </button>
      </div>
      <nav class="demo-nav">
        <a class="demo-tab active">Calculation</a>
        <a href="./pes.html" class="demo-tab">PES</a>
        <a href="./geomopt.html" class="demo-tab">Geometry Opt</a>
      </nav>
    </header>
    <div class="main-grid">
      <div id="molecule-col"></div>
      <div id="settings-col"></div>
    </div>
    <div class="action-bar">
      <button id="run-btn" class="primary-btn">Run Calculation</button>
      <button id="cancel-btn" class="secondary-btn hidden">Cancel</button>
    </div>
    <div id="results-col"></div>
  `;const i=s.querySelector("#theme-btn"),r=s.querySelector("#theme-icon");i.addEventListener("click",()=>{const n=Re();r.textContent=n==="dark"?"☀":"☾"});const l=Ae(s.querySelector("#molecule-col"),t,a,()=>{}),m=Be(s.querySelector("#settings-col"),o,e),f=Je(s.querySelector("#results-col")),b=s.querySelector("#run-btn"),x=s.querySelector("#cancel-btn");let d=null;b.addEventListener("click",async()=>{const n=l.getXyz(),h=l.getXyzFile();if(!n&&!h){alert("Please enter a molecule (XYZ text or select a sample).");return}const v=m.getParams(),u={...q,...v,xyz_text:n,xyz_file:h,xyz_dir:l.getXyzDir()};b.disabled=!0,x.classList.remove("hidden"),f.hide();const C=tt(u.post_hf_method,u.eri_method,"energy"),M=new et(C,()=>{d&&(d.abort(),d=null),M.close(),b.disabled=!1,x.classList.add("hidden")});d=new AbortController;try{let c=function(R){const g=((performance.now()-I)/1e3).toFixed(3),$=R.stage,y=R.iteration,_=R.values||[];if($==="setup")E.push(`[${g}s] ${y===0?"Setup: Initializing...":"Setup: Core Hamiltonian computed"}`);else if($==="integrals")E.push(`[${g}s] ${y===0?"Integrals: Computing ERIs...":"Integrals: Done"}`);else if($==="integrals_ri"){const p={0:"2-center ERIs",1:"Cholesky",2:"3-center ERIs",3:"B matrix",4:"RI done"};E.push(`[${g}s] RI: ${p[y]||`step ${y}`}`)}else if($==="scf"){const p=_[2]!==void 0?Number(_[2]).toFixed(10):"",k=_[1]!==void 0?Number(_[1]).toExponential(2):"";E.push(`[${g}s] SCF iter ${y}  E=${p}  ΔE=${k}`)}else if($==="posthf")E.push(`[${g}s] ${y===0?"Post-HF: Starting...":"Post-HF: Done"}`);else if($==="ccsd"){const p=_[1]!==void 0?Number(_[1]).toExponential(2):"";E.push(`[${g}s] CCSD iter ${y}  ΔE=${p}`)}else if($==="ccsd_lambda")E.push(`[${g}s] Lambda iter ${y}  residual=${_[0]!==void 0?Number(_[0]).toExponential(2):""}`);else if($==="excited"){const p={0:"MO transform",1:"Building operator",2:"Solving eigenstates"};E.push(`[${g}s] Excited: ${p[y]||`step ${y}`}`)}else if($==="schur")E.push(`[${g}s] Schur ${y===0?"diagonalization...":"done"}`);else if($==="schur_omega"){const p=_[0]!==void 0?Math.floor(Number(_[0])):"?",k=_[1]!==void 0?Number(_[1]).toFixed(8):"",D=_[2]!==void 0?Number(_[2]).toExponential(2):"";E.push(`[${g}s] Schur-omega Root ${p}  omega=${k}  d_omega=${D}`)}else if($==="davidson"){const p=_.length,k=p>0?Number(_[p-1]).toExponential(2):"",D=_.slice(0,p-1).map(G=>Number(G).toFixed(6));E.push(`[${g}s] Davidson iter ${y}  max|r|=${k}  eigs=[${D.join(", ")}]`)}else E.push(`[${g}s] ${$} iter ${y}`)};const T=(await fetch("/api/run/inprocess/stream",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({xyz_text:u.xyz_text,xyz_file:u.xyz_file,xyz_dir:u.xyz_dir,basis:u.basis,method:u.method,charge:u.charge,beta_to_alpha:u.beta_to_alpha,convergence_method:u.convergence_method,diis_size:u.diis_size,damping_factor:u.damping_factor,maxiter:u.maxiter,convergence_energy_threshold:u.convergence_energy_threshold,schwarz_screening_threshold:u.schwarz_screening_threshold,initial_guess:u.initial_guess,post_hf_method:u.post_hf_method,n_excited_states:u.n_excited_states,spin_type:u.spin_type,eri_method:u.eri_method,auxiliary_basis:u.auxiliary_basis,auxiliary_basis_dir:u.auxiliary_basis_dir,excited_solver:u.excited_solver,frozen_core:u.frozen_core,mulliken:u.mulliken,mayer:u.mayer,wiberg:u.wiberg}),signal:d.signal})).body?.getReader();if(!T)return;const H=new TextDecoder;let L="";const z=()=>new Promise(R=>requestAnimationFrame(()=>R()));let w="";const E=[],I=performance.now();for(;;){const{done:R,value:g}=await T.read();if(R)break;L+=H.decode(g,{stream:!0});const $=L.split(`

`);L=$.pop()||"";for(const y of $){const _=y.trim();if(_.startsWith("data: "))try{const p=JSON.parse(_.slice(6));if(p.type==="progress")w&&w!==p.stage&&await z(),w=p.stage,c(p),M.handleProgress({type:"progress",stage:p.stage,iteration:p.iteration,total_energy:p.values?.[2],delta_e:p.values?.[1],correlation_energy:p.values?.[0],residual:p.values?.[0],max_residual:p.values?.[0]});else if(p.type==="result"){const k=((performance.now()-I)/1e3).toFixed(3),D=p.data.summary||{},G=p.data.molecule||{};E.push(`[${k}s] Done. Total energy: ${(D.total_energy??0).toFixed(10)} Ha`),E.push(`  Basis: ${p.data.basis_set?.num_basis??"?"} functions, Occ: ${G.num_occ??"?"}, Vir: ${G.num_vir??"?"}${G.frozen_core?`, Frozen core: ${G.frozen_core}`:""}`),p.data.post_hf&&E.push(`  Post-HF (${p.data.post_hf.method}): correction=${p.data.post_hf.correction.toFixed(10)}, total=${p.data.post_hf.total_energy.toFixed(10)}`),M.complete();const F=p.data;f.show({ok:!0,raw_output:E.join(`
`),molecule:F.molecule||{},basis_set:F.basis_set||{},scf_iterations:F.scf_iterations||[],summary:F.summary||{},post_hf:F.post_hf||void 0,orbital_energies:F.orbital_energies||[],orbital_energies_beta:F.orbital_energies_beta||[],mulliken:F.mulliken||[],mayer_bond_order:F.mayer_bond_order||[],wiberg_bond_order:F.wiberg_bond_order||[],timing:F.timing||{},excited_states:F.excited_states,excited_states_method:F.excited_states_method,excited_states_spin:F.excited_states_spin})}else p.type==="error"&&(E.push(`ERROR: ${p.error}`),M.fail(p.error),f.showError(p.error,E.join(`
`)))}catch{}}}}catch(c){c instanceof Error&&c.name!=="AbortError"&&M.fail(String(c))}b.disabled=!1,x.classList.add("hidden"),d=null}),x.addEventListener("click",()=>{d&&(d.abort(),d=null),b.disabled=!1,x.classList.add("hidden")})}Pe();const _e=document.querySelector("#app");_e&&st(_e);
