import{r as de,g as le,t as Re,i as Pe}from"./styles-CDRzXZyU.js";import{f as Se,a as qe,b as De,c as Ie,d as Ne}from"./api-BlpZUjmm.js";const P={xyz_text:"",xyz_file:"",xyz_dir:".",basis:"sto-3g",method:"RHF",charge:0,beta_to_alpha:0,convergence_method:"diis",diis_size:8,diis_include_transform:!1,damping_factor:.9,rohf_parameter_name:"Roothaan",maxiter:100,convergence_energy_threshold:1e-6,schwarz_screening_threshold:1e-12,initial_guess:"core",post_hf_method:"none",n_excited_states:5,spin_type:"singlet",excited_solver:"auto",eri_method:"stored",auxiliary_basis:"",auxiliary_basis_dir:"auxiliary_basis",mulliken:!1,mayer:!1,wiberg:!1,export_molden:!1,verbose:!1,timeout:600};function Ae(s,t,a,o){let e="",i=t[0]||".";s.innerHTML=`
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
  `;const r=s.querySelector("#sample-select"),l=s.querySelector("#xyz-input"),h=s.querySelector("#mol-preview"),m=s.querySelector("#dir-group");m&&m.addEventListener("click",async d=>{const n=d.target.closest(".toggle");if(!(!n||!n.dataset.value)){m.querySelectorAll(".toggle").forEach(v=>v.classList.remove("active")),n.classList.add("active"),i=n.dataset.value,r.innerHTML='<option value="">-- Loading... --</option>';try{const v=await Se(i);r.innerHTML='<option value="">-- Custom input --</option>'+v.map(b=>`<option value="${b.filename}">${b.name} (${b.filename})</option>`).join("")}catch{r.innerHTML='<option value="">-- Custom input --</option>'}e="",h.innerHTML=""}});let f;function x(){clearTimeout(f),f=window.setTimeout(()=>{const d=l.value.trim();d?de(h,d,[]):h.innerHTML=""},300)}return r.addEventListener("change",()=>{e=r.value,e?(l.value="",l.placeholder=`Using sample: ${e}`,qe(e,i).then(d=>{d&&r.value===e&&de(h,d,[])}).catch(()=>{h.innerHTML='<p style="color:var(--color-text-dim)">Sample file selected</p>'})):(l.placeholder=`2

H  0.0 0.0 0.0
H  0.0 0.0 0.74`,h.innerHTML=""),o(l.value)}),l.addEventListener("input",()=>{l.value.trim()&&(r.value="",e=""),x()}),l.addEventListener("dragover",d=>{d.preventDefault(),l.classList.add("drag-over")}),l.addEventListener("dragleave",()=>l.classList.remove("drag-over")),l.addEventListener("drop",d=>{d.preventDefault(),l.classList.remove("drag-over");const n=d.dataTransfer?.files[0];if(n&&n.name.endsWith(".xyz")){const v=new FileReader;v.onload=()=>{l.value=v.result,r.value="",e="",x()},v.readAsText(n)}}),{getXyz:()=>l.value.trim(),getXyzFile:()=>e,getXyzDir:()=>i,setXyz:d=>{l.value=d,e="";const n=s.querySelector("#mol-preview");n&&de(n,d,[])}}}function Be(s,t,a=[]){s.innerHTML=`
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
  `,s.querySelectorAll(".toggle-group").forEach(c=>{c.addEventListener("click",S=>{const T=S.target.closest(".toggle");T&&(c.querySelectorAll(".toggle").forEach(z=>z.classList.remove("active")),T.classList.add("active"))})});const o=s.querySelector("#conv-group"),e=s.querySelector("#diis-params"),i=s.querySelector("#damping-params");o.addEventListener("click",()=>{const S=o.querySelector(".active")?.dataset.value;e.classList.toggle("hidden",S!=="diis"),i.classList.toggle("hidden",S!=="damping")});const r=s.querySelector("#method-group"),l=s.querySelector("#rohf-param-row");r.addEventListener("click",()=>{const c=r.querySelector(".active");l.classList.toggle("hidden",c?.dataset.value!=="ROHF")});const h=s.querySelector("#eri-group"),m=s.querySelector("#aux-basis-row");h.addEventListener("click",()=>{const S=h.querySelector(".active")?.dataset.value;m.classList.toggle("hidden",S!=="ri"&&S!=="direct_ri")});const f=s.querySelector("#posthf-group"),x=s.querySelector("#excited-group"),d=s.querySelector("#excited-params");f.addEventListener("click",()=>{const c=f.querySelector(".active");c?.dataset.value&&c.dataset.value!=="none"&&(x.querySelectorAll(".toggle").forEach(S=>S.classList.remove("active")),x.querySelector('[data-value="none"]')?.classList.add("active"),d.classList.add("hidden"))});const n=s.querySelector("#solver-row"),v=new Set(["adc2","adc2x","eom_mp2","eom_cc2"]);x.addEventListener("click",()=>{const S=x.querySelector(".active")?.dataset.value||"none";S!=="none"&&(f.querySelectorAll(".toggle").forEach(T=>T.classList.remove("active")),f.querySelector('[data-value="none"]')?.classList.add("active")),d.classList.toggle("hidden",S==="none"),n.classList.toggle("hidden",!v.has(S))});function b(c){return s.querySelector(`#${c} .toggle.active`)?.dataset.value||""}function u(c){return s.querySelector(`#${c}`)?.checked||!1}function C(c,S){const T=parseFloat(s.querySelector(`#${c}`)?.value||"");return isNaN(T)?S:T}return{getParams:()=>({method:b("method-group")||P.method,basis:s.querySelector("#basis-select")?.value||P.basis,charge:parseInt(b("charge-group")||"0",10),beta_to_alpha:parseInt(b("mult-group")||"0",10),initial_guess:b("guess-group")||P.initial_guess,convergence_method:b("conv-group")||P.convergence_method,diis_size:C("diis-size",P.diis_size),diis_include_transform:u("chk-diis-transform"),damping_factor:C("damping-factor",P.damping_factor),rohf_parameter_name:s.querySelector("#rohf-param-select")?.value||"Roothaan",eri_method:b("eri-group")||P.eri_method,auxiliary_basis:s.querySelector("#aux-basis-select")?.value||"",auxiliary_basis_dir:s.querySelector("#aux-basis-select")?.selectedOptions[0]?.dataset.dir||"auxiliary_basis",post_hf_method:b("excited-group")!=="none"?b("excited-group"):b("posthf-group")||P.post_hf_method,n_excited_states:C("n-excited-states",P.n_excited_states),spin_type:b("spin-type-group")||P.spin_type,excited_solver:b("solver-group")||"auto",mulliken:u("chk-mulliken"),mayer:u("chk-mayer"),wiberg:u("chk-wiberg"),export_molden:u("chk-molden"),maxiter:C("maxiter",P.maxiter),convergence_energy_threshold:C("conv-thresh",P.convergence_energy_threshold),schwarz_screening_threshold:C("schwarz-thresh",P.schwarz_screening_threshold),timeout:C("timeout",P.timeout)}),setParams:c=>{if(c.method&&M("method-group",c.method),c.basis){const S=s.querySelector("#basis-select");S&&(S.value=c.basis)}if(c.post_hf_method&&(["cis","adc2","adc2x","eom_mp2","eom_cc2","eom_ccsd"].includes(c.post_hf_method)?(M("posthf-group","none"),M("excited-group",c.post_hf_method)):(M("posthf-group",c.post_hf_method),M("excited-group","none"))),c.initial_guess&&M("guess-group",c.initial_guess),c.eri_method&&M("eri-group",c.eri_method),c.n_excited_states){const S=s.querySelector("#n-excited-states");S&&(S.value=c.n_excited_states)}}};function M(c,S){s.querySelectorAll(`[data-group="${c}"]`).forEach(z=>{z.classList.toggle("active",z.dataset.value===S)})}}function Ge(s,t,a){if(t.length<2){s.innerHTML='<p style="color:var(--color-text-dim);font-size:0.85rem;">Not enough iterations</p>';return}const o=t.filter(w=>w.iter>=1&&w.deltaE!==0).map(w=>({x:w.iter,y:Math.log10(Math.abs(w.deltaE))}));if(o.length<2){s.innerHTML='<p style="color:var(--color-text-dim);font-size:0.85rem;">No data</p>';return}const e=le(),i=320,r=200,l=52,h=12,m=24,f=32,x=i-l-h,d=r-m-f,n=o[0].x,v=o[o.length-1].x,b=Math.min(...o.map(w=>w.y),Math.log10(a))-1,u=Math.max(o[0].y,0)+.5,C=w=>l+(w-n)/(v-n||1)*x,M=w=>m+d-(w-b)/(u-b||1)*d;let c=`<svg width="${i}" height="${r}" xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;height:auto;" viewBox="0 0 ${i} ${r}">`;const S=2,T=Math.ceil(b/S)*S;for(let w=T;w<=u;w+=S){const E=M(w);E<m||E>m+d||(c+=`<line x1="${l}" y1="${E}" x2="${l+x}" y2="${E}" stroke="${e.grid}" stroke-width="0.5"/>`,c+=`<text x="${l-4}" y="${E+3}" text-anchor="end" font-size="9" fill="${e.dim}">1e${w}</text>`)}c+=`<line x1="${l}" y1="${m}" x2="${l}" y2="${m+d}" stroke="${e.axis}" stroke-width="1"/>`,c+=`<line x1="${l}" y1="${m+d}" x2="${l+x}" y2="${m+d}" stroke="${e.axis}" stroke-width="1"/>`;const z=Math.max(1,Math.round((v-n)/5));for(let w=n;w<=v;w+=z){const E=C(w);c+=`<text x="${E}" y="${m+d+14}" text-anchor="middle" font-size="9" fill="${e.dim}">${w}</text>`}const L=M(Math.log10(a));L>=m&&L<=m+d&&(c+=`<line x1="${l}" y1="${L}" x2="${l+x}" y2="${L}" stroke="${e.error}" stroke-width="1" stroke-dasharray="4,3"/>`,c+=`<text x="${l+x+2}" y="${L+3}" font-size="8" fill="${e.error}">${a.toExponential(0)}</text>`);let H="";for(let w=0;w<o.length;w++){const E=C(o[w].x),N=M(o[w].y);H+=w===0?`M${E},${N}`:` L${E},${N}`}c+=`<path d="${H}" fill="none" stroke="${e.accent}" stroke-width="1.5"/>`;for(const w of o)c+=`<circle cx="${C(w.x)}" cy="${M(w.y)}" r="2.5" fill="${e.accent}"/>`;c+=`<text x="${i/2}" y="14" text-anchor="middle" font-size="11" fill="${e.titleSvg}">SCF Convergence</text>`,c+=`<text x="${l+x/2}" y="${r-2}" text-anchor="middle" font-size="9" fill="${e.dim}">Iteration</text>`,c+=`<text x="12" y="${m+d/2}" text-anchor="middle" font-size="9" fill="${e.dim}" transform="rotate(-90,12,${m+d/2})">log10(|deltaE|)</text>`,c+="</svg>",s.innerHTML=`<h3>SCF Convergence</h3>${c}`}const we=12,je=1e-4,fe=6;function he(s){const t=[];let a=0;for(;a<s.length;){const o=[a],e=[s[a].occupation],i=s[a].energy;for(;a+1<s.length&&Math.abs(s[a+1].energy-i)<je;)a++,o.push(a),e.push(s[a].occupation);const r=o.reduce((l,h)=>l+s[h].energy,0)/o.length;t.push({indices:o,energy:r,occupations:e}),a++}return t}function oe(s,t,a){return s+t*a+(t-1)*fe}function Ee(s,t){const a=s.slice(),o=a.map((e,i)=>i);o.sort((e,i)=>a[e]-a[i]);for(let e=1;e<o.length;e++){const i=o[e-1],r=o[e];a[r]-a[i]<t&&(a[r]=a[i]+t)}return a}function Me(s,t,a){let o=2;const e=12,i=.5,r=Math.min(a,520),l=document.createElement("div");l.style.position="relative";const h=document.createElement("div");h.style.cssText="display:flex;align-items:center;justify-content:center;gap:6px;padding:2px 0 4px;font-size:11px;color:var(--color-text-dim,#888)";const m="width:22px;height:22px;border:1px solid var(--color-border,#ccc);border-radius:4px;background:var(--color-surface,#fff);color:var(--color-text,#333);cursor:pointer;font-size:14px;line-height:1;display:flex;align-items:center;justify-content:center",f=document.createElement("button");f.style.cssText=m,f.textContent="−",f.title="Zoom out";const x=document.createElement("button");x.style.cssText=m,x.textContent="+",x.title="Zoom in";const d=document.createElement("span");d.style.cssText="min-width:40px;text-align:center;font-size:10px",h.appendChild(f),h.appendChild(d),h.appendChild(x),l.appendChild(h);const n=document.createElement("div");n.style.overflowY="auto",n.style.maxHeight=`${r}px`,n.style.position="relative",l.appendChild(n);function v(u){if(u=Math.max(i,Math.min(e,u)),u===o)return;const C=n.scrollHeight>r?n.scrollTop/(n.scrollHeight-r):0;o=u,b();const M=n.scrollHeight-r;M>0&&(n.scrollTop=C*M)}function b(){n.innerHTML=t(o),d.textContent=`${Math.round(o*100)}%`,f.disabled=o<=i,x.disabled=o>=e}n.addEventListener("wheel",u=>{if(!u.ctrlKey&&!u.metaKey)return;u.preventDefault();const C=u.deltaY<0?1.2:1/1.2;v(o*C)},{passive:!1}),x.addEventListener("click",()=>v(o*1.3)),f.addEventListener("click",()=>v(o/1.3)),b(),s.appendChild(l)}function Ue(s,t,a){let i=`<polygon points="${s-3.5-3},${t-2} ${s-3.5},${t-10} ${s-3.5+3},${t-2}" fill="${a}"/>`;return i+=`<polygon points="${s+3.5-3},${t-10} ${s+3.5},${t-2} ${s+3.5+3},${t-10}" fill="${a}"/>`,i}function Ce(s,t,a){return`<polygon points="${s-3},${t-2} ${s},${t-10} ${s+3},${t-2}" fill="${a}"/>`}function Ye(s,t,a){return`<polygon points="${s-3},${t-10} ${s},${t-2} ${s+3},${t-10}" fill="${a}"/>`}function We(s,t,a,o,e){const i=(t-s)*27.2114;if(Math.abs(i)<.01)return"";const r=a(s),l=a(t);let h=`<line x1="${o}" y1="${r}" x2="${o}" y2="${l}" stroke="${e.gap}" stroke-width="1.5" stroke-dasharray="3,2"/>`;h+=`<polygon points="${o-3},${r-4} ${o},${r} ${o+3},${r-4}" fill="${e.gap}"/>`,h+=`<polygon points="${o-3},${l+4} ${o},${l} ${o+3},${l+4}" fill="${e.gap}"/>`;const m=(r+l)/2;return h+=`<text x="${o+6}" y="${m+4}" font-size="10" fill="${e.gap}">${i.toFixed(2)} eV</text>`,h}function ve(s){return s==="occ"||s==="closed"||s==="open"}function me(s){for(let t=s.length-1;t>=0;t--)if(ve(s[t].occupation))return t;return-1}function Ve(s,t){const a=t.length;if(a===0)return;const o=he(t),e=me(t),i=e+1<a?e+1:-1,r=28,l=24,h=70,m=40,f=Math.max(...o.map(z=>z.indices.length)),x=oe(h,f,m),d=Math.max(300,x+100),n=t[0].energy,v=t[a-1].energy,u=(v-n||1)*.1,C=n-u,M=v+u,c=Math.max(180,Math.min(480,o.length*40));function S(z){const L=le(),H=c*z,w=H+r+l,E=y=>r+H-(y-C)/(M-C)*H,N=o.map(y=>E(y.energy)),O=Ee(N,we);let g=`<svg width="${d}" height="${w}" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0 auto;">`;g+=`<line x1="${h-8}" y1="${r}" x2="${h-8}" y2="${r+H}" stroke="${L.axis}" stroke-width="1"/>`;for(let y=0;y<o.length;y++){const $=o[y],_=E($.energy),p=O[y],k=$.indices.length,q=oe(h,k,m),F=e>=0&&$.indices.includes(e),ae=i>=0&&$.indices.includes(i);for(let D=0;D<k;D++){const J=$.indices[D],U=t[J].occupation,Q=ve(U),A=J===e||J===i,ee=Q?L.occupied:L.virtual,te=A?2.5:1.5,G=h+D*(m+fe);g+=`<line x1="${G}" y1="${_}" x2="${G+m}" y2="${_}" stroke="${ee}" stroke-width="${te}"/>`,U==="occ"||U==="closed"?g+=Ue(G+m/2,_,ee):U==="open"&&(g+=Ce(G+m/2,_,ee))}Math.abs(p-_)>3&&(g+=`<line x1="${h-10}" y1="${_}" x2="${h-14}" y2="${p}" stroke="${L.leader}" stroke-width="0.5"/>`);const R=k>1?` (×${k})`:"";g+=`<text x="${h-16}" y="${p+4}" text-anchor="end" font-size="10" fill="${L.label}">${$.energy.toFixed(3)}${R}</text>`;const X=$.occupations.some(D=>D==="open");if(F){const D=X?"SOMO":"HOMO";g+=`<text x="${q+6}" y="${_+4}" font-size="11" fill="${L.occupied}" font-weight="bold">${D}</text>`}else X&&(g+=`<text x="${q+6}" y="${_+4}" font-size="10" fill="${L.alpha}" font-weight="bold">SOMO</text>`);ae&&!F&&(g+=`<text x="${q+6}" y="${_+4}" font-size="11" fill="${L.dim}" font-weight="bold">LUMO</text>`)}return e>=0&&i>=0&&(g+=We(t[e].energy,t[i].energy,E,x+50,L)),g+=`<text x="${d/2}" y="14" text-anchor="middle" font-size="10" fill="${L.hint}">Ctrl+Scroll or +/− to zoom</text>`,g+="</svg>",g}s.innerHTML="";const T=c+r+l;Me(s,S,T)}function Ze(s,t,a){const o=t.length,e=a.length;if(o===0&&e===0)return;const i=he(t),r=he(a),l=me(t),h=l+1<o?l+1:-1,m=me(a),f=m+1<e?m+1:-1,x=28,d=24,n=60,v=36,b=50,u=Math.max(1,...i.map(p=>p.indices.length)),C=Math.max(1,...r.map(p=>p.indices.length)),M=oe(n,u,v),c=M+b,S=oe(c,C,v),T=Math.max(420,S+80),z=[];for(const p of t)z.push(p.energy);for(const p of a)z.push(p.energy);z.sort((p,k)=>p-k);const L=z[0],H=z[z.length-1],E=(H-L||1)*.1,N=L-E,O=H+E,g=Math.max(i.length,r.length),y=Math.max(180,Math.min(480,g*40));function $(p){const k=le(),q=y*p,F=q+x+d,ae=D=>x+q-(D-N)/(O-N)*q;let R=`<svg width="${T}" height="${F}" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0 auto;">`;R+=`<text x="${T/2}" y="14" text-anchor="middle" font-size="10" fill="${k.hint}">Ctrl+Scroll or +/− to zoom</text>`;function X(D,J,U,Q,A,ee,te,G,ke){const Le=(A+ee)/2;R+=`<text x="${Le}" y="${x-4}" text-anchor="middle" font-size="10" fill="${te}" font-weight="bold">${ke}</text>`,R+=`<line x1="${A-8}" y1="${x}" x2="${A-8}" y2="${x+q}" stroke="${k.axis}" stroke-width="1"/>`;const He=D.map(V=>ae(V.energy)),ze=Ee(He,we);for(let V=0;V<D.length;V++){const Y=D[V],j=ae(Y.energy),Z=ze[V],K=Y.indices.length,se=oe(A,K,v),xe=U>=0&&Y.indices.includes(U),Te=Q>=0&&Y.indices.includes(Q);for(let B=0;B<K;B++){const W=Y.indices[B],be=ve(J[W].occupation),Fe=W===U||W===Q,re=be?te:k.virtual,Oe=Fe?2.5:1.5,ce=A+B*(v+fe);if(R+=`<line x1="${ce}" y1="${j}" x2="${ce+v}" y2="${j}" stroke="${re}" stroke-width="${Oe}"/>`,be){const ye=ce+v/2;R+=G?Ce(ye,j,re):Ye(ye,j,re)}}if(G){Math.abs(Z-j)>3&&(R+=`<line x1="${A-10}" y1="${j}" x2="${A-14}" y2="${Z}" stroke="${k.leader}" stroke-width="0.5"/>`);const B=K>1?` (×${K})`:"";R+=`<text x="${A-16}" y="${Z+4}" text-anchor="end" font-size="9" fill="${k.label}">${Y.energy.toFixed(3)}${B}</text>`}else{Math.abs(Z-j)>3&&(R+=`<line x1="${se+2}" y1="${j}" x2="${se+6}" y2="${Z}" stroke="${k.leader}" stroke-width="0.5"/>`);const B=K>1?` (×${K})`:"";R+=`<text x="${se+8}" y="${Z+4}" text-anchor="start" font-size="9" fill="${k.label}">${Y.energy.toFixed(3)}${B}</text>`}if(xe){const B=G?se+4:A-12,W=G?"start":"end";R+=`<text x="${B}" y="${j+4}" text-anchor="${W}" font-size="9" fill="${te}" font-weight="bold">HOMO</text>`}if(Te&&!xe){const B=G?se+4:A-12,W=G?"start":"end";R+=`<text x="${B}" y="${j+4}" text-anchor="${W}" font-size="9" fill="${k.dim}" font-weight="bold">LUMO</text>`}}}return X(i,t,l,h,n,M,k.alpha,!0,"α"),X(r,a,m,f,c,S,k.beta,!1,"β"),R+="</svg>",R}s.innerHTML="";const _=y+x+d;Me(s,$,_)}function Ke(s,t,a,o){if(!t||t.length===0){s.innerHTML='<p style="color:var(--color-text-dim);font-size:0.85rem;">No excited state data</p>';return}const e=le(),i=o==="triplet",r=700,l=280,h=48,m=16,f=28,x=38,d=r-h-m,n=l-f-x,v=t.map(g=>g.energy_ev),b=Math.max(0,Math.min(...v)-2),u=Math.max(...v)+2,C=.4,M=200,c=(u-b)/M,S=new Float64Array(M);for(let g=0;g<M;g++){const y=b+g*c;let $=0;for(const _ of t){const p=_.osc_strength;if(p>0){const k=y-_.energy_ev;$+=p*Math.exp(-k*k/(2*C*C))}}S[g]=$}const T=Math.max(...S)*1.15||1,z=g=>h+(g-b)/(u-b||1)*d,L=g=>f+n-g/T*n;let H=`<svg width="${r}" height="${l}" xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;height:auto;" viewBox="0 0 ${r} ${l}">`;const w=Math.ceil((u-b)/6);for(let g=Math.ceil(b);g<=u;g+=Math.max(1,w)){const y=z(g);y<h||y>h+d||(H+=`<line x1="${y}" y1="${f}" x2="${y}" y2="${f+n}" stroke="${e.grid}" stroke-width="0.5"/>`,H+=`<text x="${y}" y="${f+n+14}" text-anchor="middle" font-size="9" fill="${e.dim}">${g}</text>`)}if(H+=`<line x1="${h}" y1="${f}" x2="${h}" y2="${f+n}" stroke="${e.axis}" stroke-width="1"/>`,H+=`<line x1="${h}" y1="${f+n}" x2="${h+d}" y2="${f+n}" stroke="${e.axis}" stroke-width="1"/>`,!i){let g=`M${h},${f+n}`;for(let $=0;$<M;$++)g+=` L${z(b+$*c).toFixed(1)},${L(S[$]).toFixed(1)}`;g+=` L${z(u).toFixed(1)},${f+n} Z`,H+=`<path d="${g}" fill="${e.accent}" fill-opacity="0.15" stroke="none"/>`;let y="";for(let $=0;$<M;$++){const _=z(b+$*c),p=L(S[$]);y+=$===0?`M${_.toFixed(1)},${p.toFixed(1)}`:` L${_.toFixed(1)},${p.toFixed(1)}`}H+=`<path d="${y}" fill="none" stroke="${e.accent}" stroke-width="1.5"/>`}for(const g of t){const y=z(g.energy_ev);if(y<h||y>h+d||!i&&g.osc_strength<=0)continue;const $=i?n*.5:g.osc_strength/T*n,_=f+n-$,p=i?e.error:e.accent;H+=`<line x1="${y.toFixed(1)}" y1="${(f+n).toFixed(1)}" x2="${y.toFixed(1)}" y2="${_.toFixed(1)}" stroke="${p}" stroke-width="1.5" stroke-opacity="0.7"/>`,H+=`<circle cx="${y.toFixed(1)}" cy="${_.toFixed(1)}" r="2" fill="${p}"/>`}const E=i?"Triplet":"Singlet",N=`${a} ${E} Absorption Spectrum`;H+=`<text x="${r/2}" y="16" text-anchor="middle" font-size="11" fill="${e.titleSvg}">${N}</text>`,H+=`<text x="${h+d/2}" y="${l-4}" text-anchor="middle" font-size="9" fill="${e.dim}">Energy (eV)</text>`,i||(H+=`<text x="12" y="${f+n/2}" text-anchor="middle" font-size="9" fill="${e.dim}" transform="rotate(-90,12,${f+n/2})">Osc. Strength (arb.)</text>`),H+="</svg>";let O='<table class="result-table"><tr><th>State</th><th>Energy (Ha)</th><th>Energy (eV)</th><th>f</th><th>Transitions</th></tr>';for(const g of t)O+=`<tr>
      <td>${g.state}</td>
      <td>${g.energy_ha.toFixed(6)}</td>
      <td>${g.energy_ev.toFixed(3)}</td>
      <td>${g.osc_strength.toFixed(4)}</td>
      <td style="font-size:0.8rem">${Xe(g.transitions)}</td>
    </tr>`;O+="</table>",s.innerHTML=`<h3>Excited States (${a} ${E})</h3>${H}${O}`}function Xe(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}function Je(s){s.innerHTML='<div id="results-container"></div>';const t=s.querySelector("#results-container");function a(e){let i='<div class="results">';const r=e.molecule,l=e.basis_set;if((r.num_atoms!==void 0||l.num_basis!==void 0)&&(i+='<div class="summary-row">',r.num_atoms!==void 0&&(i+=`
          <div class="panel result-card">
            <h3>Molecule</h3>
            <table class="result-table">
              ${I("Atoms",String(r.num_atoms))}
              ${r.num_electrons!==void 0?I("Electrons",String(r.num_electrons)):""}
              ${r.alpha_electrons!==void 0?ue("Alpha",String(r.alpha_electrons)):""}
              ${r.beta_electrons!==void 0?ue("Beta",String(r.beta_electrons)):""}
            </table>
          </div>`),l.num_basis!==void 0&&(i+=`
          <div class="panel result-card">
            <h3>Basis Set</h3>
            <table class="result-table">
              ${I("Basis Functions",String(l.num_basis))}
              ${l.num_primitives!==void 0?ue("Primitives",String(l.num_primitives)):""}
              ${l.num_auxiliary!==void 0?I("Auxiliary",String(l.num_auxiliary)):""}
            </table>
          </div>`),i+="</div>"),e.summary.total_energy!==void 0&&(i+=`
        <div class="panel result-card">
          <h3>Energy Summary</h3>
          <table class="result-table">
            ${I("Method",e.summary.method||"-")}
            ${I("Total Energy",ie(e.summary.total_energy)+" Hartree")}
            ${e.summary.electronic_energy!==void 0?I("Electronic Energy",ie(e.summary.electronic_energy)+" Hartree"):""}
            ${e.summary.iterations!==void 0?I("SCF Iterations",String(e.summary.iterations)):""}
            ${e.summary.convergence_algorithm?I("Convergence",e.summary.convergence_algorithm):""}
            ${e.summary.initial_guess?I("Initial Guess",e.summary.initial_guess):""}
            ${e.summary.energy_difference!==void 0?I("Final |deltaE|",e.summary.energy_difference.toExponential(2)):""}
            ${e.summary.computing_time_ms!==void 0?I("Computing Time",e.summary.computing_time_ms.toFixed(1)+" ms"):""}
          </table>
        </div>`),e.post_hf&&(e.post_hf.correction!==void 0||e.post_hf.total_energy!==void 0)&&(i+=`
        <div class="panel result-card">
          <h3>Post-HF: ${e.post_hf.method||""}</h3>
          <table class="result-table">
            ${e.post_hf.correction!==void 0?I("Correlation Energy",ie(e.post_hf.correction)+" Hartree"):""}
            ${e.post_hf.total_energy!==void 0?I("Total Energy",ie(e.post_hf.total_energy)+" Hartree"):""}
          </table>
        </div>`),e.excited_states&&e.excited_states.length>0&&(i+='<div class="panel result-card full-width" id="spectrum-chart-container"></div>'),e.orbital_energies.length>0){const n=e.orbital_energies_beta.length>0;i+='<div class="panel result-card orbital-card" id="orbital-diagram-container"><h3>Orbital Energies</h3></div>',n?i+=`
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
          </details>`}const h=e.molecule.atoms||[],m=n=>{const v=h[n];return v?`${v.element}${n+1}`:String(n)};if(e.mulliken.length>0&&(i+=`
        <div class="panel result-card">
          <h3>Mulliken Population</h3>
          <table class="result-table">
            <tr><th>Atom</th><th>Charge</th></tr>
            ${e.mulliken.map((n,v)=>`<tr><td>${m(v)}</td><td>${n.charge.toFixed(6)}</td></tr>`).join("")}
          </table>
        </div>`),e.mayer_bond_order.length>0&&(i+=`
        <div class="panel result-card">
          <h3>Mayer Bond Order</h3>
          ${$e(e.mayer_bond_order,m)}
        </div>`),e.wiberg_bond_order.length>0&&(i+=`
        <div class="panel result-card">
          <h3>Wiberg Bond Order</h3>
          ${$e(e.wiberg_bond_order,m)}
        </div>`),i+='<div class="panel result-card" id="conv-graph-container"></div>',e.molden_content&&(i+=`
        <div class="panel result-card">
          <h3>Molden</h3>
          <button class="secondary-btn" id="download-molden">Download output.molden</button>
        </div>`),i+=`
      <details class="panel result-card">
        <summary>Raw Output</summary>
        <pre class="raw-output">${ge(e.raw_output)}</pre>
      </details>`,i+="</div>",t.innerHTML=i,e.molden_content){const n=()=>{const b=new Blob([e.molden_content],{type:"chemical/x-molden"}),u=URL.createObjectURL(b),C=document.createElement("a");C.href=u,C.download="output.molden",C.click(),URL.revokeObjectURL(u)};n(),t.querySelector("#download-molden")?.addEventListener("click",n)}const f=t.querySelector("#orbital-diagram-container");f&&e.orbital_energies.length>0&&(e.orbital_energies_beta.length>0?Ze(f,e.orbital_energies,e.orbital_energies_beta):Ve(f,e.orbital_energies));const x=t.querySelector("#conv-graph-container");if(x&&e.scf_iterations.length>0){const n=e.scf_iterations.filter(b=>b.delta_e!==void 0).map(b=>({iter:b.iteration,deltaE:b.delta_e})),v=e.summary.convergence_criterion||1e-6;Ge(x,n,v)}const d=t.querySelector("#spectrum-chart-container");d&&e.excited_states&&e.excited_states.length>0&&Ke(d,e.excited_states,e.excited_states_method||"",e.excited_states_spin||"singlet")}function o(e,i){let r='<div class="results">';r+=`
      <div class="panel result-card error-card">
        <h3>Error</h3>
        <pre class="error-output">${ge(e)}</pre>
      </div>`,i&&(r+=`
        <details class="panel result-card" open>
          <summary>Output</summary>
          <pre class="raw-output">${ge(i)}</pre>
        </details>`),r+="</div>",t.innerHTML=r}return{show:a,showError:o,hide:()=>{t.innerHTML=""}}}function I(s,t){return`<tr><td class="label">${s}</td><td class="value">${t}</td></tr>`}function ue(s,t){return`<tr><td class="label sub-label">${s}</td><td class="value">${t}</td></tr>`}function ie(s){return s.toFixed(10)}function $e(s,t){if(s.length===0)return"";const a=s.length;let o='<table class="result-table bond-table"><tr><th></th>';for(let e=0;e<a;e++)o+=`<th>${t(e)}</th>`;o+="</tr>";for(let e=0;e<a;e++){o+=`<tr><th>${t(e)}</th>`;for(let i=0;i<s[e].length;i++){const r=s[e][i],l=r>.5?"bond-strong":"";o+=`<td class="${l}">${r.toFixed(3)}</td>`}o+="</tr>"}return o+="</table>",o}function pe(s){const t={occ:"Occupied",vir:"Virtual",closed:"Closed",open:"Open","?":"?"};let a='<table class="result-table"><tr><th>#</th><th>Occupation</th><th>Energy (Hartree)</th><th>Energy (eV)</th></tr>';for(const o of s)a+=`<tr><td>${o.index}</td><td>${t[o.occupation]||o.occupation}</td><td>${o.energy.toFixed(6)}</td><td>${(o.energy*27.2114).toFixed(4)}</td></tr>`;return a+="</table>",a}function Qe(s){return s.replace(/\x1b\[[0-9;]*m/g,"")}function ge(s){return Qe(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}function ne(s){if(s<.01)return"<0.01s";if(s<10)return s.toFixed(2)+"s";if(s<60)return s.toFixed(1)+"s";const t=Math.floor(s/60),a=s%60;return`${t}m ${a.toFixed(0)}s`}class et{overlay;totalTimeEl;titleEl;cancelBtn;t0;totalTimer;steps=new Map;onCancel;constructor(t,a){this.onCancel=a,this.t0=performance.now(),this.overlay=document.createElement("div"),this.overlay.className="pt-overlay";const o=document.createElement("div");o.className="pt-card";const e=document.createElement("div");e.className="pt-title-row",this.titleEl=document.createElement("div"),this.titleEl.className="pt-title",this.titleEl.innerHTML='<span class="pt-title-icon"></span> RUNNING';const i=document.createElement("span");i.className="pt-badge active",i.textContent="GPU",e.appendChild(this.titleEl),e.appendChild(i),o.appendChild(e);const r=document.createElement("div");r.className="pt-steps";for(const h of t){const m=document.createElement("div");m.className="pt-step pending";const f=document.createElement("span");f.className="pt-icon";const x=document.createElement("span");x.className="pt-label",x.textContent=h.label;const d=document.createElement("span");d.className="pt-right";const n=document.createElement("span");n.className="pt-detail";const v=document.createElement("span");v.className="pt-time",d.appendChild(n),d.appendChild(v),m.appendChild(f),m.appendChild(x),m.appendChild(d),r.appendChild(m),this.steps.set(h.id,{el:m,iconEl:f,labelEl:x,detailEl:n,timeEl:v,status:"pending",startTime:0,timerHandle:0})}o.appendChild(r);const l=document.createElement("div");l.className="pt-footer",this.cancelBtn=document.createElement("button"),this.cancelBtn.className="pt-cancel",this.cancelBtn.textContent="Cancel",this.cancelBtn.addEventListener("click",()=>this.onCancel?.()),l.appendChild(this.cancelBtn),this.totalTimeEl=document.createElement("span"),this.totalTimeEl.className="pt-total",l.appendChild(this.totalTimeEl),o.appendChild(l),this.overlay.appendChild(o),document.body.appendChild(this.overlay),requestAnimationFrame(()=>this.overlay.classList.add("visible")),this.totalTimer=window.setInterval(()=>{this.totalTimeEl.textContent=ne((performance.now()-this.t0)/1e3)},100)}startStep(t,a){const o=this.steps.get(t);if(o){if(o.status==="active"){a&&(o.detailEl.textContent=a);return}o.status="active",o.el.className="pt-step active",o.startTime=performance.now(),a&&(o.detailEl.textContent=a),o.timerHandle=window.setInterval(()=>{o.timeEl.textContent=ne((performance.now()-o.startTime)/1e3)},100)}}updateStep(t,a){const o=this.steps.get(t);o&&(o.status==="pending"?this.startStep(t,a):o.detailEl.textContent=a)}completeStep(t,a){const o=this.steps.get(t);!o||o.status==="done"||(clearInterval(o.timerHandle),o.status="done",o.el.className="pt-step done",a&&(o.detailEl.textContent=a),o.startTime>0&&(o.timeEl.textContent=ne((performance.now()-o.startTime)/1e3)))}handleProgress(t){if(t.stage==="setup"){t.iteration===0?this.startStep("setup","Initializing..."):this.completeStep("setup");return}if(t.stage==="integrals"){t.iteration===0?this.startStep("integrals","Computing ERI..."):this.completeStep("integrals");return}if(t.stage==="scf"){const a=t.delta_e??0;t.iteration===0?(this.steps.get("setup")?.status==="active"&&this.completeStep("setup"),this.steps.get("integrals")?.status==="active"&&this.completeStep("integrals"),this.startStep("scf","iter 0")):this.updateStep("scf",`iter ${t.iteration}  ΔE=${a.toExponential(2)}`)}else if(t.stage==="posthf")t.iteration===0?(this.steps.get("scf")?.status==="active"&&this.completeStep("scf"),this.startStep("posthf","Starting...")):this.completeStep("posthf");else if(t.stage==="ccsd")this.steps.get("scf")?.status==="active"&&this.completeStep("scf"),this.updateStep("posthf",`CCSD iter ${t.iteration}  ΔE=${(t.delta_e??0).toExponential(2)}`);else if(t.stage==="ccsd_lambda")this.updateStep("posthf",`Λ iter ${t.iteration}  ‖Δλ‖=${(t.residual??0).toExponential(2)}`);else if(t.stage==="excited"){this.steps.get("scf")?.status==="active"&&this.completeStep("scf"),this.steps.get("posthf")?.status==="active"&&this.completeStep("posthf");const a={0:"MO transform...",1:"Building operator...",2:"Solving eigenstates..."};this.startStep("excited",a[t.iteration]||"Computing...")}else if(t.stage==="davidson")this.steps.get("posthf")?.status==="active"&&this.completeStep("posthf"),this.updateStep("excited",`Davidson iter ${t.iteration}  max|r|=${(t.max_residual??0).toExponential(2)}`);else if(t.stage==="schur")this.updateStep("excited",t.iteration===0?"Schur diagonalization...":"Schur done");else if(t.stage==="schur_omega"){const a=[t.total_energy,t.delta_e,t.correlation_energy].filter(r=>r!==void 0),o=a[0]!==void 0?Math.floor(a[0]):"?",e=a[1]!==void 0?Number(a[1]).toFixed(6):"",i=a[2]!==void 0?Number(a[2]).toExponential(2):"";this.updateStep("excited",`Root ${o} ω=${e} Δω=${i}`)}}complete(){for(const[,t]of this.steps)(t.status==="active"||t.status==="pending")&&(clearInterval(t.timerHandle),t.status="done",t.el.className="pt-step done",t.startTime>0&&(t.timeEl.textContent=ne((performance.now()-t.startTime)/1e3)));this.titleEl.innerHTML='<span class="pt-title-icon" style="background:var(--color-converged,#10b981)"></span> COMPLETE',this.cancelBtn.textContent="Close",this.cancelBtn.onclick=()=>this.close(),setTimeout(()=>this.close(),1500)}fail(t){for(const[,a]of this.steps)(a.status==="active"||a.status==="pending")&&(clearInterval(a.timerHandle),a.status="error",a.el.className="pt-step error");this.titleEl.innerHTML='<span class="pt-title-icon" style="background:var(--color-error,#e74c3c)"></span> ERROR',this.cancelBtn.textContent="Close",this.cancelBtn.onclick=()=>this.close(),setTimeout(()=>this.close(),3e3)}close(){clearInterval(this.totalTimer);for(const t of this.steps.values())clearInterval(t.timerHandle);this.overlay.classList.remove("visible"),this.overlay.addEventListener("transitionend",()=>this.overlay.remove(),{once:!0}),setTimeout(()=>{this.overlay.parentNode&&this.overlay.remove()},500)}}function tt(s,t="stored",a="energy"){const o=[{id:"setup",label:"Setup"}];t==="ri"||t==="RI"?(o.push({id:"ri-2c",label:"RI: 2-Center Integrals"}),o.push({id:"ri-3c",label:"RI: 3-Center Integrals"}),o.push({id:"ri-b",label:"RI: B Matrix"})):o.push({id:"integrals",label:"Integrals"}),o.push({id:"scf",label:"SCF Loop"});const e={mp2:"MP2 Correlation",mp3:"MP3 Correlation",mp4:"MP4 Correlation",cc2:"CC2 Correlation",ccsd:"CCSD Correlation",ccsd_t:"CCSD(T) Correlation",ccsd_density:"CCSD + Lambda",fci:"Full CI"},i={cis:"CIS Excited States",adc2:"ADC(2) Excited States",adc2x:"ADC(2)-x Excited States",eom_mp2:"EOM-MP2 Excited States",eom_cc2:"EOM-CC2 Excited States",eom_ccsd:"EOM-CCSD Excited States"};return s in e&&o.push({id:"posthf",label:e[s]}),s in i&&(s==="eom_ccsd"&&o.push({id:"posthf",label:"CCSD Ground State"}),o.push({id:"excited",label:i[s]})),a==="gradient"?o.push({id:"gradient",label:"Nuclear Gradient"}):a==="hessian"?o.push({id:"hessian",label:"Hessian / Frequencies"}):a==="optimize"&&o.push({id:"optimize",label:"Geometry Optimization"}),o.push({id:"properties",label:"Properties"}),o}async function st(s){let t=["."],a=[{filename:"H2.xyz",name:"H2"},{filename:"H2O.xyz",name:"H2O"}],o=["sto-3g","3-21g","6-31g","cc-pvdz","cc-pvtz"],e=[];try{const[n,v,b,u]=await Promise.all([De(),Se(),Ie(),Ne()]);n.length>0&&(t=n),v.length>0&&(a=v),b.length>0&&(o=b),u.length>0&&(e=u)}catch(n){console.warn("API not available, using defaults:",n)}s.innerHTML=`
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
  `;const i=s.querySelector("#theme-btn"),r=s.querySelector("#theme-icon");i.addEventListener("click",()=>{const n=Re();r.textContent=n==="dark"?"☀":"☾"});const l=Ae(s.querySelector("#molecule-col"),t,a,()=>{}),h=Be(s.querySelector("#settings-col"),o,e),m=Je(s.querySelector("#results-col")),f=s.querySelector("#run-btn"),x=s.querySelector("#cancel-btn");let d=null;f.addEventListener("click",async()=>{const n=l.getXyz(),v=l.getXyzFile();if(!n&&!v){alert("Please enter a molecule (XYZ text or select a sample).");return}const b=h.getParams(),u={...P,...b,xyz_text:n,xyz_file:v,xyz_dir:l.getXyzDir()};f.disabled=!0,x.classList.remove("hidden"),m.hide();const C=tt(u.post_hf_method,u.eri_method,"energy"),M=new et(C,()=>{d&&(d.abort(),d=null),M.close(),f.disabled=!1,x.classList.add("hidden")});d=new AbortController;try{let c=function(O){const g=((performance.now()-N)/1e3).toFixed(3),y=O.stage,$=O.iteration,_=O.values||[];if(y==="setup")E.push(`[${g}s] ${$===0?"Setup: Initializing...":"Setup: Core Hamiltonian computed"}`);else if(y==="integrals")E.push(`[${g}s] ${$===0?"Integrals: Computing ERIs...":"Integrals: Done"}`);else if(y==="integrals_ri"){const p={0:"2-center ERIs",1:"Cholesky",2:"3-center ERIs",3:"B matrix",4:"RI done"};E.push(`[${g}s] RI: ${p[$]||`step ${$}`}`)}else if(y==="scf"){const p=_[2]!==void 0?Number(_[2]).toFixed(10):"",k=_[1]!==void 0?Number(_[1]).toExponential(2):"";E.push(`[${g}s] SCF iter ${$}  E=${p}  ΔE=${k}`)}else if(y==="posthf")E.push(`[${g}s] ${$===0?"Post-HF: Starting...":"Post-HF: Done"}`);else if(y==="ccsd"){const p=_[1]!==void 0?Number(_[1]).toExponential(2):"";E.push(`[${g}s] CCSD iter ${$}  ΔE=${p}`)}else if(y==="ccsd_lambda")E.push(`[${g}s] Lambda iter ${$}  residual=${_[0]!==void 0?Number(_[0]).toExponential(2):""}`);else if(y==="excited"){const p={0:"MO transform",1:"Building operator",2:"Solving eigenstates"};E.push(`[${g}s] Excited: ${p[$]||`step ${$}`}`)}else if(y==="schur")E.push(`[${g}s] Schur ${$===0?"diagonalization...":"done"}`);else if(y==="schur_omega"){const p=_[0]!==void 0?Math.floor(Number(_[0])):"?",k=_[1]!==void 0?Number(_[1]).toFixed(8):"",q=_[2]!==void 0?Number(_[2]).toExponential(2):"";E.push(`[${g}s] Schur-omega Root ${p}  omega=${k}  d_omega=${q}`)}else if(y==="davidson"){const p=_.length,k=p>0?Number(_[p-1]).toExponential(2):"",q=_.slice(0,p-1).map(F=>Number(F).toFixed(6));E.push(`[${g}s] Davidson iter ${$}  max|r|=${k}  eigs=[${q.join(", ")}]`)}else E.push(`[${g}s] ${y} iter ${$}`)};const T=(await fetch("/api/run/inprocess/stream",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({xyz_text:u.xyz_text,xyz_file:u.xyz_file,xyz_dir:u.xyz_dir,basis:u.basis,method:u.method,charge:u.charge,beta_to_alpha:u.beta_to_alpha,convergence_method:u.convergence_method,diis_size:u.diis_size,damping_factor:u.damping_factor,maxiter:u.maxiter,convergence_energy_threshold:u.convergence_energy_threshold,schwarz_screening_threshold:u.schwarz_screening_threshold,initial_guess:u.initial_guess,post_hf_method:u.post_hf_method,n_excited_states:u.n_excited_states,spin_type:u.spin_type,eri_method:u.eri_method,auxiliary_basis:u.auxiliary_basis,auxiliary_basis_dir:u.auxiliary_basis_dir,excited_solver:u.excited_solver,mulliken:u.mulliken,mayer:u.mayer,wiberg:u.wiberg}),signal:d.signal})).body?.getReader();if(!T)return;const z=new TextDecoder;let L="";const H=()=>new Promise(O=>requestAnimationFrame(()=>O()));let w="";const E=[],N=performance.now();for(;;){const{done:O,value:g}=await T.read();if(O)break;L+=z.decode(g,{stream:!0});const y=L.split(`

`);L=y.pop()||"";for(const $ of y){const _=$.trim();if(_.startsWith("data: "))try{const p=JSON.parse(_.slice(6));if(p.type==="progress")w&&w!==p.stage&&await H(),w=p.stage,c(p),M.handleProgress({type:"progress",stage:p.stage,iteration:p.iteration,total_energy:p.values?.[2],delta_e:p.values?.[1],correlation_energy:p.values?.[0],residual:p.values?.[0],max_residual:p.values?.[0]});else if(p.type==="result"){const k=((performance.now()-N)/1e3).toFixed(3),q=p.data.summary||{};E.push(`[${k}s] Done. Total energy: ${(q.total_energy??0).toFixed(10)} Ha`),p.data.post_hf&&E.push(`  Post-HF (${p.data.post_hf.method}): correction=${p.data.post_hf.correction.toFixed(10)}, total=${p.data.post_hf.total_energy.toFixed(10)}`),M.complete();const F=p.data;m.show({ok:!0,raw_output:E.join(`
`),molecule:F.molecule||{},basis_set:F.basis_set||{},scf_iterations:F.scf_iterations||[],summary:F.summary||{},post_hf:F.post_hf||void 0,orbital_energies:F.orbital_energies||[],orbital_energies_beta:F.orbital_energies_beta||[],mulliken:F.mulliken||[],mayer_bond_order:F.mayer_bond_order||[],wiberg_bond_order:F.wiberg_bond_order||[],timing:F.timing||{},excited_states:F.excited_states,excited_states_method:F.excited_states_method,excited_states_spin:F.excited_states_spin})}else p.type==="error"&&(E.push(`ERROR: ${p.error}`),M.fail(p.error),m.showError(p.error,E.join(`
`)))}catch{}}}}catch(c){c instanceof Error&&c.name!=="AbortError"&&M.fail(String(c))}f.disabled=!1,x.classList.add("hidden"),d=null}),x.addEventListener("click",()=>{d&&(d.abort(),d=null),f.disabled=!1,x.classList.add("hidden")})}Pe();const _e=document.querySelector("#app");_e&&st(_e);
