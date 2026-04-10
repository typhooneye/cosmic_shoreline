'use strict';

// ── Constants ─────────────────────────────────────────────────────────────
const ATM_META = {
  CO2:       { label:'CO₂',           color:'#3a6fd8' },
  CH4:       { label:'CH₄',           color:'#d97706' },
  H2O:       { label:'H₂O',           color:'#0891b2' },
  N2O2_N22:  { label:'N₂/O₂ (N22)',   color:'#7c3aed' },
  N2O2_CP24: { label:'N₂/O₂ (CP24)',  color:'#16a34a' },
};
const ATM_ORDER = ['CO2','CH4','H2O','N2O2_N22','N2O2_CP24'];
const MODEL_COLORS = { Jackson:'#3a6fd8', Selsis:'#d97706', custom:'#dc2626' };
const PLANET_DASHES = [[], [6,3], [2,2], [8,3,2,3], [4,2,4,2,1,2]];

// Paths relative to index.html (web-tool/) → repo root is ../
const REPO = '../';
const DATA_FILES = [
  'y_ages_selsis07.npy','x_starmasses_selsis07.npy','Fx_over_Fbol_selsis07.npy',
  'j12_starmasses.npy','j12_ages.npy','j12_LXUV_over_Lbol.npy',
  'guinan16_Lx_over_Lbol.npy','guinan16_mass_range.npy','guinan16_ages.npy',
  'L_B15.npy','tB15_Gyr.npy','Mstar_B15.npy','Rs_B15.npy',
  'tian2009_log_masses.npy','tian2009_log_fluxes.npy','tian2009_log_escape.npy',
  'tian2009_masses.npy','tian2009_fluxes.npy','tian2009_escape.npy',
  'tian2009_GP_GP.npy','tian2009_GP_fluxes.npy','tian2009_GP_escape.npy',
  'x_data_Earth.npy','y_data_Earth.npy','x_data_iron.npy','y_data_iron.npy',
  'x_data_rock.npy','y_data_rock.npy',
  'total_melt_mass_shallower_than_2900km.npy',
];

// ── Pyodide state ─────────────────────────────────────────────────────────
let pyodide = null;
let csvReady = false;

function setLoad(msg, pct) {
  document.getElementById('load-msg').textContent = msg;
  document.getElementById('progress-bar').style.width = pct + '%';
}

async function initApp() {
  try {
    setLoad('Loading Python runtime (first visit ~30 s)…', 5);
    pyodide = await loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/' });

    setLoad('Loading numpy, scipy, pandas…', 25);
    await pyodide.loadPackage(['numpy', 'scipy', 'pandas']);

    setLoad('Fetching data files…', 45);
    pyodide.FS.mkdir('/data');
    await Promise.all(DATA_FILES.map(async f => {
      const resp = await fetch(REPO + 'data-interpolation/' + f);
      if (!resp.ok) throw new Error('Cannot load ' + f + ' — make sure data-interpolation/ is committed to the repo');
      pyodide.FS.writeFile('/data/' + f, new Uint8Array(await resp.arrayBuffer()));
    }));

    setLoad('Initialising CosmicShoreline…', 70);
    const csCode = await fetch(REPO + 'cosmic_shoreline.py').then(r => {
      if (!r.ok) throw new Error('Cannot load cosmic_shoreline.py');
      return r.text();
    });
    pyodide.runPython(csCode);
    pyodide.runPython(`
import numpy as _np, scipy.integrate as _sci_int, json as _json

cs = CosmicShoreline(data_path='/data/')

_ATM = {
  'CO2':       {'MMW': 44},
  'CH4':       {'MMW': 16},
  'H2O':       {'MMW': 18},
  'N2O2_N22':  {'MMW': 28, 'N2O2_model': 'N22'},
  'N2O2_CP24': {'MMW': 28, 'N2O2_model': 'CP24'},
}

def _lxuv_custom(st_mass, t_arr, xp):
    t_Gyr = t_arr / 1e9
    L0 = 10 ** xp['log10_L0']
    Lx = _np.where(t_Gyr < xp['t_sat'],
                   xp['k_sat']   * L0 * t_Gyr ** xp['idx_sat'],
                   xp['k_unsat'] * L0 * t_Gyr ** xp['idx_unsat'])
    Lx_W = Lx / 1e7
    Lbol = 3.846e26 * 10 ** cs.L_bol_interpolator_B15((st_mass, _np.log10(t_arr)))
    Rs   = cs.R_star_interpolator_B15((st_mass, _np.log10(t_arr)))
    As   = 4 * _np.pi * (Rs * 6.96e8) ** 2 * 1e4
    r    = Lx_W / Lbol
    b1, b2, g1, g2 = 116, 3040, -0.35, -0.76
    Leuv1 = b1 * (Lbol*1e7/As)**g1 * r**(g1+1) * Lbol
    Leuv2 = b2 * (Lbol*1e7/As)**g2 * r**(g2+1) * Lbol
    return Lx_W + Leuv1 + Leuv2

def _compute_one(cfg, pl_orbsmax, pl_masse, st_mass, t1, dt, pl_radiuse,
                 xray_model, xray_params, num_steps, want_series):
    kw = {}
    if pl_radiuse not in ('', None): kw['pl_radiuse'] = float(pl_radiuse)
    if 'N2O2_model' in cfg: kw['N2O2_model'] = cfg['N2O2_model']

    if xray_model == 'custom' and xray_params:
        t_arr  = _np.linspace(t1, t1+dt, num_steps)
        L_XUV  = _lxuv_custom(st_mass, t_arr, xray_params)
        rates  = cs.M_C_dot_loss(cfg['MMW'], pl_orbsmax, pl_masse, L_XUV, **kw)
        total  = _sci_int.trapezoid(rates, t_arr) * 365.25 * 24 * 3600
    else:
        method = 'Selsis' if xray_model == 'Selsis' else 'Jackson'
        t_arr, rates, total = cs.integrate_carbon_loss(
            MMW=cfg['MMW'], pl_orbsmax=pl_orbsmax, pl_masse=pl_masse,
            st_mass=st_mass, t1=t1, dt=dt, num_steps=num_steps,
            output='multi', method=method, **kw)

    res = {'total_loss_kg': float(total), 'total_loss_earth_atm': float(total/5.1e18)}
    if want_series:
        step = max(1, len(t_arr)//200)
        res['time_array'] = t_arr[::step].tolist()
        res['loss_rates'] = rates[::step].tolist()
    return res

def run_planets(planets_json, atmospheres_json, t1, dt, want_series):
    planets    = _json.loads(planets_json)
    atmospheres = _json.loads(atmospheres_json)
    num_steps  = 100 if not want_series else 200
    out = []
    for p in planets:
        atm_res = {}; atm_err = {}
        xm  = p.get('xray_model', 'Jackson')
        xp  = p.get('xray_params')
        pl_r = p.get('pl_radiuse') or ''
        for atm in atmospheres:
            if atm not in _ATM:
                atm_err[atm] = 'Unknown atmosphere'; continue
            try:
                atm_res[atm] = _compute_one(
                    _ATM[atm], float(p['pl_orbsmax']), float(p['pl_masse']),
                    float(p['st_mass']), float(t1), float(dt), pl_r,
                    xm, xp, num_steps, bool(want_series))
            except Exception as e:
                atm_err[atm] = str(e)
        out.append({'label': p.get('label','Planet'),
                    'pl_vesc': p.get('pl_vesc'),
                    'atmospheres': atm_res, 'errors': atm_err})
    return _json.dumps(out)
`);

    setLoad('Loading planet catalogue…', 88);
    try {
      const resp = await fetch(REPO + 'exoplanets_data/TSM/NASAExoArchive_2025-02-28_aggregate.csv');
      if (resp.ok) {
        pyodide.FS.writeFile('/data/archive.csv', new Uint8Array(await resp.arrayBuffer()));
        await pyodide.runPythonAsync(`
import pandas as _pd

_df = _pd.read_csv('/data/archive.csv', keep_default_na=True, low_memory=False)
_df['pl_masse']     = _np.where(_df['pl_bmasselim']==1, _np.nan, _df['pl_bmasse'])
_df['pl_masseerr1'] = _df['pl_bmasseerr1']; _df['pl_masseerr2'] = _df['pl_bmasseerr2']
_df = _df[(_df['pl_masse']<10)|(_df['pl_masse'].isna())]
_mc = cs.M_R_fit(_df['pl_rade'].to_numpy(), x_M_or_R='R')
_df['pl_masse_calc'] = _np.where(_df['pl_masse'].isna(), _mc, _df['pl_masse'])
_df['pl_rho_ratio']  = cs.M_R_fit(_df['pl_masse'].to_numpy(), x_M_or_R='M')**3 / _df['pl_rade']**3
_ml = ((_df['st_mass']<0.6)&(((_df['pl_rade']<1.6)&_df['pl_masse'].isna())|((_df['pl_rho_ratio']>0.6)&(_df['pl_masse']<10))))
_mm = ((_df['st_mass']>=0.6)&(_df['st_mass']<1.4)&(_np.log10(_df['pl_rade'])<(-0.11*_np.log10(_df['pl_orbper'])+0.37))&(((_df['pl_rade']<1.6)&_df['pl_masse'].isna())|~_df['pl_masse'].isna()))
_rocks = _df[_ml|_mm].copy()
_G=6.67430e-11; _em=5.972e24; _er=6.371e6
_rocks['pl_vesc'] = _np.sqrt(2*_G*_em*_rocks['pl_masse_calc']/(_rocks['pl_rade']*_er))/1e3

def lookup_planets(names_json):
    names = _json.loads(names_json)
    out = []
    for nm in names:
        row = _df[_df['pl_name']==nm]
        if row.empty: out.append({'pl_name':nm,'error':'Not found'}); continue
        r = row.iloc[0]
        pm = r['pl_masse_calc'] if 'pl_masse_calc' in r.index and _pd.notna(r.get('pl_masse_calc')) \
             else (float(r['pl_bmasse']) if _pd.notna(r.get('pl_bmasse'))
                   else (cs.M_R_fit(float(r['pl_rade']),x_M_or_R='R') if _pd.notna(r.get('pl_rade')) else None))
        if pm is None or (_pd.isna(pm) if not isinstance(pm,float) else False):
            out.append({'pl_name':nm,'error':'Cannot determine mass'}); continue
        if not _pd.notna(r.get('pl_orbsmax')):
            out.append({'pl_name':nm,'error':'Missing pl_orbsmax'}); continue
        out.append({'pl_name':nm,'pl_orbsmax':float(r['pl_orbsmax']),'pl_masse':float(pm),
                    'pl_rade':(float(r['pl_rade']) if _pd.notna(r.get('pl_rade')) else None),
                    'st_mass':(float(r['st_mass']) if _pd.notna(r.get('st_mass')) else None)})
    return _json.dumps(out)

def get_rocks_meta():
    v = _rocks[_rocks['pl_orbsmax'].notna()&_rocks['pl_masse_calc'].notna()&_rocks['st_mass'].notna()&_rocks['pl_vesc'].notna()].copy()
    return _json.dumps(v[['pl_name','pl_orbsmax','pl_masse_calc','pl_rade','st_mass','pl_vesc']].rename(columns={'pl_masse_calc':'pl_masse'}).to_dict(orient='records'))
`);
        csvReady = true;
      }
    } catch (_) {}

    setLoad('Ready', 100);
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('run-btn').disabled = false;
    refreshXrayPanel();
  } catch (err) {
    document.getElementById('load-msg').textContent = 'Failed to load.';
    document.getElementById('load-error').textContent = err.message;
    document.getElementById('load-error').style.display = '';
    console.error(err);
  }
}

// ── Mode tabs ─────────────────────────────────────────────────────────────
let currentMode = 'manual';
document.querySelectorAll('.mode-tab:not([data-xray])').forEach(btn => {
  btn.addEventListener('click', () => {
    currentMode = btn.dataset.mode;
    document.querySelectorAll('.mode-tab:not([data-xray])').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('panel-manual').style.display = currentMode==='manual' ? '' : 'none';
    document.getElementById('panel-names').style.display  = currentMode==='names'  ? '' : 'none';
    document.getElementById('panel-rocks').style.display  = currentMode==='rocks'  ? '' : 'none';
  });
});

// ── X-ray tabs ────────────────────────────────────────────────────────────
let xrayModel = 'Jackson';
document.querySelectorAll('#xray-tabs .mode-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    xrayModel = btn.dataset.xray;
    document.querySelectorAll('#xray-tabs .mode-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('xray-custom-panel').style.display = xrayModel==='custom' ? '' : 'none';
    scheduleXrayRefresh(0);
  });
});

// ── Atmosphere multi-select ───────────────────────────────────────────────
const selected = new Set(['CO2']);
document.querySelectorAll('.atm-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const k = btn.dataset.atm;
    if (selected.has(k)) {
      if (selected.size===1) return;
      selected.delete(k);
      btn.classList.remove('active');
    } else {
      selected.add(k);
      btn.classList.add('active');
    }
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────
function fmtSci(x) {
  if (x==null||!isFinite(x)) return '—';
  const e=Math.floor(Math.log10(Math.abs(x)));
  const c=(x/Math.pow(10,e)).toFixed(2);
  const s={'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻'};
  return c+' × 10'+String(e).split('').map(ch=>s[ch]??ch).join('');
}

function parseArr(v) {
  if (!v||String(v).trim()==='') return [null];
  return String(v).replace(/;/g,',').split(',').map(s=>s.trim()).filter(Boolean).map(Number);
}

function buildGlobalXrayParams() {
  if (xrayModel!=='custom') return {};
  return { xray_params: {
    k_sat:    parseArr(document.getElementById('x-k_sat').value)[0],
    k_unsat:  parseArr(document.getElementById('x-k_unsat').value)[0],
    log10_L0: parseArr(document.getElementById('x-log10_L0').value)[0],
    t_sat:    parseArr(document.getElementById('x-t_sat').value)[0],
    idx_sat:  parseArr(document.getElementById('x-idx_sat').value)[0],
    idx_unsat:parseArr(document.getElementById('x-idx_unsat').value)[0],
  }};
}

function collectManualPlanets() {
  const stm = parseArr(document.getElementById('m-st_mass').value);
  const plm = parseArr(document.getElementById('m-pl_masse').value);
  const plr = parseArr(document.getElementById('m-pl_radiuse').value);
  const plo = parseArr(document.getElementById('m-pl_orbsmax').value);
  let xa = {};
  if (xrayModel==='custom') xa={
    k_sat:    parseArr(document.getElementById('x-k_sat').value),
    k_unsat:  parseArr(document.getElementById('x-k_unsat').value),
    log10_L0: parseArr(document.getElementById('x-log10_L0').value),
    t_sat:    parseArr(document.getElementById('x-t_sat').value),
    idx_sat:  parseArr(document.getElementById('x-idx_sat').value),
    idx_unsat:parseArr(document.getElementById('x-idx_unsat').value),
  };
  const n = Math.max(stm.length,plm.length,plo.length,...Object.values(xa).map(a=>a.length));
  return Array.from({length:n},(_,i)=>{
    const r=plr[i%plr.length];
    const p={
      label: n===1?'Planet':`Planet ${i+1}`,
      st_mass:stm[i%stm.length], pl_masse:plm[i%plm.length],
      pl_orbsmax:plo[i%plo.length], pl_radiuse:(r==null||isNaN(r))?'':r,
      xray_model:xrayModel,
    };
    if (xrayModel==='custom') p.xray_params={
      k_sat:xa.k_sat[i%xa.k_sat.length], k_unsat:xa.k_unsat[i%xa.k_unsat.length],
      log10_L0:xa.log10_L0[i%xa.log10_L0.length], t_sat:xa.t_sat[i%xa.t_sat.length],
      idx_sat:xa.idx_sat[i%xa.idx_sat.length], idx_unsat:xa.idx_unsat[i%xa.idx_unsat.length],
    };
    return p;
  });
}

// ── Main chart ────────────────────────────────────────────────────────────
const ctx = document.getElementById('chart').getContext('2d');
let chart = buildTimeChart();

function buildTimeChart() {
  if (chart) chart.destroy();
  return new Chart(ctx, { type:'line', data:{labels:[],datasets:[]}, options:{
    responsive:true, maintainAspectRatio:false, animation:{duration:350},
    scales:{
      x:{type:'linear',title:{display:true,text:'Time (yr)',color:'#6b7280'},
         ticks:{color:'#6b7280',maxTicksLimit:6,callback:v=>v>=1e9?(v/1e9).toFixed(1)+'G':v>=1e6?(v/1e6).toFixed(0)+'M':v},grid:{color:'#e0e2ea'}},
      y:{type:'logarithmic',title:{display:true,text:'Loss rate (kg/s)',color:'#6b7280'},
         ticks:{color:'#6b7280',maxTicksLimit:6,callback:v=>v.toExponential(0)},grid:{color:'#e0e2ea'}},
    },
    plugins:{
      legend:{display:true,position:'top',labels:{color:'#1a1d2e',boxWidth:12,padding:14}},
      tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${c.parsed.y.toExponential(2)} kg/s`,title:c=>`t = ${Number(c[0].label).toExponential(2)} yr`}},
    },
  }});
}

function buildScatterChart() {
  if (chart) chart.destroy();
  return new Chart(ctx, { type:'scatter', data:{datasets:[]}, options:{
    responsive:true, maintainAspectRatio:false, animation:{duration:350},
    scales:{
      x:{type:'logarithmic',title:{display:true,text:'Escape velocity (km/s)',color:'#6b7280'},ticks:{color:'#6b7280',maxTicksLimit:6},grid:{color:'#e0e2ea'}},
      y:{type:'logarithmic',title:{display:true,text:'Cumulative loss (kg)',color:'#6b7280'},ticks:{color:'#6b7280',maxTicksLimit:6,callback:v=>v.toExponential(0)},grid:{color:'#e0e2ea'}},
    },
    plugins:{
      legend:{display:true,position:'top',labels:{color:'#1a1d2e',boxWidth:10,padding:14}},
      tooltip:{callbacks:{label:c=>`${c.dataset.label} | ${c.raw.name}: vesc=${c.raw.x.toFixed(1)} km/s, loss=${c.raw.y.toExponential(2)} kg`}},
    },
  }});
}

// ── X-ray panel chart ─────────────────────────────────────────────────────
const xrayCtx = document.getElementById('xray-chart').getContext('2d');
const xrayChart = new Chart(xrayCtx, {
  type:'line',
  data:{labels:[],datasets:[{data:[],borderColor:MODEL_COLORS.Jackson,backgroundColor:MODEL_COLORS.Jackson+'12',borderWidth:1.5,pointRadius:0,fill:true,tension:0.3}]},
  options:{
    responsive:true, maintainAspectRatio:false, animation:{duration:250},
    scales:{
      x:{type:'linear',min:1e6,max:5e9,title:{display:true,text:'Time (yr)',color:'#6b7280',font:{size:10}},
         ticks:{color:'#6b7280',maxTicksLimit:5,font:{size:10},callback:v=>v>=1e9?(v/1e9).toFixed(1)+'G':v>=1e6?(v/1e6).toFixed(0)+'M':v},grid:{color:'#e0e2ea'}},
      y:{type:'logarithmic',title:{display:true,text:'X-ray luminosity (erg/s)',color:'#6b7280',font:{size:10}},
         ticks:{color:'#6b7280',maxTicksLimit:4,font:{size:10},callback:v=>v.toExponential(0)},grid:{color:'#e0e2ea'}},
    },
    plugins:{legend:{display:false},tooltip:{callbacks:{
      label:c=>`${c.parsed.y.toExponential(2)} erg/s`,
      title:c=>`t = ${Number(c[0].label).toExponential(2)} yr`,
    }}},
  },
});

let xrayTimer = null;
function scheduleXrayRefresh(delay=400){ clearTimeout(xrayTimer); xrayTimer=setTimeout(refreshXrayPanel,delay); }

async function refreshXrayPanel() {
  if (!pyodide) return;
  const st_mass = parseArr(document.getElementById('m-st_mass').value)[0] || 1.0;
  const xm = xrayModel;
  const xp = (xm==='custom') ? buildGlobalXrayParams().xray_params : null;

  const pyCode = `
import numpy as _np2
_t2 = _np2.linspace(1e6, 5e9, 300)
_t2_Gyr = _t2 / 1e9
_sm2 = ${st_mass}
_xm2 = "${xm}"
${xp ? `_xp2 = ${JSON.stringify(xp)}` : ''}

if _xm2 == 'custom':
    _L02 = 10 ** _xp2['log10_L0']
    _Lx2 = _np2.where(_t2_Gyr < _xp2['t_sat'],
                      _xp2['k_sat']   * _L02 * _t2_Gyr ** _xp2['idx_sat'],
                      _xp2['k_unsat'] * _L02 * _t2_Gyr ** _xp2['idx_unsat'])
else:
    _mth2 = 'Selsis' if _xm2 == 'Selsis' else 'Jackson'
    _r2   = cs.calculate_L_X_to_bol(_sm2, _t2, method=_mth2)
    _Lb2  = 3.846e26 * 10 ** cs.L_bol_interpolator_B15((_sm2, _np2.log10(_t2)))
    _Lx2  = _r2 * _Lb2 * 1e7

[_t2.tolist(), _Lx2.tolist()]
`;
  try {
    const res = await pyodide.runPythonAsync(pyCode);
    const [tarr, lxarr] = res.toJs();
    xrayChart.data.labels           = tarr;
    xrayChart.data.datasets[0].data = lxarr;
    const c = MODEL_COLORS[xm] || '#888';
    xrayChart.data.datasets[0].borderColor     = c;
    xrayChart.data.datasets[0].backgroundColor = c+'12';
    xrayChart.update();
  } catch (err) {
    console.error('xray panel:', err);
  }
}

document.getElementById('m-st_mass').addEventListener('input', ()=>scheduleXrayRefresh());
['x-k_sat','x-k_unsat','x-log10_L0','x-t_sat','x-idx_sat','x-idx_unsat'].forEach(id=>{
  document.getElementById(id).addEventListener('input', ()=>scheduleXrayRefresh());
});

// ── Results helpers ───────────────────────────────────────────────────────
function updateResultsStrip(planets_data, atm_keys) {
  const strip = document.getElementById('results-strip');
  document.getElementById('results-placeholder')?.remove();
  strip.querySelectorAll('.result-card').forEach(c=>c.remove());
  for (const atm of ATM_ORDER) {
    if (!atm_keys.includes(atm)) continue;
    const meta=ATM_META[atm];
    let totalKg=0,ok=0;
    for (const p of planets_data) {
      const r=p.atmospheres[atm];
      if(r){totalKg+=r.total_loss_kg;ok++;}
    }
    const card=document.createElement('div');
    card.className='result-card';
    if (ok>0) {
      const lbl=planets_data.length>1?`${meta.label} (sum, ${ok}p)`:meta.label;
      card.innerHTML=`<div class="rc-header"><span class="rc-dot" style="background:${meta.color}"></span>${lbl}</div><div class="rc-val" style="color:${meta.color}">${fmtSci(totalKg)} kg</div><div class="rc-sub">${fmtSci(totalKg/5.1e18)} Earth atm</div>`;
    } else {
      const err=planets_data[0]?.errors?.[atm]||'error';
      card.innerHTML=`<div class="rc-header"><span class="rc-dot" style="background:#9ca3af"></span>${meta.label}</div><div class="rc-val" style="color:#dc2626;font-size:13px">Error</div><div class="rc-sub" style="color:#dc2626">${err}</div>`;
    }
    strip.appendChild(card);
  }
}

function updateSummaryTable(planets_data, atm_keys) {
  const wrap=document.getElementById('summary-wrap');
  const table=document.getElementById('summary-table');
  if (planets_data.length<=1){wrap.style.display='none';return;}
  const ks=ATM_ORDER.filter(k=>atm_keys.includes(k));
  let h=`<thead><tr><th>Planet</th>${ks.map(k=>`<th>${ATM_META[k].label} (kg)</th>`).join('')}</tr></thead><tbody>`;
  for (const p of planets_data){
    h+=`<tr><td>${p.label}</td>${ks.map(k=>{const r=p.atmospheres[k];return`<td>${r?fmtSci(r.total_loss_kg):'—'}</td>`;}).join('')}</tr>`;
  }
  table.innerHTML=h+'</tbody>';
  wrap.style.display='';
}

function updateTimeChart(planets_data, atm_keys) {
  const ks=ATM_ORDER.filter(k=>atm_keys.includes(k));
  chart.data.datasets=[];
  let sharedTime=null;
  for (const p of planets_data){
    for(const k of ks){
      if(p.atmospheres[k]?.time_array){sharedTime=p.atmospheres[k].time_array;break;}
    }
    if(sharedTime)break;
  }
  if (sharedTime) chart.data.labels=sharedTime;
  planets_data.forEach((p,pi)=>{
    const dash=PLANET_DASHES[pi%PLANET_DASHES.length];
    for (const k of ks){
      const r=p.atmospheres[k];
      if(!r?.loss_rates) continue;
      chart.data.datasets.push({
        label:planets_data.length>1?`${ATM_META[k].label} — ${p.label}`:ATM_META[k].label,
        data:r.loss_rates, borderColor:ATM_META[k].color, backgroundColor:ATM_META[k].color+'10',
        borderWidth:2, borderDash:dash, pointRadius:0, fill:false, tension:0.3,
      });
    }
  });
  document.getElementById('chart-title').textContent='Mass loss rate over time';
  chart.update();
}

function updateScatterChart(planets_data, atm_keys) {
  const ks=ATM_ORDER.filter(k=>atm_keys.includes(k));
  chart.data.datasets=[];
  for (const k of ks){
    const pts=planets_data.filter(p=>p.atmospheres[k]&&p.pl_vesc).map(p=>({x:p.pl_vesc,y:p.atmospheres[k].total_loss_kg,name:p.label}));
    if(!pts.length) continue;
    chart.data.datasets.push({label:ATM_META[k].label,data:pts,backgroundColor:ATM_META[k].color+'aa',borderColor:ATM_META[k].color,borderWidth:0.5,pointRadius:4});
  }
  document.getElementById('chart-title').textContent='Rocky planet sample — escape velocity vs. cumulative atmospheric loss';
  chart.update();
}

// ── Run button ────────────────────────────────────────────────────────────
const runBtn=document.getElementById('run-btn');
const statusMsg=document.getElementById('status-msg');

runBtn.addEventListener('click', async ()=>{
  if (!pyodide){statusMsg.textContent='⚠ Still loading…';return;}
  runBtn.disabled=true;
  statusMsg.textContent='Computing…';
  statusMsg.className='status-msg';

  const t1=parseFloat(document.getElementById('t1').value)||1e6;
  const dt=parseFloat(document.getElementById('dt').value)||5e9;
  const atm_keys=[...selected];
  const isScatter=currentMode==='rocks';

  try {
    let planets=[];

    if (currentMode==='manual') {
      planets=collectManualPlanets();
    } else if (currentMode==='names') {
      if (!csvReady) throw new Error('Planet catalogue not loaded — CSV may be missing from repo');
      const raw=document.getElementById('n-names').value;
      const names=raw.replace(/\n/g,',').split(',').map(s=>s.trim()).filter(Boolean);
      if (!names.length) throw new Error('Enter at least one planet name');
      statusMsg.textContent='Looking up planet parameters…';
      const lookups=JSON.parse(pyodide.globals.get('lookup_planets')(JSON.stringify(names)));
      const found=[],errs=[];
      lookups.forEach(l=>l.error?errs.push(`${l.pl_name}: ${l.error}`):found.push(l));
      document.getElementById('name-preview').textContent=errs.length?'⚠ '+errs.join(' | '):'';
      if (!found.length) throw new Error('None of the planet names could be resolved');
      const xp=buildGlobalXrayParams();
      planets=found.map(l=>({label:l.pl_name,pl_orbsmax:l.pl_orbsmax,pl_masse:l.pl_masse,
        st_mass:l.st_mass,pl_radiuse:l.pl_rade||'',xray_model:xrayModel,...xp}));
    } else {
      if (!csvReady) throw new Error('Planet catalogue not loaded — CSV may be missing from repo');
      statusMsg.textContent='Loading rocky sample…';
      const meta=JSON.parse(pyodide.globals.get('get_rocks_meta')());
      const xp=buildGlobalXrayParams();
      planets=meta.map(p=>({label:p.pl_name,pl_orbsmax:p.pl_orbsmax,pl_masse:p.pl_masse,
        st_mass:p.st_mass,pl_radiuse:p.pl_rade||'',pl_vesc:p.pl_vesc,xray_model:xrayModel,...xp}));
    }

    statusMsg.textContent=`Computing ${planets.length} planet(s) × ${atm_keys.length} atmosphere(s)…`;
    const fn=pyodide.globals.get('run_planets');
    const raw=fn(JSON.stringify(planets), JSON.stringify(atm_keys), t1, dt, !isScatter);
    const result=JSON.parse(raw);

    if (isScatter) {
      result.forEach((r,i)=>r.pl_vesc=planets[i].pl_vesc);
    }

    if (isScatter) {
      chart=buildScatterChart();
      updateScatterChart(result,atm_keys);
      document.getElementById('summary-wrap').style.display='none';
    } else {
      chart=buildTimeChart();
      updateTimeChart(result,atm_keys);
      updateSummaryTable(result,atm_keys);
    }

    updateResultsStrip(result, atm_keys);
    const ne=result.reduce((a,p)=>a+Object.keys(p.errors).length,0);
    statusMsg.textContent=ne?`⚠ ${ne} atmosphere(s) failed on some planets`:'';
    statusMsg.className=ne?'status-msg error':'status-msg';
    refreshXrayPanel();
  } catch(err) {
    statusMsg.textContent='⚠ '+err.message;
    statusMsg.className='status-msg error';
    console.error(err);
  } finally {
    runBtn.disabled=false;
  }
});

// ── Boot ──────────────────────────────────────────────────────────────────
initApp();
