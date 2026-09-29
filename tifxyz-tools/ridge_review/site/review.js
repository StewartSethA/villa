// Ridge review: blind human labelling of ridge_hit decisions. No server; works from file:// or GitHub Pages.
(function () {
  var S = window.SAMPLES || [];
  var KEY = 'ridge_review.v1';
  var st = {};
  try { st = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { st = {}; }
  st.labels = st.labels || {}; st.i = st.i || 0; st.exported = st.exported || false;
  function save() { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) { /* private window: labels live until export */ } }
  var $ = function (id) { return document.getElementById(id); };
  if (st.who) $('who').value = st.who;
  $('who').oninput = function () { st.who = this.value; save(); };

  var VIEWS = [
    ['ct_u', 'pred_u', 'Section: normal (up) × surface direction u'],
    ['ct_v', 'pred_v', 'Section: normal (up) × surface direction v'],
    ['ct_p-4', 'pred_p-4', 'Plane 4 voxels below the surface'],
    ['ct_p+0', 'pred_p+0', 'Plane at the surface'],
    ['ct_p+4', 'pred_p+4', 'Plane 4 voxels above the surface'],
    ['ct_xy', 'pred_xy', 'Axis-aligned CT slice at the cell\'s z']
  ];
  function cur() { return S[st.i]; }
  function lab() { var s = cur(); return st.labels[s.sample_id] = st.labels[s.sample_id] || {}; }

  function show() {
    if (!S.length) { $('views').textContent = 'No samples (samples.js missing).'; return; }
    st.i = Math.max(0, Math.min(S.length - 1, st.i));
    var s = cur(), L = st.labels[s.sample_id] || {};
    $('pos').textContent = (st.i + 1) + ' / ' + S.length;
    var px = s.geometry ? s.geometry.px_um : s.voxel_um;
    $('meta').textContent = s.scroll + ' · sample ' + s.sample_id + ' · 1 px = 1 voxel = ' + px + ' µm (sections ×6, planes ×4, xy ×3 zoom) · sections span ±' +
      (s.geometry ? s.geometry.normal_offsets_vox[1] : 16) + ' voxels along the normal';
    var html = '';
    VIEWS.forEach(function (v) {
      html += '<figure><div class="stack"><img alt="' + v[2] + '" src="img/' + s.sample_id + '/' + v[0] + '.png">' +
        '<img class="pred" alt="" src="img/' + s.sample_id + '/' + v[1] + '.png"></div><figcaption>' + v[2] + '</figcaption></figure>';
    });
    $('views').innerHTML = html;
    document.body.classList.toggle('showpred', !!L.pred_viewed_now);
    $('pred').checked = !!L.pred_viewed_now;
    Array.prototype.forEach.call(document.querySelectorAll('.lab button'), function (b) { b.classList.toggle('on', L.label === b.dataset.l); });
    $('sev').value = L.severity || '';
    $('note').value = L.note || '';
    var n = Object.keys(st.labels).filter(function (k) { return st.labels[k].label; }).length;
    $('count').textContent = n + ' labelled';
    $('prog').style.width = (100 * n / S.length) + '%';
    save();
  }
  function setLabel(l) {
    var L = lab(); L.label = l; L.t = new Date().toISOString(); L.pred_viewed = !!L.pred_viewed;
    show(); setTimeout(function () { if (st.i < S.length - 1) { st.i++; show(); } }, 150);
  }
  function togglePred(on) {
    var L = lab(); L.pred_viewed_now = on; if (on) L.pred_viewed = true;
    document.body.classList.toggle('showpred', on); save();
  }
  Array.prototype.forEach.call(document.querySelectorAll('.lab button'), function (b) { b.onclick = function () { setLabel(b.dataset.l); }; });
  $('pred').onchange = function () { togglePred(this.checked); };
  $('sev').onchange = function () { lab().severity = this.value; save(); };
  $('note').oninput = function () { lab().note = this.value; save(); };
  $('prev').onclick = function () { st.i--; show(); };
  $('next').onclick = function () { st.i++; show(); };
  document.addEventListener('keydown', function (e) {
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT' && e.target.type === 'text') return;
    if (e.key === '1') setLabel('sheet'); else if (e.key === '2') setLabel('not_sheet'); else if (e.key === '3') setLabel('unsure');
    else if (e.key === 'p') { $('pred').checked = !$('pred').checked; togglePred($('pred').checked); }
    else if (e.key === 'ArrowLeft') { st.i--; show(); } else if (e.key === 'ArrowRight') { st.i++; show(); }
  });
  $('export').onclick = function () {
    var out = { tool: 'ridge_review', version: 1, reviewer: st.who || '', exported_utc: new Date().toISOString(),
      n_samples: S.length, labels: {} };
    Object.keys(st.labels).forEach(function (k) {
      var L = st.labels[k]; if (!L.label) return;
      out.labels[k] = { label: L.label, severity: L.severity || '', note: L.note || '', pred_viewed: !!L.pred_viewed, t: L.t };
    });
    var b = new Blob([JSON.stringify(out, null, 1)], { type: 'application/json' });
    var a = document.createElement('a'); a.href = URL.createObjectURL(b);
    a.download = 'ridge_review_labels_' + (st.who || 'anon') + '_' + out.exported_utc.slice(0, 10) + '.json'; a.click();
    st.exported = true; save();
  };
  $('reveal').onclick = function () {
    if (!st.exported) { alert('Export your labels first; the answers are shown only afterwards.'); return; }
    var sc = document.createElement('script'); sc.src = 'key.js';
    sc.onload = function () {
      var K = {}; (window.KEY || []).forEach(function (k) { K[k.sample_id] = k; });
      var t = {}, rows = '';
      Object.keys(st.labels).forEach(function (id) {
        var L = st.labels[id], k = K[id]; if (!L.label || !k) return;
        var key = k.class + ' · ' + k.stratum; t[key] = t[key] || { sheet: 0, not_sheet: 0, unsure: 0 }; t[key][L.label]++;
      });
      Object.keys(t).sort().forEach(function (key) { var r = t[key]; rows += '<tr><td>' + key + '</td><td>' + r.sheet + '</td><td>' + r.not_sheet + '</td><td>' + r.unsure + '</td></tr>'; });
      $('answers').innerHTML = '<h3>Your labels against what the guard did</h3><table><tr><th>guard decision · severity stratum</th><th>sheet</th><th>not sheet</th><th>unsure</th></tr>' + rows +
        '</table><p class="muted">A "removed" cell you called sheet is a ridge_hit false removal. A "kept" cell you called not sheet is a miss. Run score.py on the exported JSON for rates with CIs, reweighted to production.</p>';
    };
    document.body.appendChild(sc);
  };
  $('theme').onclick = function () {
    var r = document.documentElement, d = r.getAttribute('data-theme');
    r.setAttribute('data-theme', d === 'dark' ? 'light' : 'dark');
  };
  show();
})();
