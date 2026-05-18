/**
 * techtree.js — auto-layout and rendering for tech tree dependency graphs.
 *
 * Reads JSON data from <script type="application/json" id="techtree-data-*">
 * elements, computes a hierarchical layout (no coordinates in source TOML),
 * and renders an SVG into the .techtree-canvas placeholder.
 *
 * Layout algorithm:
 *   1. Assign layers: layer[n] = longest path from sources (nodes with no
 *      stated prerequisites). Sources start at 0; boss nodes get the highest
 *      layer numbers and appear at the top of the SVG.
 *   2. Compact source nodes: pull each source up to (min_child_layer - 1) so
 *      that primers sit adjacent to the first chapter they enable, not
 *      isolated at the very bottom.
 *   3. Order nodes within each layer using the barycenter heuristic to
 *      minimise edge crossings.
 *   4. Assign x/y coordinates and render SVG with cubic-bezier edges.
 */
(function () {
  'use strict';

  const NODE_W  = 178;
  const NODE_H  = 58;
  const H_GAP   = 24;   // horizontal gap between sibling nodes
  const V_GAP   = 78;   // vertical gap between layers
  const PAD_X   = 36;   // left/right canvas padding
  const PAD_Y   = 44;   // top/bottom canvas padding

  /* ------------------------------------------------------------------ */
  /* Layout                                                               */
  /* ------------------------------------------------------------------ */

  function computeLayout(nodes, edges) {
    if (!nodes || nodes.length === 0) return { pos: {}, svgW: 400, svgH: 200 };

    const ids      = nodes.map(n => n.id);
    const parents  = {};  // id → [prerequisite ids]
    const children = {};  // id → [dependent ids]
    ids.forEach(id => { parents[id] = []; children[id] = []; });
    (edges || []).forEach(e => {
      if (parents[e.to] && children[e.from]) {
        parents[e.to].push(e.from);
        children[e.from].push(e.to);
      }
    });

    // Phase 1: longest-path layer assignment (sources = layer 0).
    const layer = {};
    ids.forEach(id => { layer[id] = 0; });
    let changed = true;
    for (let pass = 0; pass < ids.length && changed; pass++) {
      changed = false;
      (edges || []).forEach(e => {
        const next = layer[e.from] + 1;
        if (next > layer[e.to]) { layer[e.to] = next; changed = true; }
      });
    }

    // Phase 2: compact source nodes up to just below their first child,
    // so primers appear next to the chapter they enable instead of
    // sitting alone at the very bottom.
    ids.forEach(id => {
      if (parents[id].length === 0 && children[id].length > 0) {
        const minChild = Math.min(...children[id].map(c => layer[c]));
        layer[id] = Math.max(0, minChild - 1);
      }
    });

    // Group nodes by layer.
    const maxLayer = Math.max(...ids.map(id => layer[id]));
    const byLayer  = {};
    for (let l = 0; l <= maxLayer; l++) byLayer[l] = [];
    ids.forEach(id => byLayer[layer[id]].push(id));

    // Phase 3: barycenter ordering — minimise edge crossings.
    // Process layers bottom-up (layer 0 to maxLayer).
    for (let l = 1; l <= maxLayer; l++) {
      const orderBelow = {};
      (byLayer[l - 1] || []).forEach((id, i) => { orderBelow[id] = i; });
      byLayer[l].sort((a, b) => {
        const avg = (id) => {
          const ps = parents[id].filter(p => layer[p] === l - 1);
          if (ps.length === 0) return Infinity;
          return ps.reduce((s, p) => s + (orderBelow[p] ?? 0), 0) / ps.length;
        };
        return avg(a) - avg(b);
      });
    }

    // Compute SVG dimensions.
    const maxInLayer = Math.max(...Object.values(byLayer).map(g => g.length));
    const svgW = maxInLayer * (NODE_W + H_GAP) - H_GAP + 2 * PAD_X;
    const svgH = (maxLayer + 1) * (NODE_H + V_GAP) - V_GAP + 2 * PAD_Y;

    // Assign (x, y) — layer 0 at bottom, maxLayer at top.
    const pos = {};
    Object.entries(byLayer).forEach(([lStr, gIds]) => {
      const l       = parseInt(lStr, 10);
      const y       = PAD_Y + (maxLayer - l) * (NODE_H + V_GAP) + NODE_H / 2;
      const totalW  = gIds.length * NODE_W + (gIds.length - 1) * H_GAP;
      const startX  = (svgW - totalW) / 2 + NODE_W / 2;
      gIds.forEach((id, i) => {
        pos[id] = { x: startX + i * (NODE_W + H_GAP), y };
      });
    });

    return { pos, svgW, svgH };
  }

  /* ------------------------------------------------------------------ */
  /* SVG helpers                                                          */
  /* ------------------------------------------------------------------ */

  const NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v));
    if (parent) parent.appendChild(e);
    return e;
  }

  /* ------------------------------------------------------------------ */
  /* Render one tree into its canvas element                              */
  /* ------------------------------------------------------------------ */

  function renderTree(canvas, data, pageBase, uid) {
    const nodes   = data.nodes || [];
    const edges   = data.edges || [];
    const halfW   = NODE_W / 2;
    const halfH   = NODE_H / 2;

    const { pos, svgW, svgH } = computeLayout(nodes, edges);

    const svg = el('svg', {
      viewBox:             `0 0 ${svgW} ${svgH}`,
      class:               'techtree-svg',
      xmlns:               NS,
      'aria-labelledby':   `tt-title-${uid}`,
      preserveAspectRatio: 'xMidYMid meet',
    }, canvas);

    /* Defs: arrow + halftone. */
    const defs = el('defs', {}, svg);
    defs.innerHTML =
      `<marker id="arr-${uid}" viewBox="0 0 10 10" refX="9" refY="5"` +
      ` markerWidth="6" markerHeight="6" orient="auto-start-reverse">` +
      `<path d="M 0 0 L 10 5 L 0 10 Z" fill="#1a1a1a"/></marker>` +
      `<pattern id="ht-${uid}" width="5" height="5" patternUnits="userSpaceOnUse">` +
      `<circle cx="1" cy="1" r="0.7" fill="#1a1a1a" opacity="0.18"/></pattern>`;

    el('title', { id: `tt-title-${uid}` }, svg).textContent =
      data.title || 'Tech tree';

    /* Edges (drawn first so they appear behind nodes). */
    const edgeG = el('g', { class: 'techtree-edges' }, svg);
    edges.forEach(e => {
      const a = pos[e.from];
      const b = pos[e.to];
      if (!a || !b) return;
      /* Edge exits from TOP of prerequisite (lower on screen, higher y) and
         enters the BOTTOM of the dependent (higher on screen, lower y). */
      const x1 = a.x, y1 = a.y - halfH;  // top of "from" node
      const x2 = b.x, y2 = b.y + halfH;  // bottom of "to" node
      const cy1 = y1 - 50, cy2 = y2 + 50;
      el('path', {
        d:             `M ${x1} ${y1} C ${x1} ${cy1}, ${x2} ${cy2}, ${x2} ${y2}`,
        fill:          'none',
        stroke:        '#1a1a1a',
        'stroke-width': '2.5',
        'marker-end':  `url(#arr-${uid})`,
      }, edgeG);
    });

    /* Nodes. */
    const nodeG = el('g', { class: 'techtree-nodes' }, svg);
    nodes.forEach(node => {
      const p = pos[node.id];
      if (!p) return;
      const kind    = node.kind || 'primer';
      const hasLink = node.link && node.link !== '';
      const rx      = p.x - halfW;
      const ry      = p.y - halfH;

      const g = el('g', {
        class:      `techtree-node techtree-node--${kind}`,
        'data-id':  node.id,
      }, nodeG);

      /* Wrapper: <a> for linked nodes, <g> for unlinked. */
      let wrap;
      if (hasLink) {
        let href = node.link;
        if (!href.startsWith('/') && !href.startsWith('http')) {
          href = pageBase + href;
        }
        wrap = el('a', { href, class: 'techtree-node-link' }, g);
      } else {
        wrap = el('g', {}, g);
      }

      /* Shadow rect. */
      el('rect', { x: rx + 4, y: ry + 4, width: NODE_W, height: NODE_H, fill: '#1a1a1a' }, wrap);

      /* Halftone overlay for boss nodes. */
      if (kind === 'boss') {
        el('rect', { x: rx, y: ry, width: NODE_W, height: NODE_H,
          fill: `url(#ht-${uid})` }, wrap);
      }

      /* Main rect. */
      el('rect', {
        class:          'techtree-node-rect',
        x:              rx,  y:           ry,
        width:          NODE_W, height:   NODE_H,
        stroke:         '#1a1a1a', 'stroke-width': '3',
      }, wrap);

      /* Chapter sub-label (above node). */
      if (node.sub) {
        el('text', {
          class:          'techtree-node-sub',
          x:              p.x,  y:             ry - 6,
          'text-anchor':  'middle',
          'font-size':    '11',
          'font-weight':  '700',
        }, wrap).textContent = node.sub;
      }

      /* Main label (possibly two lines). */
      const lines   = (node.label || '').split('\n');
      const fontSize = lines.length > 1 ? '14' : '16';
      const textEl  = el('text', {
        class:                'techtree-node-label',
        x:                    p.x,  y:                   p.y,
        'text-anchor':        'middle',
        'dominant-baseline':  'middle',
        'font-family':        'Bangers, Impact, sans-serif',
        'font-size':          fontSize,
        'letter-spacing':     '0.04em',
      }, wrap);

      lines.forEach((line, i) => {
        const tspan = el('tspan', { x: p.x }, textEl);
        if (lines.length > 1) {
          tspan.setAttribute('dy', i === 0 ? '-0.55em' : '1.1em');
        }
        tspan.textContent = line;
      });
    });
  }

  /* ------------------------------------------------------------------ */
  /* View toggle (graph ↔ list)                                           */
  /* ------------------------------------------------------------------ */

  function setupToggle(fig) {
    const graphPane = fig.querySelector('[data-techtree-pane="graph"]');
    const listPane  = fig.querySelector('[data-techtree-pane="list"]');
    const buttons   = fig.querySelectorAll('[data-techtree-view]');

    const setView = (view) => {
      buttons.forEach(b => {
        const active = b.dataset.techtreeView === view;
        b.classList.toggle('is-active', active);
        b.setAttribute('aria-selected', active ? 'true' : 'false');
      });
      if (graphPane) graphPane.hidden = (view !== 'graph');
      if (listPane)  listPane.hidden  = (view !== 'list');
      fig.dataset.activeView = view;
    };

    const initial = window.matchMedia('(max-width: 700px)').matches ? 'list' : 'graph';
    setView(initial);
    buttons.forEach(b => b.addEventListener('click', () => setView(b.dataset.techtreeView)));
  }

  /* ------------------------------------------------------------------ */
  /* Bootstrap                                                            */
  /* ------------------------------------------------------------------ */

  function initTechtrees() {
    document.querySelectorAll('[data-techtree]').forEach(fig => {
      const uid    = fig.dataset.techtreeUid || fig.id || 'tt';
      const dataEl = document.getElementById(`techtree-data-${uid}`);
      if (!dataEl) return;

      let data;
      try { data = JSON.parse(dataEl.textContent); }
      catch (err) { console.warn('techtree: bad JSON for', uid, err); return; }

      const canvas   = fig.querySelector('.techtree-canvas');
      const pageBase = fig.dataset.pageBase || '/';
      if (canvas) renderTree(canvas, data, pageBase, uid);

      setupToggle(fig);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTechtrees);
  } else {
    initTechtrees();
  }
})();
