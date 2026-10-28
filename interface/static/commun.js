/* Utilitaires communs aux deux fenêtres. */
"use strict";

const CLASSES = ["NORMAL", "PATHOLOGICAL"];

function verifierModeleBinaire(modele) {
  if (!modele || !Array.isArray(modele.classes)
      || modele.classes.length !== CLASSES.length
      || !CLASSES.every((classe, index) => modele.classes[index] === classe)) {
    throw new Error(
      "Le serveur utilise une ancienne configuration. Arrêtez-le puis relancez l'application binaire."
    );
  }
}

const CLASSE_INFO = {
  NORMAL: { nom: "ECG normal", court: "Normal" },
  PATHOLOGICAL: { nom: "ECG pathologique", court: "Pathologique" },
};

const STAT_INFO = {
  mean: { nom: "Moyenne", unite: "mV", court: "moy." },
  std: { nom: "Écart-type", unite: "mV", court: "σ" },
  rms: { nom: "Valeur efficace (RMS)", unite: "mV", court: "rms" },
  median: { nom: "Médiane", unite: "mV", court: "méd." },
  iqr: { nom: "Écart interquartile", unite: "mV", court: "IQR" },
  min: { nom: "Minimum", unite: "mV", court: "min" },
  max: { nom: "Maximum", unite: "mV", court: "max" },
  skewness: { nom: "Asymétrie (skewness)", unite: "", court: "asym." },
  kurtosis: { nom: "Aplatissement (kurtosis)", unite: "", court: "kurt." },
  line_length: { nom: "Longueur de ligne", unite: "mV", court: "long." },
};

/* Nombre au format français : virgule décimale, vrai signe moins. */
function fr(x, d = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const s = Number(x).toFixed(d).replace(".", ",");
  return s.startsWith("-") ? "−" + s.slice(1) : s;
}
function pourcent(x, d = 0) { return fr(100 * x, d) + " %"; }
function entier(n) { return Number(n).toLocaleString("fr-FR"); }

function cssVar(nom) {
  return getComputedStyle(document.documentElement).getPropertyValue(nom).trim();
}
function couleurClasse(c) { return cssVar(`--c-${c}`); }

/* ---------- Couleurs ---------- */

function hexVersRvb(hex) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function melange(a, b, t) {
  const x = hexVersRvb(a), y = hexVersRvb(b);
  const r = x.map((v, i) => Math.round(v + (y[i] - v) * t));
  return `rgb(${r[0]},${r[1]},${r[2]})`;
}
function alpha(hex, a) {
  const [r, g, b] = hexVersRvb(hex);
  return `rgba(${r},${g},${b},${a})`;
}
/* Couleur divergente d'une valeur normalisée z (bleu négatif, orange positif). */
function couleurZ(z, borne = 3) {
  const t = Math.max(-1, Math.min(1, z / borne));
  const neutre = cssVar("--neutre-z");
  return t < 0 ? melange(neutre, cssVar("--accent"), -t) : melange(neutre, cssVar("--orange"), t);
}
function texteSurZ(z, borne = 3) { return Math.abs(z) / borne > 0.55 ? "#ffffff" : cssVar("--texte"); }

/* ---------- Mathématiques ---------- */

const sigmoide = (x) => 1 / (1 + Math.exp(-x));
const relu = (x) => Math.max(0, x);
const limite = (x, a, b) => Math.max(a, Math.min(b, x));
const interpole = (a, b, t) => a + (b - a) * t;
const adoucir = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/* Quantile par interpolation linéaire (méthode par défaut de numpy). */
function quantile(trie, q) {
  const pos = (trie.length - 1) * q;
  const i = Math.floor(pos);
  const f = pos - i;
  return i + 1 < trie.length ? trie[i] + f * (trie[i + 1] - trie[i]) : trie[i];
}

/* ---------- Canvas ---------- */

/* Ajuste la résolution du canvas à l'écran ; renvoie le contexte et la taille. */
function preparerCanvas(canvas, hauteur) {
  const ratio = window.devicePixelRatio || 1;
  const largeur = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = hauteur ?? canvas.clientHeight;
  canvas.width = Math.round(largeur * ratio);
  canvas.height = Math.round(h * ratio);
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, l: largeur, h };
}

function texteCanvas(ctx, texte, x, y, { taille = 12, couleur, graisse = 400, aligne = "left", base = "alphabetic", police } = {}) {
  ctx.font = `${graisse} ${taille}px ${police || cssVar("--police")}`;
  ctx.fillStyle = couleur || cssVar("--texte");
  ctx.textAlign = aligne;
  ctx.textBaseline = base;
  ctx.fillText(texte, x, y);
}

function rectArrondi(ctx, x, y, l, h, r) {
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, l, h, r) : ctx.rect(x, y, l, h);
}

/* ---------- Animation ---------- */

/* Interpolation temporelle ; renvoie une fonction d'annulation. */
function animer(duree, maj, fin) {
  const debut = performance.now();
  let id = 0;
  const etape = (t) => {
    const p = Math.min(1, (t - debut) / duree);
    maj(p);
    if (p < 1) id = requestAnimationFrame(etape);
    else if (fin) fin();
  };
  id = requestAnimationFrame(etape);
  return () => cancelAnimationFrame(id);
}

const mouvementReduit = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Appelle fn la première fois que l'élément devient visible. */
function quandVisible(el, fn, seuil = 0.25) {
  const obs = new IntersectionObserver((entrees) => {
    if (entrees.some((e) => e.isIntersecting)) { obs.disconnect(); fn(); }
  }, { threshold: seuil });
  obs.observe(el);
}

/* ---------- Serveur ---------- */

async function api(chemin, options) {
  const reponse = await fetch(chemin, options);
  let donnees = {};
  try { donnees = await reponse.json(); } catch (e) { /* réponse vide */ }
  if (!reponse.ok) throw new Error(donnees.erreur || `Erreur ${reponse.status}`);
  return donnees;
}

/* Dernière analyse, partagée entre les deux fenêtres (navigateur local). */
const CLE_DERNIER = "ecg-mlp:binaire:derniere-analyse";
function memoriserAnalyse(analyse) {
  try { localStorage.setItem(CLE_DERNIER, JSON.stringify(analyse)); } catch (e) { /* stockage indisponible */ }
}
function relireAnalyse() {
  try { return JSON.parse(localStorage.getItem(CLE_DERNIER)); } catch (e) { return null; }
}

/* ---------- DOM ---------- */

function el(balise, attributs = {}, ...enfants) {
  const n = document.createElement(balise);
  for (const [k, v] of Object.entries(attributs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "style" && typeof v === "object") {
      // Les variables CSS (--nom) exigent setProperty.
      for (const [p, val] of Object.entries(v)) p.startsWith("--") ? n.style.setProperty(p, val) : (n.style[p] = val);
    }
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const e of enfants.flat()) if (e !== null && e !== undefined && e !== false) n.append(e);
  return n;
}

const infobulle = (() => {
  let noeud = null;
  return {
    montrer(html, x, y) {
      if (!noeud) { noeud = el("div", { class: "infobulle", role: "tooltip" }); document.body.append(noeud); }
      noeud.innerHTML = html;
      const l = noeud.offsetWidth, h = noeud.offsetHeight;
      const px = x + 14 + l > window.innerWidth - 8 ? x - l - 14 : x + 14;
      const py = y + 14 + h > window.innerHeight - 8 ? y - h - 10 : y + 14;
      noeud.style.left = Math.max(8, px) + "px";
      noeud.style.top = Math.max(8, py) + "px";
      noeud.classList.add("visible");
    },
    cacher() { if (noeud) noeud.classList.remove("visible"); },
  };
})();

function pastilleClasse(c, texte) {
  return el("span", { class: "pastille", style: { "--couleur": couleurClasse(c) } }, texte ?? c);
}

/* Redessine à chaque changement de taille (et de thème). */
function surRedimension(fn) {
  let attente = 0;
  const relancer = () => { cancelAnimationFrame(attente); attente = requestAnimationFrame(fn); };
  window.addEventListener("resize", relancer);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", relancer);
  return relancer;
}

/* Toute erreur d'affichage est signalée dans la page plutôt que passée sous silence. */
window.addEventListener("error", (e) => {
  const bandeau = document.getElementById("erreur-page") || document.body.appendChild(
    el("div", { id: "erreur-page", class: "erreur", role: "alert", style: { position: "fixed", bottom: "12px", left: "12px", right: "12px", zIndex: 60 } }));
  bandeau.textContent = `Erreur d'affichage : ${e.message} (${(e.filename || "").split("/").pop()}:${e.lineno})`;
});
