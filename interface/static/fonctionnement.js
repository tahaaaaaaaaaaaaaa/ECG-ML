/* Fenêtre 2 : fonctionnement du réseau, avec les vrais poids du modèle. */
"use strict";

const $ = (id) => document.getElementById(id);
const STATS_MODELE = ["mean", "std", "rms", "median", "iqr", "min", "max", "skewness", "kurtosis", "line_length"];
const CLE_GRAINE = "ecg-mlp:binaire:graine";

const F = {
  modele: null,
  distributions: null,
  courbes: null,
  roc: null,
  catalogue: null,
  analyse: null,
  graine: 0,
  poids: {},
  passe: null,
  ecartLocal: 0,
};

/* ============================================================
   Outils de dessin
   ============================================================ */

let PAL = null;
function palette() {
  if (!PAL) {
    const v = cssVar;
    PAL = {
      texte: v("--texte"), texte2: v("--texte-2"), texte3: v("--texte-3"),
      bord: v("--bord"), bordFort: v("--bord-fort"),
      surface: v("--surface"), surface2: v("--surface-2"), surface3: v("--surface-3"),
      accent: v("--accent"), orange: v("--orange"), rouge: v("--rouge"), vert: v("--vert"),
      neutre: v("--neutre-z"), eteint: v("--eteint"),
      police: v("--police"), mono: v("--mono"),
      classes: Object.fromEntries(CLASSES.map((c) => [c, v(`--c-${c}`)])),
    };
  }
  return PAL;
}
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { PAL = null; });

function versRvb(couleur) {
  if (couleur.startsWith("#")) return hexVersRvb(couleur);
  return couleur.match(/\d+/g).slice(0, 3).map(Number);
}
function melangeRgb(a, b, t) {
  const x = versRvb(a), y = versRvb(b);
  return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",")})`;
}

function zc(z, borne = 3) {
  const p = palette();
  const t = limite(z / borne, -1, 1);
  return t < 0 ? melange(p.neutre, p.accent, -t) : melange(p.neutre, p.orange, t);
}

function ecrire(ctx, s, x, y, { taille = 12, couleur, graisse = 400, aligne = "left", base = "alphabetic", mono = false } = {}) {
  const p = palette();
  ctx.font = `${graisse} ${taille}px ${mono ? p.mono : p.police}`;
  ctx.fillStyle = couleur || p.texte;
  ctx.textAlign = aligne;
  ctx.textBaseline = base;
  ctx.fillText(s, x, y);
}

/* Canvas redimensionné seulement si nécessaire, puis effacé. */
function contexte(canvas, hauteur) {
  const ratio = window.devicePixelRatio || 1;
  const l = canvas.clientWidth || canvas.parentElement.clientWidth || canvas._l || 0;
  if (l && canvas._l !== l || canvas._h !== hauteur || canvas._r !== ratio) {
    preparerCanvas(canvas, hauteur);
    Object.assign(canvas, { _l: l, _h: hauteur, _r: ratio });
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, l, hauteur);
  return { ctx, l, h: hauteur };
}

function ligne(ctx, x1, y1, x2, y2, couleur, largeur = 1, tirets = null) {
  ctx.beginPath();
  ctx.setLineDash(tirets || []);
  ctx.strokeStyle = couleur;
  ctx.lineWidth = largeur;
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.setLineDash([]);
}

function disque(ctx, x, y, r, remplissage, contour, largeur = 1) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  if (remplissage) { ctx.fillStyle = remplissage; ctx.fill(); }
  if (contour) { ctx.strokeStyle = contour; ctx.lineWidth = largeur; ctx.stroke(); }
}

function etiquette(ctx, s, x, y, couleur, { aligne = "center", taille = 11 } = {}) {
  const p = palette();
  ctx.font = `600 ${taille}px ${p.police}`;
  const l = ctx.measureText(s).width + 10;
  const x0 = aligne === "center" ? x - l / 2 : aligne === "right" ? x - l : x;
  rectArrondi(ctx, x0, y - taille / 2 - 4, l, taille + 8, 6);
  ctx.fillStyle = p.surface;
  ctx.fill();
  ctx.strokeStyle = couleur;
  ctx.lineWidth = 1;
  ctx.stroke();
  ecrire(ctx, s, x0 + l / 2, y + 0.5, { taille, couleur, graisse: 600, aligne: "center", base: "middle" });
}

function graduations(min, max, n = 5) {
  const brut = (max - min) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(brut)));
  const r = brut / mag;
  const pas = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  const decimales = Math.max(0, -Math.floor(Math.log10(pas) + 1e-9));
  const valeurs = [];
  for (let v = Math.ceil(min / pas) * pas; v <= max + pas * 1e-6; v += pas) valeurs.push(+v.toFixed(10));
  return { valeurs, decimales };
}

function frAuto(x) {
  const v = Math.abs(x);
  return fr(x, v >= 100 ? 1 : v >= 10 ? 2 : 3);
}

function frExp(x) {
  if (!x) return "0";
  const [m, e] = x.toExponential(1).split("e");
  const exposant = String(Number(e)).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]);
  return `${m.replace(".", ",")} × 10${exposant}`;
}

/* Boucle d'animation active seulement quand l'élément est visible. */
function boucleVisible(noeud, image) {
  let visible = false, id = 0;
  const tic = (t) => { image(t); if (visible) id = requestAnimationFrame(tic); };
  new IntersectionObserver(([e]) => {
    if (e.isIntersecting && !visible) { visible = true; id = requestAnimationFrame(tic); }
    else if (!e.isIntersecting) { visible = false; cancelAnimationFrame(id); }
  }).observe(noeud);
  image(performance.now());
}

/* Animation ponctuelle : appelle dessiner(t en ms) jusqu'à duree, puis l'état final. */
function lecteur(duree, dessiner) {
  let id = 0, debut = 0, enCours = false;
  const tic = (maintenant) => {
    const t = maintenant - debut;
    if (t >= duree) { enCours = false; dessiner(Infinity); return; }
    dessiner(t);
    id = requestAnimationFrame(tic);
  };
  return {
    jouer() {
      cancelAnimationFrame(id);
      if (mouvementReduit()) { enCours = false; dessiner(Infinity); return; }
      debut = performance.now(); enCours = true; id = requestAnimationFrame(tic);
    },
    arreter() { cancelAnimationFrame(id); enCours = false; },
    get enCours() { return enCours; },
  };
}

const prog = (t, debut, duree) => limite((t - debut) / duree, 0, 1);

/* Relance les animations CSS d'une zone ; sans elles, l'état final reste affiché. */
function rejouerCss(zone) {
  zone.classList.remove("anime");
  void zone.offsetWidth;
  zone.classList.add("anime");
}

/* ============================================================
   Données du modèle
   ============================================================ */

const graineCourante = () => F.analyse.graines[F.graine];
const poidsCourants = () => F.poids[F.graine];
const derivations = () => F.analyse.signal.derivations;

async function chargerPoids(i) {
  if (!F.poids[i]) F.poids[i] = await api(`/api/poids/${i}`);
  return F.poids[i];
}

/* Passage avant refait dans le navigateur avec les poids du modèle. */
function passeAvant(couches, z) {
  const pre = [], act = [z];
  let x = z;
  couches.forEach((c, k) => {
    const s = c.W.map((rang, i) => rang.reduce((acc, w, j) => acc + w * x[j], c.b[i]));
    pre.push(s);
    x = k < couches.length - 1 ? s.map(relu) : s;
    act.push(x);
  });
  return { pre, act };
}

function recalculer() {
  const g = graineCourante();
  F.passe = passeAvant(poidsCourants().couches, F.analyse.entrees.normalisees);
  const ecarts = [
    ...F.passe.act[1].map((v, i) => Math.abs(v - g.couches_cachees[0][i])),
    ...F.passe.act[2].map((v, i) => Math.abs(v - g.couches_cachees[1][i])),
    ...F.passe.act[3].map((v, i) => Math.abs(v - g.logits[i])),
  ];
  F.ecartLocal = Math.max(...ecarts);
}

function nomEntree(k) {
  const d = derivations()[Math.floor(k / 10)];
  return { derivation: d, stat: STATS_MODELE[k % 10] };
}

/* ============================================================
   Fil conducteur : ECG suivi et graine
   ============================================================ */

const Fil = {
  init() {
    $("fil-graine").replaceChildren(...F.modele.graines.map((g, i) =>
      el("button", { type: "button", "aria-pressed": String(i === F.graine), onclick: () => changerGraine(i), "aria-label": `Graine ${g.graine}` }, String(g.graine))));
    $("fil-hasard").addEventListener("click", autreEcg);
  },
  maj() {
    const s = F.analyse.source;
    $("fil-ecg").textContent = s.ecg_id ? `ECG ${s.ecg_id}` : `Fichier ${s.nom}`;
    const badges = [];
    if (s.partition === "validation") badges.push(el("span", { class: "pastille sans-point" }, "validation"));
    else if (s.partition === "train") badges.push(el("span", { class: "pastille attention sans-point" }, "train"));
    if (s.etiquettes) badges.push(...CLASSES.filter((c) => s.etiquettes[c]).map((c) => pastilleClasse(c)));
    $("fil-badges").replaceChildren(...badges);
    $("fil-graine").querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", String(i === F.graine)));
    $("lien-utilisation").href = s.type === "base" ? `/utilisation?ecg=${s.ecg_id}` : "/utilisation";
  },
};

function erreurFil(message) {
  $("fil").querySelector(".erreur")?.remove();
  if (message) $("fil").append(el("p", { class: "erreur", style: { width: "100%", margin: 0 } }, message));
}

async function changerGraine(i) {
  try {
    await chargerPoids(i);
  } catch (e) { erreurFil(e.message); return; }
  F.graine = i;
  try { localStorage.setItem(CLE_GRAINE, String(i)); } catch (e) { /* stockage indisponible */ }
  toutMettreAJour();
}

async function autreEcg() {
  const bouton = $("fil-hasard");
  bouton.disabled = true;
  erreurFil(null);
  try {
    if (!F.catalogue) F.catalogue = await api("/api/catalogue");
    if (!F.catalogue.donnees_brutes) throw new Error("Signaux bruts de PTB-XL introuvables : importez un fichier dans la fenêtre 1.");
    const liste = F.catalogue.ecg;
    const id = liste[Math.floor(Math.random() * liste.length)][0];
    F.analyse = await api(`/api/ecg/${id}`);
    memoriserAnalyse(F.analyse);
    history.replaceState(null, "", `?ecg=${id}${location.hash}`);
    toutMettreAJour();
  } catch (e) {
    erreurFil(e.message);
  } finally {
    bouton.disabled = false;
  }
}

/* ============================================================
   1. Vue d'ensemble
   ============================================================ */

const Vue = {
  init() {},
  maj() {
    const a = F.analyse, g = graineCourante();
    const k = 10 + 1; // II, écart-type
    const n1 = F.passe.act[1].filter((v) => v > 0).length;
    const n2 = F.passe.act[2].filter((v) => v > 0).length;
    const meilleure = g.probabilites.indexOf(Math.max(...g.probabilites));
    $("pv-ecg").textContent = a.source.ecg_id ? `ECG ${a.source.ecg_id}` : a.source.nom;
    $("pv-stat").textContent = `ex. II, σ = ${fr(a.caracteristiques.II_std, 3)} mV`;
    $("pv-entree").textContent = `→ z = ${fr(a.entrees.normalisees[k], 2)}`;
    $("pv-c1").textContent = `${n1} actifs`;
    $("pv-c2").textContent = `${n2} actifs`;
    $("pv-sortie").textContent = `${CLASSES[meilleure]} : ${fr(g.probabilites[meilleure], 2)}`;
    $("pv-decision").textContent = CLASSES.filter((c, i) => g.decisions[i]).join(", ");
  },
};

/* ============================================================
   2. Le neurone artificiel
   ============================================================ */

const INDICES = ["₁", "₂", "₃"];

const Neurone = {
  x: [0.8, -0.5, 1.2],
  w: [0.9, -1.1, 0.4],
  b: -0.3,
  act: "relu",

  init() {
    const zone = $("neurone-controles");
    const curseur = (nom, valeur, maj) => {
      const sortie = el("output", {}, fr(valeur, 2));
      const entree = el("input", { type: "range", min: -2, max: 2, step: 0.05, value: valeur, "aria-label": nom });
      entree.addEventListener("input", () => { maj(Number(entree.value)); sortie.textContent = fr(Number(entree.value), 2); this.formule(); });
      return el("label", { class: "curseur" }, el("span", {}, nom), entree, sortie);
    };
    for (let i = 0; i < 3; i++) {
      zone.append(curseur(`x${INDICES[i]}`, this.x[i], (v) => { this.x[i] = v; }));
      zone.append(curseur(`w${INDICES[i]}`, this.w[i], (v) => { this.w[i] = v; }));
    }
    zone.append(curseur("b", this.b, (v) => { this.b = v; }));
    const boutons = $("neurone-activation").querySelectorAll("button");
    boutons.forEach((b) => b.addEventListener("click", () => {
      this.act = b.dataset.act;
      boutons.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      this.formule();
    }));
    this.formule();
    boucleVisible($("neurone-canvas"), (t) => this.dessiner(t));
  },
  maj() {},

  z() { return this.x.reduce((s, v, i) => s + v * this.w[i], this.b); },
  f(z) { return this.act === "relu" ? relu(z) : sigmoide(z); },

  formule() {
    const par = (v) => (v < 0 ? `(${fr(v, 2)})` : fr(v, 2));
    const termes = this.x.map((v, i) => `${par(this.w[i])}×${par(v)}`).join(" + ");
    const z = this.z();
    const nom = this.act === "relu" ? "ReLU" : "σ";
    $("neurone-formule").textContent =
      `z = ${termes} + ${par(this.b)} = ${fr(z, 2)}   →   a = ${nom}(${fr(z, 2)}) = ${fr(this.f(z), 2)}`;
  },

  dessiner(t) {
    const p = palette();
    const canvas = $("neurone-canvas");
    const etroit = canvas.clientWidth < 520;
    const { ctx, l, h } = contexte(canvas, etroit ? 270 : 300);
    const rIn = etroit ? 18 : 22;
    const xIn = etroit ? 44 : 60;
    const ys = [0.2, 0.5, 0.8].map((f) => h * f);
    const xSum = l * (etroit ? 0.45 : 0.47), ySum = h * 0.5, rSum = etroit ? 20 : 26;
    const boiteL = Math.min(170, l * 0.25), boiteH = etroit ? 96 : 116;
    const boiteX = xSum + rSum + (l * 0.08), boiteY = ySum - boiteH / 2;
    const xOut = l - (etroit ? 30 : 44), rOut = etroit ? 18 : 24;
    const z = this.z(), a = this.f(z);
    const signe = (v) => (v >= 0 ? p.orange : p.accent);

    // Liens entrées -> somme, avec particules proportionnelles à x·w.
    this.x.forEach((xi, i) => {
      const x1 = xIn + rIn, y1 = ys[i], x2 = xSum - rSum, y2 = ySum;
      ligne(ctx, x1, y1, x2, y2, alpha(signe(this.w[i]), 0.85), 1 + 2.2 * Math.abs(this.w[i]));
      const prod = xi * this.w[i];
      for (let k = 0; k < 3; k++) {
        const ph = (t / 1900 + k / 3 + i * 0.13) % 1;
        const r = 2 + 3.2 * Math.min(1, Math.abs(prod));
        ctx.globalAlpha = Math.sin(Math.PI * ph);
        disque(ctx, interpole(x1, x2, ph), interpole(y1, y2, ph), r, signe(prod));
        ctx.globalAlpha = 1;
      }
      etiquette(ctx, `w${INDICES[i]} = ${fr(this.w[i], 2)}`, interpole(x1, x2, 0.42), interpole(y1, y2, 0.42) - 13, signe(this.w[i]), { taille: etroit ? 10 : 11 });
      disque(ctx, xIn, ys[i], rIn, zc(xi, 2), p.bordFort);
      ecrire(ctx, fr(xi, 2), xIn, ys[i] + 0.5, { taille: etroit ? 10 : 12, graisse: 650, aligne: "center", base: "middle", couleur: Math.abs(xi) > 1.1 ? "#fff" : p.texte });
      ecrire(ctx, `x${INDICES[i]}`, xIn - rIn - 6, ys[i] + 0.5, { taille: 13, aligne: "right", base: "middle", couleur: p.texte2 });
    });

    // Biais.
    ligne(ctx, xSum, 26, xSum, ySum - rSum - 4, p.texte3, 1.4, [4, 3]);
    ecrire(ctx, `b = ${fr(this.b, 2)}`, xSum, 18, { taille: 12, aligne: "center", couleur: p.texte2, graisse: 600 });

    // Somme -> activation -> sortie.
    const phSortie = ((t / 1900) + 0.5) % 1;
    ligne(ctx, xSum + rSum, ySum, boiteX, ySum, p.bordFort, 2);
    ligne(ctx, boiteX + boiteL, ySum, xOut - rOut, ySum, p.bordFort, 2);
    ctx.globalAlpha = Math.sin(Math.PI * phSortie);
    disque(ctx, interpole(xSum + rSum, boiteX, phSortie), ySum, 2.5 + 2.5 * Math.min(1, Math.abs(z)), signe(z));
    if (a > 0) disque(ctx, interpole(boiteX + boiteL, xOut - rOut, phSortie), ySum, 2.5 + 2.5 * Math.min(1, a), p.accent);
    ctx.globalAlpha = 1;

    disque(ctx, xSum, ySum, rSum, p.surface, p.texte, 1.5);
    ecrire(ctx, "Σ", xSum, ySum + 1, { taille: etroit ? 18 : 22, graisse: 600, aligne: "center", base: "middle" });
    ecrire(ctx, `z = ${fr(z, 2)}`, xSum, ySum + rSum + 18, { taille: 12, graisse: 650, aligne: "center" });

    // Boîte d'activation avec le graphe de f.
    rectArrondi(ctx, boiteX, boiteY, boiteL, boiteH, 10);
    ctx.fillStyle = p.surface;
    ctx.fill();
    ctx.strokeStyle = p.bordFort;
    ctx.lineWidth = 1;
    ctx.stroke();
    ecrire(ctx, this.act === "relu" ? "ReLU" : "sigmoïde", boiteX + 8, boiteY + 15, { taille: 11, graisse: 650, couleur: p.texte2 });
    const gx0 = boiteX + 10, gx1 = boiteX + boiteL - 10, gy0 = boiteY + 22, gy1 = boiteY + boiteH - 10;
    const zMax = 4, fMax = this.act === "relu" ? 4 : 1;
    const X = (v) => interpole(gx0, gx1, (v + zMax) / (2 * zMax));
    const Y = (v) => interpole(gy1, gy0, v / fMax);
    ligne(ctx, gx0, gy1, gx1, gy1, p.bord, 1);
    ligne(ctx, X(0), gy0, X(0), gy1, p.bord, 1);
    ctx.beginPath();
    for (let i = 0; i <= 60; i++) {
      const v = -zMax + (2 * zMax * i) / 60;
      const y = Y(Math.min(fMax, this.f(v)));
      i ? ctx.lineTo(X(v), y) : ctx.moveTo(X(v), y);
    }
    ctx.strokeStyle = p.accent;
    ctx.lineWidth = 2;
    ctx.stroke();
    const zc_ = limite(z, -zMax, zMax);
    ligne(ctx, X(zc_), gy1, X(zc_), Y(Math.min(fMax, this.f(zc_))), p.texte3, 1, [3, 3]);
    disque(ctx, X(zc_), Y(Math.min(fMax, this.f(zc_))), 4.5, p.orange, p.surface, 1.5);

    // Sortie.
    const intensite = this.act === "relu" ? Math.min(1, a / 2.5) : a;
    disque(ctx, xOut, ySum, rOut, alpha(p.accent, 0.12 + 0.88 * intensite), p.accent, 1.5);
    ecrire(ctx, "a", xOut, ySum + 1, { taille: 15, graisse: 650, aligne: "center", base: "middle", couleur: intensite > 0.55 ? "#fff" : p.texte });
    ecrire(ctx, `a = ${fr(a, 2)}`, xOut, ySum + rOut + 18, { taille: 12, graisse: 650, aligne: "center" });
    if (this.act === "relu" && z <= 0) ecrire(ctx, "inactif", xOut, ySum + rOut + 33, { taille: 11, aligne: "center", couleur: p.texte3 });
  },
};

/* ============================================================
   3.1 Du signal aux statistiques
   ============================================================ */

const ORDRE_STATS = ["mean", "std", "rms", "median", "iqr", "min", "max", "skewness", "kurtosis", "line_length"];

const DEFINITIONS = {
  mean: ["Valeur moyenne des 5 000 échantillons. Proche de 0 quand la ligne de base est centrée.", "μ = (1/N) Σ xᵢ"],
  std: ["Dispersion des valeurs autour de la moyenne.", "σ = √[ (1/N) Σ (xᵢ − μ)² ]"],
  rms: ["Valeur efficace : mesure de l'énergie moyenne du signal. Elle vérifie RMS² = σ² + μ².", "RMS = √[ (1/N) Σ xᵢ² ]"],
  median: ["Valeur qui partage les échantillons en deux moitiés. Les pics brefs (QRS) la déplacent peu.", "50 % des xᵢ ≤ médiane"],
  iqr: ["Écart entre le 3ᵉ et le 1ᵉʳ quartile : largeur de la moitié centrale des valeurs, insensible aux extrêmes.", "IQR = Q3 − Q1"],
  min: ["Plus petite valeur du tracé, souvent au creux d'une onde Q ou S.", "min = minᵢ xᵢ"],
  max: ["Plus grande valeur du tracé, souvent au sommet d'une onde R.", "max = maxᵢ xᵢ"],
  skewness: ["Asymétrie de la distribution des amplitudes. Positive : les grandes valeurs positives (pics R) sont rares mais éloignées. Nulle pour une distribution symétrique.", "G₁ = √[N(N−1)]/(N−2) · m₃ / m₂^{3/2}"],
  kurtosis: ["Aplatissement en excès : élevé quand quelques valeurs très éloignées (pics brefs) s'ajoutent à une masse de valeurs proches. Vaut 0 pour une loi normale.", "G₂ : estimateur corrigé de m₄ / m₂² − 3"],
  line_length: ["Somme des variations entre échantillons successifs : la longueur du tracé. Elle augmente avec les oscillations rapides et le bruit.", "L = Σ |xᵢ₊₁ − xᵢ|"],
};

const Stats = {
  derivation: 1,
  etape: 0,
  auto: true,
  debut: 0,
  cache: null,

  init() {
    const choix = $("stats-derivation");
    choix.replaceChildren(...derivations().map((d, i) => el("option", { value: i }, d)));
    choix.value = String(this.derivation);
    choix.addEventListener("change", () => { this.derivation = Number(choix.value); this.maj(); });
    $("stats-puces").replaceChildren(...ORDRE_STATS.map((s, i) =>
      el("button", { type: "button", "aria-pressed": String(i === this.etape),
        onclick: () => { this.choisir(i); this.mettreEnPause(true); } }, STAT_INFO[s].nom.replace(/ \(.+\)/, ""))));
    $("stats-lecture").addEventListener("click", () => this.mettreEnPause(this.auto));
    boucleVisible($("stats-canvas"), (t) => this.image(t));
  },
  maj() { this.preparer(); this.debut = performance.now(); this.legende(); this.dessiner(1); },

  mettreEnPause(pause) {
    this.auto = !pause;
    $("stats-lecture").textContent = this.auto ? "Pause" : "Lecture";
    if (this.auto) this.debut = performance.now();
  },
  choisir(i) {
    this.etape = i;
    this.debut = performance.now();
    $("stats-puces").querySelectorAll("button").forEach((b, k) => b.setAttribute("aria-pressed", String(k === i)));
    this.legende();
    if (this.cache) this.dessiner(mouvementReduit() ? 1 : 0);
  },
  duree() { return ORDRE_STATS[this.etape] === "line_length" ? 5600 : 3800; },

  preparer() {
    const v = F.analyse.signal.valeurs[this.derivation];
    const tri = Float64Array.from(v).sort();
    const nom = derivations()[this.derivation];
    const c = (s) => F.analyse.caracteristiques[`${nom}_${s}`];
    let iMin = 0, iMax = 0;
    v.forEach((x, i) => { if (x < v[iMin]) iMin = i; if (x > v[iMax]) iMax = i; });
    const cumul = new Float64Array(v.length);
    for (let i = 1; i < v.length; i++) cumul[i] = cumul[i - 1] + Math.abs(v[i] - v[i - 1]);
    const nb = 40, min = tri[0], max = tri[tri.length - 1];
    const histo = new Array(nb).fill(0);
    v.forEach((x) => { histo[Math.min(nb - 1, Math.floor(((x - min) / (max - min || 1)) * nb))] += 1; });
    const mu = c("mean"), sigma = c("std");
    const queues = v.filter((x) => Math.abs(x - mu) > 3 * sigma).length / v.length;
    this.cache = { v, nom, c, q1: quantile(tri, 0.25), q3: quantile(tri, 0.75), iMin, iMax, cumul, histo, min, max, queues };
  },

  legende() {
    if (!this.cache) return;
    const s = ORDRE_STATS[this.etape];
    const info = STAT_INFO[s];
    const valeur = this.cache.c(s);
    const [texte, formule] = DEFINITIONS[s];
    $("stats-legende").replaceChildren(
      el("h4", {}, info.nom, " ", el("span", { class: "discret" }, `· dérivation ${this.cache.nom}`)),
      el("div", { class: "valeur" }, `${frAuto(valeur)}${info.unite ? " " + info.unite : ""}`),
      el("p", {}, texte),
      el("code", {}, formule));
  },

  image(t) {
    if (!this.cache) return;
    let p = (t - this.debut) / this.duree();
    if (p >= 1 && this.auto && !mouvementReduit()) {
      this.choisir((this.etape + 1) % ORDRE_STATS.length);
      p = 0;
    }
    this.dessiner(mouvementReduit() ? 1 : limite(p, 0, 1));
  },

  dessiner(p) {
    const P = palette();
    const C = this.cache;
    const canvas = $("stats-canvas");
    const { ctx, l, h } = contexte(canvas, 290);
    const lH = l < 480 ? 58 : 84;
    const x0 = 46, x1 = l - lH - 18, y0 = 14, y1 = h - 30;
    const xh0 = x1 + 12, xh1 = l - 8;
    const etendue = C.max - C.min || 1;
    const bas = C.min - 0.06 * etendue, haut = C.max + 0.1 * etendue;
    const Y = (v) => interpole(y1, y0, (v - bas) / (haut - bas));
    const X = (i) => interpole(x0, x1, i / (C.v.length - 1));
    const s = ORDRE_STATS[this.etape];
    const e = adoucir(limite(p / 0.45, 0, 1));

    // Axes et graduations.
    const { valeurs, decimales } = graduations(bas, haut, 5);
    valeurs.forEach((v) => {
      ligne(ctx, x0, Y(v), x1, Y(v), P.bord, 1);
      ecrire(ctx, fr(v, decimales), x0 - 6, Y(v) + 0.5, { taille: 10, aligne: "right", base: "middle", couleur: P.texte3 });
    });
    ecrire(ctx, "mV", 8, y0 + 4, { taille: 10, couleur: P.texte3, base: "top" });
    for (let sec = 0; sec <= 10; sec += 2) {
      ecrire(ctx, `${sec} s`, interpole(x0, x1, sec / 10), y1 + 16, { taille: 10, aligne: "center", couleur: P.texte3 });
    }
    if (bas < 0 && haut > 0) ligne(ctx, x0, Y(0), x1, Y(0), P.texte3, 1, [2, 3]);

    // Histogramme des amplitudes (à droite).
    const histoActif = ["iqr", "skewness", "kurtosis"].includes(s);
    const hMax = Math.max(...C.histo);
    const hauteurCase = (Y(C.min) - Y(C.max)) / C.histo.length;
    C.histo.forEach((n, k) => {
      const v = C.min + (k + 0.5) * (etendue / C.histo.length);
      const yb = Y(C.min + ((k + 1) * etendue) / C.histo.length);
      let couleur = alpha(P.texte3, 0.3);
      let longueur = (n / hMax) * (xh1 - xh0);
      if (histoActif) {
        couleur = alpha(P.accent, 0.55);
        if (s === "iqr" && (v < C.q1 || v > C.q3)) couleur = alpha(P.texte3, 0.3);
        if (s === "kurtosis" && Math.abs(v - C.c("mean")) > 3 * C.c("std")) couleur = P.orange;
        longueur *= s === "skewness" || s === "kurtosis" ? e : 1;
      }
      ctx.fillStyle = couleur;
      ctx.fillRect(xh0, yb + 0.5, Math.max(n ? 1 : 0, longueur), Math.max(1, hauteurCase - 1));
    });
    ecrire(ctx, "répartition", (xh0 + xh1) / 2, y1 + 16, { taille: 10, aligne: "center", couleur: P.texte3 });

    // Signal (ou tracé progressif pour la longueur de ligne).
    const k = s === "line_length" ? Math.floor(limite(p / 0.85, 0, 1) * (C.v.length - 1)) : -1;
    ctx.beginPath();
    C.v.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
    ctx.strokeStyle = s === "line_length" ? alpha(P.texte3, 0.6) : P.texte2;
    ctx.lineWidth = 1;
    ctx.stroke();

    const fin = interpole(x0, x1, e);
    const accent = P.accent;
    switch (s) {
      case "mean":
        ligne(ctx, x0, Y(C.c("mean")), fin, Y(C.c("mean")), accent, 2.2);
        if (e > 0.9) etiquette(ctx, `μ = ${fr(C.c("mean"), 3)} mV`, x1 - 60, Y(C.c("mean")) - 14, accent);
        break;
      case "std": {
        const m = C.c("mean"), sd = C.c("std") * e;
        ctx.fillStyle = alpha(accent, 0.14);
        ctx.fillRect(x0, Y(m + sd), x1 - x0, Y(m - sd) - Y(m + sd));
        ligne(ctx, x0, Y(m + sd), x1, Y(m + sd), accent, 1.2, [5, 4]);
        ligne(ctx, x0, Y(m - sd), x1, Y(m - sd), accent, 1.2, [5, 4]);
        ligne(ctx, x0, Y(m), x1, Y(m), accent, 1.6);
        if (e > 0.9) etiquette(ctx, `μ ± σ (σ = ${fr(C.c("std"), 3)} mV)`, x1 - 90, Y(m + sd) - 14, accent);
        break;
      }
      case "rms": {
        const r = C.c("rms") * e;
        ligne(ctx, x0, Y(0), x1, Y(0), P.texte3, 1.2);
        ligne(ctx, x0, Y(r), x1, Y(r), P.orange, 1.8, [6, 4]);
        ligne(ctx, x0, Y(-r), x1, Y(-r), P.orange, 1.8, [6, 4]);
        if (e > 0.9) etiquette(ctx, `± RMS = ${fr(C.c("rms"), 3)} mV`, x1 - 74, Y(r) - 14, P.orange);
        break;
      }
      case "median":
        ligne(ctx, x0, Y(C.c("median")), fin, Y(C.c("median")), accent, 2.2);
        if (e > 0.9) etiquette(ctx, `médiane = ${fr(C.c("median"), 3)} mV`, x1 - 76, Y(C.c("median")) - 14, accent);
        break;
      case "iqr": {
        const m = C.c("median");
        const a = interpole(m, C.q3, e), b = interpole(m, C.q1, e);
        ctx.fillStyle = alpha(accent, 0.2);
        ctx.fillRect(x0, Y(a), x1 - x0, Y(b) - Y(a));
        ligne(ctx, x0, Y(a), x1, Y(a), accent, 1.2);
        ligne(ctx, x0, Y(b), x1, Y(b), accent, 1.2);
        if (e > 0.9) etiquette(ctx, `Q1 → Q3 : 50 % des échantillons`, x0 + 120, Y(a) - 14, accent);
        break;
      }
      case "min":
      case "max": {
        const v = C.c(s), i = s === "min" ? C.iMin : C.iMax;
        ligne(ctx, x0, Y(v), x1, Y(v), alpha(P.orange, e), 1.4, [5, 4]);
        disque(ctx, X(i), Y(v), 3 + 4 * e, P.orange, P.surface, 1.5);
        if (e > 0.9) etiquette(ctx, `${s} = ${fr(v, 3)} mV`, limite(X(i), x0 + 50, x1 - 50), Y(v) + (s === "min" ? -16 : 18), P.orange);
        break;
      }
      case "skewness":
      case "kurtosis": {
        const m = C.c("mean");
        ligne(ctx, xh0 - 4, Y(m), xh1, Y(m), accent, 1.5);
        if (s === "kurtosis") {
          for (const borne of [m + 3 * C.c("std"), m - 3 * C.c("std")]) {
            if (borne >= C.min && borne <= C.max) ligne(ctx, xh0 - 4, Y(borne), xh1, Y(borne), P.orange, 1, [3, 3]);
          }
        }
        if (e > 0.9) {
          const texte = s === "skewness"
            ? `asymétrie = ${fr(C.c("skewness"), 2)}`
            : `kurtosis = ${fr(C.c("kurtosis"), 2)} · ${pourcent(C.queues, 1)} des échantillons au-delà de μ ± 3σ`;
          etiquette(ctx, texte, x0 + 8, y0 + 14, s === "skewness" ? accent : P.orange, { aligne: "left" });
        }
        break;
      }
      case "line_length": {
        ctx.beginPath();
        for (let i = 0; i <= k; i++) (i ? ctx.lineTo(X(i), Y(C.v[i])) : ctx.moveTo(X(i), Y(C.v[i])));
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1.8;
        ctx.stroke();
        disque(ctx, X(k), Y(C.v[k]), 4.5, P.orange, P.surface, 1.5);
        etiquette(ctx, `longueur parcourue : ${fr(C.cumul[k], 1)} mV`, x0 + 8, y0 + 14, accent, { aligne: "left" });
        break;
      }
    }
  },
};

/* ============================================================
   3.2 La normalisation
   ============================================================ */

const Norm = {
  entree: 11, // II, écart-type
  p: 0,
  annuler: null,

  init() {
    const choix = $("norm-entree");
    derivations().forEach((d, r) => {
      choix.append(el("optgroup", { label: d }, ...STATS_MODELE.map((s, j) =>
        el("option", { value: r * 10 + j }, `${d} · ${STAT_INFO[s].nom}`))));
    });
    choix.value = String(this.entree);
    choix.addEventListener("change", () => { this.entree = Number(choix.value); this.maj(); });
    $("norm-bouton").addEventListener("click", () => {
      const cible = this.p < 0.5 ? 1 : 0;
      const depart = this.p;
      if (this.annuler) this.annuler();
      $("norm-bouton").textContent = cible ? "Revenir aux valeurs brutes" : "Normaliser";
      if (mouvementReduit()) { this.p = cible; this.dessiner(); return; }
      this.annuler = animer(1500, (e) => { this.p = interpole(depart, cible, adoucir(e)); this.dessiner(); });
    });
  },
  maj() { this.dessiner(); this.formule(); },

  donnees() {
    const nom = MODELE_ENTREES()[this.entree];
    const d = F.distributions.entrees[nom];
    const e = F.analyse.entrees;
    return { nom, d, x: e.brutes[this.entree], mu: e.moyennes[this.entree], sigma: e.ecarts_types[this.entree], z: e.normalisees[this.entree] };
  },

  formule() {
    const { nom, x, mu, sigma, z, d } = this.donnees();
    const { stat } = nomEntree(this.entree);
    const unite = STAT_INFO[stat].unite ? ` ${STAT_INFO[stat].unite}` : "";
    $("norm-formule").textContent = `z = (x − μ) / σ = (${frAuto(x)} − ${frAuto(mu)}) / ${frAuto(sigma)} = ${fr(z, 2)}`;
    $("norm-note").textContent = `Histogramme : ${nom} sur les ${entier(F.distributions.n_ecg)} ECG du train, entre les percentiles 0,5 et 99,5`
      + ` (${d.hors_bornes[0]} valeurs plus basses et ${d.hors_bornes[1]} plus hautes non représentées). Unité brute :${unite || " sans unité"}.`;
  },

  dessiner() {
    const P = palette();
    const { x, mu, sigma, z, d } = this.donnees();
    const canvas = $("norm-canvas");
    const { ctx, l, h } = contexte(canvas, 264);
    const x0 = 22, x1 = l - 22, yHaut = 62, yAxe = h - 56;
    const lo = Math.min(d.bornes[0], mu - 1.08 * sigma), hi = Math.max(d.bornes[1], mu + 1.08 * sigma);
    const X = (v) => interpole(x0, x1, (v - lo) / (hi - lo));
    const p = this.p;

    // Histogramme (identique dans les deux échelles : la forme ne change pas).
    const n = d.effectifs.length, maxN = Math.max(...d.effectifs);
    const largeur = (d.bornes[1] - d.bornes[0]) / n;
    d.effectifs.forEach((c, k) => {
      const a = X(d.bornes[0] + k * largeur), b = X(d.bornes[0] + (k + 1) * largeur);
      const hb = (c / maxN) * (yAxe - yHaut);
      ctx.fillStyle = alpha(P.texte3, 0.35);
      ctx.fillRect(a + 0.5, yAxe - hb, Math.max(1, b - a - 1), hb);
    });
    ligne(ctx, x0, yAxe, x1, yAxe, P.texte2, 1);

    // Graduations brutes puis en écarts-types.
    const brutes = graduations(lo, hi, 5);
    ctx.globalAlpha = 1 - p;
    brutes.valeurs.forEach((v) => {
      ligne(ctx, X(v), yAxe, X(v), yAxe + 5, P.texte3, 1);
      ecrire(ctx, fr(v, brutes.decimales), X(v), yAxe + 18, { taille: 10.5, aligne: "center", couleur: P.texte2 });
    });
    ctx.globalAlpha = p;
    const kMin = Math.ceil((lo - mu) / sigma), kMax = Math.floor((hi - mu) / sigma);
    const pasK = kMax - kMin > 12 ? 5 : kMax - kMin > 6 ? 2 : 1;
    for (let k = Math.ceil(kMin / pasK) * pasK; k <= kMax; k += pasK) {
      ligne(ctx, X(mu + k * sigma), yAxe, X(mu + k * sigma), yAxe + 5, P.accent, 1);
      ecrire(ctx, k > 0 ? `+${k}` : fr(k, 0), X(mu + k * sigma), yAxe + 18, { taille: 10.5, aligne: "center", couleur: P.accent, graisse: 600 });
    }
    ctx.globalAlpha = 1;
    ecrire(ctx, p < 0.5 ? "valeur brute x" : "valeur normalisée z", x1, yAxe + 36, { taille: 10.5, aligne: "right", couleur: P.texte3 });

    // Moyenne et ± un écart-type du train.
    ligne(ctx, X(mu), yHaut - 8, X(mu), yAxe, P.accent, 1.6);
    ligne(ctx, X(mu - sigma), yHaut, X(mu - sigma), yAxe, P.accent, 1, [4, 4]);
    ligne(ctx, X(mu + sigma), yHaut, X(mu + sigma), yAxe, P.accent, 1, [4, 4]);
    const lab = (brut, norme, v) => {
      ctx.globalAlpha = 1 - p;
      ecrire(ctx, brut, X(v), yHaut - 14, { taille: 10.5, aligne: "center", couleur: P.accent, graisse: 600 });
      ctx.globalAlpha = p;
      ecrire(ctx, norme, X(v), yHaut - 14, { taille: 10.5, aligne: "center", couleur: P.accent, graisse: 600 });
      ctx.globalAlpha = 1;
    };
    lab("μ", "0", mu);
    lab("μ − σ", "−1", mu - sigma);
    lab("μ + σ", "+1", mu + sigma);

    // Cet ECG.
    const dehors = x < lo ? -1 : x > hi ? 1 : 0;
    const xe = dehors ? (dehors < 0 ? x0 + 4 : x1 - 4) : X(x);
    ligne(ctx, xe, 26, xe, yAxe, P.orange, 2);
    disque(ctx, xe, yAxe, 5, P.orange, P.surface, 1.5);
    const texteBrut = `cet ECG : ${frAuto(x)}${dehors ? (dehors > 0 ? " →" : " ←") : ""}`;
    const texteZ = `cet ECG : z = ${fr(z, 2)}${dehors ? (dehors > 0 ? " →" : " ←") : ""}`;
    const xl = limite(xe, x0 + 70, x1 - 70);
    ctx.globalAlpha = 1 - p;
    etiquette(ctx, texteBrut, xl, 15, P.orange);
    ctx.globalAlpha = p;
    etiquette(ctx, texteZ, xl, 15, P.orange);
    ctx.globalAlpha = 1;
  },
};

const MODELE_ENTREES = () => F.modele.entrees;

/* ============================================================
   3.3 Les 120 entrées
   ============================================================ */

const Grille = {
  init() {
    this.lecteur = lecteur(7800, (t) => this.dessiner(t));
    $("grille-rejouer").addEventListener("click", () => this.lecteur.jouer());
    quandVisible($("grille-canvas"), () => this.lecteur.jouer(), 0.4);
  },
  maj() { if (!this.lecteur.enCours) this.dessiner(Infinity); },

  dessiner(t) {
    const P = palette();
    const canvas = $("grille-canvas");
    const etroit = canvas.clientWidth < 560;
    const { ctx, l, h } = contexte(canvas, etroit ? 400 : 440);
    const z = F.analyse.entrees.normalisees;
    const D = derivations();
    const gx = 38, gy = 62;
    const c = Math.max(10, Math.min(26, (l * 0.6 - gx) / 10 - 2));
    const pas = c + 2;
    const x10 = (jp) => gx + jp * pas;
    // Bande des 120 entrées, à droite.
    const bandeL = etroit ? 12 : 18, bandeX = l - (etroit ? 52 : 74), haut = 46, basB = h - 14;
    const sp = (basB - haut) / 120;
    const yBande = (k) => haut + k * sp;

    const eTete = prog(t, 0, 400);
    const eCouleur = prog(t, 1500, 900);
    const eVol = (k) => adoucir(prog(t, 2600 + k * 5, 900));
    const eFin = prog(t, 3600, 1200);

    ecrire(ctx, "12 dérivations × 10 statistiques = 120",
      gx, 22, { taille: 13, graisse: 650 });

    // En-têtes de colonnes et de lignes.
    ORDRE_STATS.forEach((s, j) => {
      ctx.save();
      ctx.globalAlpha = eTete;
      ctx.translate(x10(j) + c / 2 - 2, gy - 6);
      ctx.rotate(-Math.PI / 4);
      ecrire(ctx, STAT_INFO[s].court, 0, 0, { taille: 10, couleur: P.texte3 });
      ctx.restore();
    });
    ctx.globalAlpha = eTete;
    D.forEach((d, r) => ecrire(ctx, d, gx - 6, gy + r * pas + c / 2 + 0.5, { taille: 10, aligne: "right", base: "middle", couleur: P.texte2, graisse: 600 }));
    ctx.globalAlpha = 1;

    // Tableau : cellules neutres, puis colorées par la valeur normalisée de cet ECG.
    for (let r = 0; r < 12; r++) {
      for (let j = 0; j < 10; j++) {
        const apparition = prog(t, r * 130 + j * 18, 250);
        if (apparition <= 0) continue;
        const x = x10(j);
        const y = gy + r * pas;
        const taille = c * (0.6 + 0.4 * adoucir(apparition));
        ctx.globalAlpha = apparition;
        rectArrondi(ctx, x + (c - taille) / 2, y + (c - taille) / 2, taille, taille, 3);
        ctx.fillStyle = eCouleur > 0
          ? melangeRgb(P.surface3, zc(z[r * 10 + j]), eCouleur)
          : P.surface3;
        ctx.fill();
        ctx.strokeStyle = P.bordFort;
        ctx.lineWidth = 0.6;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // Bande des 120 entrées : contour, cases arrivées, copies en vol.
    if (prog(t, 2400, 300) > 0) {
      ctx.globalAlpha = prog(t, 2400, 300);
      rectArrondi(ctx, bandeX - bandeL / 2 - 2, haut - 2, bandeL + 4, basB - haut + 4, 4);
      ctx.fillStyle = P.surface;
      ctx.fill();
      ctx.strokeStyle = P.bordFort;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    for (let k = 0; k < 120; k++) {
      const v = eVol(k);
      if (v <= 0) continue;
      const r = Math.floor(k / 10), jp = k % 10;
      const x = interpole(x10(jp) + c / 2, bandeX, v);
      const y = interpole(gy + r * pas + c / 2, yBande(k) + sp / 2, v);
      const lw = interpole(c, bandeL, v), hw = interpole(c, Math.max(1, sp - 0.5), v);
      ctx.fillStyle = zc(z[k]);
      ctx.fillRect(x - lw / 2, y - hw / 2, lw, hw);
    }

    // Accolades par dérivation et titre de la bande.
    if (eFin > 0) {
      ctx.globalAlpha = eFin;
      ecrire(ctx, "120 entrées", bandeX, haut - 12, { taille: 12, graisse: 650, aligne: "center" });
      D.forEach((d, r) => {
        const ya = yBande(r * 10) + 1, yb = yBande(r * 10 + 10) - 1;
        const xb = bandeX + bandeL / 2 + 6;
        ctx.strokeStyle = P.texte3;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(xb, ya); ctx.lineTo(xb + 4, ya); ctx.lineTo(xb + 4, yb); ctx.lineTo(xb, yb);
        ctx.stroke();
        ecrire(ctx, d, xb + 8, (ya + yb) / 2 + 0.5, { taille: 9.5, base: "middle", couleur: P.texte2 });
      });
      const xFleche0 = x10(10) + 10, xFleche1 = bandeX - bandeL / 2 - 10, yF = gy + 6 * pas;
      if (xFleche1 - xFleche0 > 30) {
        ligne(ctx, xFleche0, yF, xFleche1, yF, P.texte3, 1.4);
        ctx.beginPath();
        ctx.moveTo(xFleche1, yF); ctx.lineTo(xFleche1 - 7, yF - 4); ctx.lineTo(xFleche1 - 7, yF + 4); ctx.closePath();
        ctx.fillStyle = P.texte3;
        ctx.fill();
        ecrire(ctx, "bout à bout", (xFleche0 + xFleche1) / 2, yF - 8, { taille: 10.5, aligne: "center", couleur: P.texte2 });
      }
      ctx.globalAlpha = 1;
    }
  },
};

/* ============================================================
   4. Les couches cachées
   ============================================================ */

const Reseau = {
  survol: null,
  dropout: false,
  minuterie: 0,
  lot: 0,
  masques: null,
  cleStatique: "",
  statique: null,

  init() {
    this.lecteur = lecteur(7400, (t) => this.dessiner(t));
    $("reseau-propager").addEventListener("click", () => { this.arreterDropout(); this.lecteur.jouer(); });
    $("reseau-dropout").addEventListener("click", () => (this.dropout ? this.arreterDropout() : this.demarrerDropout()));
    const canvas = $("reseau-canvas");
    canvas.addEventListener("mousemove", (e) => this.surSurvol(e));
    canvas.addEventListener("mouseleave", () => { this.survol = null; infobulle.cacher(); if (!this.lecteur.enCours && !this.dropout) this.dessiner(Infinity); });
    quandVisible(canvas, () => this.lecteur.jouer(), 0.35);
  },

  maj() {
    this.preparer();
    const g = graineCourante();
    const n1 = F.passe.act[1].filter((v) => v > 0).length;
    const n2 = F.passe.act[2].filter((v) => v > 0).length;
    $("reseau-actifs").innerHTML = `Pour cet ECG (graine ${g.graine}) : <b>${n1} neurones actifs sur 128</b> dans la couche 1 et <b>${n2} sur 64</b> dans la couche 2. Les autres ont une somme pondérée négative, que la ReLU ramène à 0.`;
    $("reseau-note").textContent = `Calcul refait dans le navigateur avec les ${entier(F.modele.architecture.parametres)} poids et biais du modèle : écart maximal avec le calcul de PyTorch ${frExp(F.ecartLocal)}.`;
    if (!this.lecteur.enCours && !this.dropout) this.dessiner(Infinity);
  },

  preparer() {
    const P = poidsCourants().couches;
    const act = F.passe.act;
    const tops = [600, 400, 320];
    this.contrib = P.map((couche, L) => {
      const liste = [];
      couche.W.forEach((rang, i) => rang.forEach((w, j) => { const c = w * act[L][j]; if (c !== 0) liste.push([j, i, c]); }));
      liste.sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]));
      return liste.slice(0, tops[L]);
    });
    this.cleStatique = "";
  },

  geometrie(l, h) {
    const tailles = [120, 128, 64, 2];
    const xs = [l * 0.07, l * 0.36, l * 0.62, l * 0.8];
    const haut = 52, bas = h - 18;
    return tailles.map((n, L) => {
      const sortiePas = Math.min(58, (bas - haut) / (n + 1));
      const pas = L === 3 ? sortiePas : (bas - haut) / n;
      const r = L === 3 ? 13 : Math.max(1.4, Math.min(L === 2 ? 5 : 3.4, pas * 0.42));
      return {
        n, x: xs[L], pas, r,
        y: (i) => L === 3
          ? h / 2 + (i - (n - 1) / 2) * sortiePas
          : haut + (i + 0.5) * pas,
      };
    });
  },

  /* Tous les liens, très pâles, dessinés une fois dans un canvas hors écran. */
  couchesStatiques(l, h, G) {
    const ratio = window.devicePixelRatio || 1;
    const cle = `${l}x${h}x${ratio}x${F.graine}x${palette().accent}`;
    if (cle === this.cleStatique && this.statique) return this.statique;
    const P = palette();
    const hors = document.createElement("canvas");
    hors.width = Math.round(l * ratio);
    hors.height = Math.round(h * ratio);
    const ctx = hors.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    poidsCourants().couches.forEach((couche, L) => {
      const a = G[L], b = G[L + 1];
      let wMax = 0;
      couche.W.forEach((rang) => rang.forEach((w) => { wMax = Math.max(wMax, Math.abs(w)); }));
      const base = L === 2 ? [0.06, 0.12, 0.22] : [0.01, 0.025, 0.055];
      for (const signe of [1, -1]) {
        for (let q = 0; q < 3; q++) {
          ctx.beginPath();
          couche.W.forEach((rang, i) => rang.forEach((w, j) => {
            if (Math.sign(w) !== signe) return;
            const n = Math.abs(w) / wMax;
            if ((q === 0 && n >= 0.25) || (q === 1 && (n < 0.25 || n >= 0.55)) || (q === 2 && n < 0.55)) return;
            ctx.moveTo(a.x, a.y(j));
            ctx.lineTo(b.x, b.y(i));
          }));
          ctx.strokeStyle = alpha(signe > 0 ? P.orange : P.accent, base[q]);
          ctx.lineWidth = 0.7;
          ctx.stroke();
        }
      }
    });
    this.statique = hors;
    this.cleStatique = cle;
    return hors;
  },

  dessiner(t) {
    const P = palette();
    const canvas = $("reseau-canvas");
    const { ctx, l, h } = contexte(canvas, canvas.clientWidth < 600 ? 470 : 580);
    const G = this.geometrie(l, h);
    this.G = G;
    const act = F.passe.act, pre = F.passe.pre, g = graineCourante();
    ctx.drawImage(this.couchesStatiques(l, h, G), 0, 0, l, h);

    const debutsLiens = [900, 3000, 4900], dureesLiens = [1500, 1300, 1000];
    const debutsNoeuds = [0, 2400, 4300, 5900];

    // Liens les plus contributifs, puis impulsions.
    this.contrib.forEach((liste, L) => {
      const e = prog(t, debutsLiens[L], dureesLiens[L]);
      if (e <= 0) return;
      const a = G[L], b = G[L + 1];
      const cMax = Math.abs(liste[0]?.[2] || 1);
      const n = Math.ceil(e * liste.length);
      for (let k = 0; k < n; k++) {
        const [j, i, c] = liste[k];
        ligne(ctx, a.x, a.y(j), b.x, b.y(i), alpha(c > 0 ? P.orange : P.accent, 0.08 + 0.6 * Math.abs(c) / cMax), L === 2 ? 1.4 : 0.9);
      }
      if (t !== Infinity) {
        for (let k = 0; k < Math.min(70, liste.length); k++) {
          const s = (t - debutsLiens[L] - k * 12) / 650;
          if (s <= 0 || s >= 1) continue;
          const [j, i, c] = liste[k];
          disque(ctx, interpole(a.x, b.x, s), interpole(a.y(j), b.y(i), s), L === 2 ? 3 : 2.2, c > 0 ? P.orange : P.accent);
        }
      }
    });

    // Liens entrants du neurone survolé.
    if (this.survol && this.survol.couche > 0 && t === Infinity) {
      const L = this.survol.couche, i = this.survol.i;
      const a = G[L - 1], b = G[L];
      const W = poidsCourants().couches[L - 1].W[i];
      const c = W.map((w, j) => w * act[L - 1][j]);
      const cMax = Math.max(...c.map(Math.abs)) || 1;
      c.forEach((v, j) => { if (v) ligne(ctx, a.x, a.y(j), b.x, b.y(i), alpha(v > 0 ? P.orange : P.accent, 0.15 + 0.85 * Math.abs(v) / cMax), 1.2); });
    }

    // Neurones.
    G.forEach((c, L) => {
      const eN = prog(t, debutsNoeuds[L], L === 0 ? 900 : 500);
      const valeurs = act[L];
      const vMax = Math.max(...valeurs.map(Math.abs)) || 1;
      for (let i = 0; i < c.n; i++) {
        const y = c.y(i);
        if (L === 0) {
          const visible = t >= i * 7.5;
          disque(ctx, c.x, y, c.r, visible ? zc(valeurs[i]) : P.surface, visible ? null : P.bordFort, 0.8);
        } else if (L < 3) {
          if (eN <= 0) { disque(ctx, c.x, y, c.r, P.surface, P.bordFort, 0.8); continue; }
          const v = valeurs[i];
          const couleur = v > 0 ? alpha(P.accent, eN * (0.3 + 0.7 * v / vMax)) : alpha(P.eteint.startsWith("#") ? P.eteint : "#d9d6ce", eN);
          disque(ctx, c.x, y, c.r, couleur, v > 0 ? null : P.bordFort, 0.6);
        } else {
          const cl = CLASSES[i];
          const p = g.probabilites[i];
          disque(ctx, c.x, y, c.r, eN > 0 ? alpha(P.classes[cl], 0.12 + 0.88 * p * eN) : P.surface, P.classes[cl], 1.5);
          if (eN > 0) {
            ctx.globalAlpha = eN;
            ecrire(ctx, cl, c.x + c.r + 8, y - 3, { taille: 12, graisse: 700, couleur: P.classes[cl] });
            const retenue = g.decisions[i] === 1;
            ecrire(ctx, `${fr(p, 2)}${retenue ? "  ✓ retenue" : ""}`, c.x + c.r + 8, y + 12, { taille: 11, couleur: retenue ? P.texte : P.texte3, graisse: retenue ? 650 : 400 });
            ctx.globalAlpha = 1;
          }
        }
      }
      if (this.survol && this.survol.couche === L) disque(ctx, c.x, c.y(this.survol.i), c.r + 3, null, P.texte, 1.5);
    });

    // Titres de colonnes et compteurs.
    const titres = ["Entrée · 120", "Cachée 1 · 128", "Cachée 2 · 64", "Sortie · 2"];
    G.forEach((c, L) => {
      ecrire(ctx, titres[L], c.x, 16, { taille: 11.5, graisse: 650, aligne: L === 0 ? "left" : "center", couleur: P.texte });
      if ((L === 1 || L === 2) && prog(t, debutsNoeuds[L], 500) > 0) {
        const n = act[L].filter((v) => v > 0).length;
        ctx.globalAlpha = prog(t, debutsNoeuds[L], 500);
        ecrire(ctx, `${n} actifs`, c.x, 33, { taille: 11, aligne: "center", couleur: P.accent, graisse: 600 });
        ctx.globalAlpha = 1;
      }
    });
    if (t === Infinity) {
      ecrire(ctx, "z normalisé", G[0].x, 33, { taille: 11, aligne: "left", couleur: P.texte3 });
    }
  },

  surSurvol(e) {
    if (!this.G || this.lecteur.enCours || this.dropout) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    let trouve = null;
    this.G.forEach((c, L) => {
      if (Math.abs(x - c.x) > Math.max(9, c.r + 4)) return;
      const i = Math.round((y - c.y(0)) / c.pas);
      if (i >= 0 && i < c.n && Math.abs(y - c.y(i)) <= Math.max(c.pas / 2, c.r + 2)) trouve = { couche: L, i };
    });
    const ancien = JSON.stringify(this.survol);
    this.survol = trouve;
    if (JSON.stringify(trouve) !== ancien) this.dessiner(Infinity);
    if (!trouve) { infobulle.cacher(); return; }
    infobulle.montrer(this.texteNeurone(trouve), e.clientX, e.clientY);
  },

  texteNeurone({ couche, i }) {
    const act = F.passe.act, pre = F.passe.pre, g = graineCourante(), a = F.analyse.entrees;
    if (couche === 0) {
      const { derivation, stat } = nomEntree(i);
      return `<b>Entrée ${i + 1} : ${derivation} · ${STAT_INFO[stat].nom}</b><br>valeur brute ${frAuto(a.brutes[i])} → z = ${fr(a.normalisees[i], 2)}`;
    }
    if (couche < 3) {
      const n = couche === 1 ? 120 : 128;
      const v = act[couche][i];
      return `<b>Couche cachée ${couche}, neurone ${i + 1}</b><br>somme pondérée de ${n} valeurs + biais : z = ${fr(pre[couche - 1][i], 2)}<br>ReLU → ${v > 0 ? fr(v, 2) : "0 : neurone inactif"}`;
    }
    const c = CLASSES[i];
    return `<b>${c} · ${CLASSE_INFO[c].nom}</b><br>logit ${fr(g.logits[i], 2)} → softmax → ${fr(g.probabilites[i], 3)}<br>seuil ${fr(g.seuils[i], 2)} → ${g.decisions[i] ? "retenue" : "non retenue"}`;
  },

  demarrerDropout() {
    this.lecteur.arreter();
    this.dropout = true;
    this.lot = 0;
    infobulle.cacher();
    $("reseau-dropout").setAttribute("aria-pressed", "true");
    $("reseau-dropout").textContent = "Revenir à l'utilisation";
    const echantillon = (n, k) => Array.from({ length: k }, () => Math.floor(Math.random() * n));
    this.echantillons = [echantillon(120 * 128, 1400), echantillon(128 * 64, 900), Array.from({ length: 320 }, (_, k) => k)];
    const tirage = () => {
      this.lot += 1;
      this.masques = [Array.from({ length: 128 }, () => Math.random() < 0.3), Array.from({ length: 64 }, () => Math.random() < 0.3)];
      this.dessinerDropout();
    };
    tirage();
    this.minuterie = setInterval(tirage, 1100);
  },

  arreterDropout() {
    if (!this.dropout) return;
    clearInterval(this.minuterie);
    this.dropout = false;
    $("reseau-dropout").setAttribute("aria-pressed", "false");
    $("reseau-dropout").textContent = "Voir le dropout (apprentissage)";
    this.dessiner(Infinity);
  },

  dessinerDropout() {
    const P = palette();
    const canvas = $("reseau-canvas");
    const { ctx, l, h } = contexte(canvas, canvas.clientWidth < 600 ? 470 : 580);
    const G = this.geometrie(l, h);
    const tailles = [[120, 128], [128, 64], [64, 2]];
    const eteint = (L, i) => (L === 1 || L === 2) && this.masques[L - 1][i];
    this.echantillons.forEach((liste, L) => {
      const [nIn] = tailles[L];
      ctx.beginPath();
      liste.forEach((k) => {
        const i = Math.floor(k / nIn), j = k % nIn;
        if (eteint(L, j) || eteint(L + 1, i)) return;
        ctx.moveTo(G[L].x, G[L].y(j));
        ctx.lineTo(G[L + 1].x, G[L + 1].y(i));
      });
      ctx.strokeStyle = alpha(P.texte3, L === 2 ? 0.3 : 0.12);
      ctx.lineWidth = 0.7;
      ctx.stroke();
    });
    G.forEach((c, L) => {
      for (let i = 0; i < c.n; i++) {
        const y = c.y(i);
        if (L === 0) disque(ctx, c.x, y, c.r, alpha(P.texte3, 0.6));
        else if (L === 3) disque(ctx, c.x, y, c.r, P.surface, P.classes[CLASSES[i]], 1.5);
        else if (eteint(L, i)) {
          const d = Math.max(2, c.r);
          ligne(ctx, c.x - d, y - d, c.x + d, y + d, P.rouge, 1.3);
          ligne(ctx, c.x - d, y + d, c.x + d, y - d, P.rouge, 1.3);
        } else disque(ctx, c.x, y, c.r, alpha(P.accent, 0.75));
      }
    });
    const n1 = this.masques[0].filter(Boolean).length, n2 = this.masques[1].filter(Boolean).length;
    ecrire(ctx, `Apprentissage, mini-lot n° ${this.lot} : ${n1} neurones éteints sur 128 et ${n2} sur 64 (tirage au hasard)`, l / 2, 16, { taille: 12, graisse: 650, aligne: "center" });
  },
};

/* ============================================================
   5.1 La sigmoïde
   ============================================================ */

const Sigmoide = {
  vu: false,
  init() {
    this.lecteur = lecteur(2900, (t) => this.dessiner(t));
    $("sigmoide-rejouer").addEventListener("click", () => this.lecteur.jouer());
    quandVisible($("sigmoide-canvas"), () => { this.vu = true; this.lecteur.jouer(); }, 0.4);
  },
  maj() {
    const g = graineCourante();
    $("sigmoide-table").replaceChildren(
      el("thead", {}, el("tr", {}, el("th", {}, "Classe"), el("th", {}, "logit"), el("th", {}, "softmax"))),
      el("tbody", {}, ...CLASSES.map((c, k) => el("tr", {},
        el("td", {}, pastilleClasse(c)), el("td", {}, fr(g.logits[k], 2)), el("td", {}, el("b", {}, fr(g.probabilites[k], 3)))))));
    if (this.vu) this.lecteur.jouer(); else this.dessiner(Infinity);
  },

  dessiner(t) {
    const P = palette();
    const g = graineCourante();
    const { ctx, l, h } = contexte($("sigmoide-canvas"), 300);
    const difference = g.logits[1] - g.logits[0];
    const zMax = limite(Math.ceil(Math.abs(difference)) + 1, 6, 14);
    const x0 = 40, x1 = l - 16, y0 = 18, y1 = h - 38;
    const X = (z) => interpole(x0, x1, (z + zMax) / (2 * zMax));
    const Y = (p) => interpole(y1, y0, p);

    for (const p of [0, 0.5, 1]) {
      ligne(ctx, x0, Y(p), x1, Y(p), P.bord, 1);
      ecrire(ctx, fr(p, p === 0.5 ? 1 : 0), x0 - 6, Y(p) + 0.5, { taille: 10.5, aligne: "right", base: "middle", couleur: P.texte3 });
    }
    const pas = zMax > 8 ? 4 : 2;
    for (let z = -Math.floor(zMax / pas) * pas; z <= zMax; z += pas) {
      ligne(ctx, X(z), y1, X(z), y1 + 4, P.texte3, 1);
      ecrire(ctx, z > 0 ? `+${z}` : fr(z, 0), X(z), y1 + 16, { taille: 10.5, aligne: "center", couleur: P.texte3 });
    }
    ligne(ctx, X(0), y0, X(0), y1, P.bord, 1);
    ecrire(ctx, "logit z", x1, y1 + 30, { taille: 10.5, aligne: "right", couleur: P.texte3 });
    ecrire(ctx, "P(classe)", x0 + 4, y0 - 4, { taille: 10.5, couleur: P.texte3 });

    const eC = prog(t, 0, 800);
    ctx.beginPath();
    for (let i = 0; i <= 200 * eC; i++) {
      const z = -zMax + (2 * zMax * i) / 200;
      i ? ctx.lineTo(X(z), Y(sigmoide(z))) : ctx.moveTo(X(z), Y(sigmoide(z)));
    }
    ctx.strokeStyle = P.texte2;
    ctx.lineWidth = 2.2;
    ctx.stroke();

    const ordre = CLASSES.map((c, k) => k).sort((a, b) => g.logits[a] - g.logits[b]);
    const places = [];
    ordre.forEach((k, rang) => {
      const c = CLASSES[k], z = k === 1 ? difference : -difference, p = g.probabilites[k];
      const debut = 800 + rang * 280;
      const e1 = adoucir(prog(t, debut, 650)), e2 = prog(t, debut + 650, 400);
      if (e1 <= 0) return;
      const xz = X(limite(z, -zMax, zMax)), yp = Y(p * e1);
      const couleur = P.classes[c];
      ligne(ctx, xz, y1, xz, yp, alpha(couleur, 0.8), 1.2, [3, 3]);
      if (e2 > 0) { ctx.globalAlpha = e2; ligne(ctx, x0, Y(p), xz, Y(p), alpha(couleur, 0.7), 1, [3, 3]); ctx.globalAlpha = 1; }
      disque(ctx, xz, y1, 3.5, couleur);
      disque(ctx, xz, yp, 5.5, couleur, P.surface, 1.5);
      if (e2 > 0) {
        let ye = Y(p) - 12;
        while (places.some(([xa, ya]) => Math.abs(xa - xz) < 46 && Math.abs(ya - ye) < 14)) ye -= 15;
        places.push([xz, ye]);
        ctx.globalAlpha = e2;
        ecrire(ctx, c, limite(xz, x0 + 18, x1 - 18), ye, { taille: 11, graisse: 700, aligne: "center", couleur });
        ctx.globalAlpha = 1;
      }
    });
  },
};

/* ============================================================
   5.2 Probabilités exclusives
   ============================================================ */

const Comparaison = {
  vu: false,
  init() { quandVisible($("comparaison"), () => { this.vu = true; rejouerCss($("comparaison")); }, 0.4); },
  maj() {
    const g = graineCourante();
    const m = Math.max(...g.logits);
    const ex = g.logits.map((z) => Math.exp(z - m));
    const somme = ex.reduce((a, b) => a + b, 0);
    const softmax = ex.map((v) => v / somme);
    const colonne = (titre, sousTitre, valeurs, utilisee) => el("div", { class: `colonne-proba ${utilisee ? "utilisee" : ""}` },
      el("h4", {}, titre, el("small", {}, sousTitre)),
      ...CLASSES.map((c, k) => el("div", { class: "barre-proba", style: { "--couleur": couleurClasse(c) } },
        el("b", {}, c),
        el("span", { class: "piste" }, el("span", { class: "plein", style: { width: `${(100 * valeurs[k]).toFixed(1)}%` } })),
        el("span", {}, fr(valeurs[k], 2)))),
      el("div", { class: "somme" }, el("span", {}, "Somme des probabilités"), el("b", {}, fr(valeurs.reduce((a, b) => a + b, 0), 2))));
    $("comparaison").replaceChildren(
      colonne("Softmax", "sortie du modèle : classes exclusives", softmax, true));
    if (this.vu) rejouerCss($("comparaison"));
  },
};

/* ============================================================
   5.3 Les seuils
   ============================================================ */

const Seuils = {
  vu: false,
  init() { quandVisible($("seuils"), () => { this.vu = true; this.jouer(); }, 0.5); },
  maj() {
    const g = graineCourante();
    $("seuils").replaceChildren(...CLASSES.map((c, k) => {
      const retenue = g.decisions[k] === 1;
      return el("div", { class: `ligne-seuil ${retenue ? "retenue" : ""}`, style: { "--couleur": couleurClasse(c) } },
        el("b", {}, c),
        el("div", { class: "piste" },
          el("div", { class: "plein", style: { width: `${(100 * g.probabilites[k]).toFixed(1)}%` } }),
          el("div", { class: "trait", style: { left: `${100 * g.seuils[k]}%` }, "data-texte": `seuil ${fr(g.seuils[k], 2)}` })),
        el("span", { class: "p" }, fr(g.probabilites[k], 2)),
        el("span", { class: "verdict" }, retenue
          ? el("span", { class: "pastille", style: { "--couleur": couleurClasse(c) } }, "retenue")
          : el("span", { class: "pastille sans-point" }, "non retenue")));
    }));
    const retenues = CLASSES.filter((c, k) => g.decisions[k]);
    $("seuils-regle").textContent =
      `Pour cet ECG (graine ${g.graine}), la classe pathologique est ${g.decisions[1] ? "retenue" : "écartée"} au seuil ${fr(g.seuils[1], 2)}. Classe prédite : ${retenues.join(", ")}.`;
    if (this.vu) this.jouer();
  },
  jouer() { rejouerCss($("seuils")); },
};

/* ============================================================
   5.4 Précision, rappel et F1 (résultats enregistrés du test)
   ============================================================ */

const Mesures = {
  classe: 0,
  vu: false,
  init() {
    quandVisible($("mesures"), () => { this.vu = true; this.jouer(); }, 0.4);
  },
  resultats() {
    return F.modele.test.par_graine.find((g) => g.graine === graineCourante().graine);
  },
  maj() {
    const r = this.resultats();
    $("mesures-classe").replaceChildren(...CLASSES.map((c, k) => el("button", {
      type: "button", "aria-pressed": String(k === this.classe), onclick: () => { this.classe = k; this.maj(); },
    }, c)));
    this.matrice(r);
    this.tableau(r);
    if (this.vu) this.jouer();
  },

  matrice(r) {
    const c = CLASSES[this.classe];
    const [[vn, fp], [fn, vp]] = r.par_classe[c].confusion_matrix;
    const cellule = (code, nom, n, juste) => el("div", {
      class: `cellule ${juste ? "juste" : "erreur-cas"}`, "data-code": code, title: nom,
    }, el("b", { "data-n": n }, entier(n)), el("span", {}, code));
    $("mesures-matrice").replaceChildren(
      el("span", {}), el("span", { class: "tete" }, `${c} retenue`), el("span", { class: "tete" }, "non retenue"),
      el("span", { class: "tete-ligne" }, "annotée"), cellule("VP", "vrais positifs", vp, true), cellule("FN", "faux négatifs", fn, false),
      el("span", { class: "tete-ligne" }, "non annotée"), cellule("FP", "faux positifs", fp, false), cellule("VN", "vrais négatifs", vn, true));

    const precision = vp / (vp + fp), rappel = vp / (vp + fn);
    const f1 = (2 * precision * rappel) / (precision + rappel);
    const formule = (nom, calcul, valeur, cases) => {
      const n = el("div", { class: "formule-mesure" },
        el("div", { class: "nom" }, el("span", {}, nom), el("b", {}, fr(valeur, 3))),
        el("div", { class: "calcul" }, calcul),
        el("div", { class: "piste" }, el("span", { class: "plein", style: { width: `${(100 * valeur).toFixed(1)}%` } })));
      n.addEventListener("mouseenter", () => this.allumer(cases));
      n.addEventListener("mouseleave", () => this.allumer([]));
      return n;
    };
    $("mesures-formules").replaceChildren(
      formule("Précision", `VP / (VP + FP) = ${entier(vp)} / ${entier(vp + fp)}`, precision, ["VP", "FP"]),
      formule("Rappel", `VP / (VP + FN) = ${entier(vp)} / ${entier(vp + fn)}`, rappel, ["VP", "FN"]),
      formule("F1", `2 × ${fr(precision, 3)} × ${fr(rappel, 3)} / (${fr(precision, 3)} + ${fr(rappel, 3)})`, f1, ["VP", "FP", "FN"]));
  },

  allumer(cases) {
    $("mesures-matrice").querySelectorAll(".cellule").forEach((n) => n.classList.toggle("allumee", cases.includes(n.dataset.code)));
  },

  tableau(r) {
    const ligne = (entete, valeurs, attributs = {}) => el("tr", attributs,
      el("th", { scope: "row" }, entete), ...valeurs.map((v) => el("td", {}, typeof v === "number" && v < 1.0001 ? fr(v, 3) : v)));
    $("mesures-table").replaceChildren(
      el("thead", {}, el("tr", {}, el("th", {}, `Graine ${r.graine}`), el("th", {}, "Précision"), el("th", {}, "Rappel"),
        el("th", {}, "F1"), el("th", {}, "Positifs"))),
      el("tbody", {},
        ...CLASSES.map((c, k) => {
          const m = r.par_classe[c];
          return ligne(pastilleClasse(c), [m.precision, m.recall, m.f1, entier(m.support)],
            { class: k === this.classe ? "choisie" : null, onclick: () => { this.classe = k; this.maj(); } });
        }),
        ligne("Macro", [r.precision_macro, r.recall_macro, r.f1_macro, ""], { class: "globale" })));
  },

  jouer() {
    rejouerCss($("mesures"));
    if (mouvementReduit()) return;
    const nombres = [...$("mesures-matrice").querySelectorAll("b[data-n]")];
    animer(900, (e) => nombres.forEach((n) => { n.textContent = entier(Math.round(Number(n.dataset.n) * adoucir(e))); }));
  },
};

/* ============================================================
   6.1 La perte
   ============================================================ */

const Perte = {
  y: 1,
  init() {
    const boutons = $("perte-y").querySelectorAll("button");
    boutons.forEach((b) => b.addEventListener("click", () => {
      this.y = Number(b.dataset.y);
      boutons.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      this.dessiner();
    }));
    $("perte-p").addEventListener("input", () => this.dessiner());
  },
  maj() {
    this.dessiner();
    const g = graineCourante(), e = F.analyse.source.etiquettes;
    if (!e) { $("perte-ecg").textContent = "Cet ECG n'a pas d'annotation : sa perte ne peut pas être calculée."; return; }
    const vraieClasse = e.PATHOLOGICAL ? 1 : 0;
    const perte = -Math.log(Math.max(g.probabilites[vraieClasse], 1e-7));
    $("perte-ecg").textContent = `Pour cet ECG et la graine ${g.graine}, l'entropie croisée vaut ${fr(perte, 3)}.`
      + ` Classe annotée : ${CLASSES[vraieClasse]} ; probabilité attribuée : ${fr(g.probabilites[vraieClasse], 2)}.`;
  },
  dessiner() {
    const P = palette();
    const p = Number($("perte-p").value);
    const { ctx, l, h } = contexte($("perte-canvas"), 262);
    const x0 = 40, x1 = l - 16, y0 = 28, y1 = h - 34;
    const yMax = 4.6;
    const X = (v) => interpole(x0, x1, v);
    const Y = (v) => interpole(y1, y0, Math.min(v, yMax) / yMax);
    for (let v = 0; v <= 4; v++) {
      ligne(ctx, x0, Y(v), x1, Y(v), P.bord, 1);
      ecrire(ctx, String(v), x0 - 6, Y(v) + 0.5, { taille: 10.5, aligne: "right", base: "middle", couleur: P.texte3 });
    }
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => ecrire(ctx, fr(v, v % 0.5 ? 2 : v === 0.5 ? 1 : 0), X(v), y1 + 16, { taille: 10.5, aligne: "center", couleur: P.texte3 }));
    ecrire(ctx, "probabilité pathologique p", x1, y1 + 30, { taille: 10.5, aligne: "right", couleur: P.texte3 });
    ecrire(ctx, "perte", 8, 6, { taille: 10.5, couleur: P.texte3, base: "top" });
    const courbe = (f, couleur, actif) => {
      ctx.beginPath();
      for (let i = 1; i < 400; i++) { const v = i / 400; i > 1 ? ctx.lineTo(X(v), Y(f(v))) : ctx.moveTo(X(v), Y(f(v))); }
      ctx.strokeStyle = actif ? couleur : alpha(couleur, 0.25);
      ctx.lineWidth = actif ? 2.6 : 1.5;
      ctx.stroke();
    };
    courbe((v) => -Math.log(v), P.accent, this.y === 1);
    courbe((v) => -Math.log(1 - v), P.orange, this.y === 0);
    ecrire(ctx, "si pathologique : −ln p", X(0.1), Y(3.1), { taille: 11, graisse: 650, couleur: P.accent });
    ecrire(ctx, "si normale : −ln(1 − p)", X(0.9), Y(3.1), { taille: 11, graisse: 650, couleur: P.orange, aligne: "right" });
    const perte = this.y === 1 ? -Math.log(p) : -Math.log(1 - p);
    const couleur = this.y === 1 ? P.accent : P.orange;
    ligne(ctx, X(p), y1, X(p), Y(perte), couleur, 1, [3, 3]);
    ligne(ctx, x0, Y(perte), X(p), Y(perte), couleur, 1, [3, 3]);
    disque(ctx, X(p), Y(perte), 6, couleur, P.surface, 2);
    const juste = (this.y === 1 && p >= 0.5) || (this.y === 0 && p < 0.5);
    etiquette(ctx, `p = ${fr(p, 2)} : perte = ${fr(perte, 3)} ${juste ? "(juste)" : "(fausse)"}`, limite(X(p), x0 + 120, x1 - 120), Math.max(y0 + 12, Y(perte) - 20), couleur);
  },
};

/* ============================================================
   6.2 La descente de gradient (un seul poids, illustration)
   ============================================================ */

const Gradient = {
  w0: -1.5,
  w: -1.5,
  eta: 0.6,
  pas: 0,
  traces: [],
  minuterie: 0,
  annuler: null,
  wAffiche: -1.5,

  L(w) { return 0.5 * (w - 1) ** 2 + 0.3; },
  pente(w) { return w - 1; },

  init() {
    const boutons = $("gradient-eta").querySelectorAll("button");
    boutons.forEach((b) => b.addEventListener("click", () => {
      this.eta = Number(b.dataset.eta);
      boutons.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      this.zero();
    }));
    $("gradient-pas").addEventListener("click", () => { this.stop(); this.unPas(); });
    $("gradient-lecture").addEventListener("click", () => this.lecture());
    $("gradient-zero").addEventListener("click", () => this.zero());
    this.zero();
  },
  maj() { this.dessiner(); },

  zero() {
    this.stop();
    this.w = this.wAffiche = this.w0;
    this.pas = 0;
    this.traces = [this.w0];
    this.formule();
    this.dessiner();
  },
  stop() { clearInterval(this.minuterie); this.minuterie = 0; $("gradient-lecture").textContent = "Lecture"; },
  lecture() {
    if (this.minuterie) { this.stop(); return; }
    $("gradient-lecture").textContent = "Pause";
    this.minuterie = setInterval(() => {
      if (Math.abs(this.w - 1) < 0.01 || Math.abs(this.w - 1) > 4.6 || this.pas >= 25) { this.stop(); return; }
      this.unPas();
    }, 700);
  },
  unPas() {
    if (Math.abs(this.w - 1) > 4.6) return;
    const avant = this.w;
    this.dernierePente = this.pente(avant);
    this.w = avant - this.eta * this.dernierePente;
    this.pas += 1;
    this.traces.push(this.w);
    this.formule(avant);
    if (this.annuler) this.annuler();
    if (mouvementReduit()) { this.wAffiche = this.w; this.dessiner(); return; }
    this.annuler = animer(450, (e) => {
      this.wAffiche = interpole(avant, this.w, adoucir(e));
      this.saut = Math.sin(Math.PI * e) * Math.min(1, Math.abs(this.w - avant) / 2);
      this.dessiner();
    });
  },
  formule(avant) {
    const f = $("gradient-formule");
    if (!this.pas) { f.textContent = `départ : w = ${fr(this.w0, 2)} ; perte = ${fr(this.L(this.w0), 3)} ; η = ${fr(this.eta, 2)}`; return; }
    const diverge = Math.abs(this.w - 1) > Math.abs(avant - 1);
    f.textContent = `pas ${this.pas} : pente = ${fr(this.dernierePente, 2)} ; w ← ${fr(avant, 2)} − ${fr(this.eta, 2)} × (${fr(this.dernierePente, 2)}) = ${fr(this.w, 2)} ; perte = ${fr(this.L(this.w), 3)}${diverge ? "  (s'éloigne du minimum)" : ""}`;
  },

  dessiner() {
    const P = palette();
    const { ctx, l, h } = contexte($("gradient-canvas"), 250);
    const wMin = -4, wMax = 6, x0 = 34, x1 = l - 16, y0 = 26, y1 = h - 34;
    const Lmax = this.L(wMin);
    const X = (w) => interpole(x0, x1, (limite(w, wMin, wMax) - wMin) / (wMax - wMin));
    const Y = (v) => interpole(y1, y0, Math.min(v, Lmax) / Lmax);
    ligne(ctx, x0, y1, x1, y1, P.texte3, 1);
    ecrire(ctx, "poids w", x1, y1 + 26, { taille: 10.5, aligne: "right", couleur: P.texte3 });
    ecrire(ctx, "perte", 8, 6, { taille: 10.5, couleur: P.texte3, base: "top" });
    ecrire(ctx, "minimum", X(1), y1 + 14, { taille: 10.5, aligne: "center", couleur: P.vert, graisse: 600 });
    ligne(ctx, X(1), Y(this.L(1)), X(1), y1, alpha(P.vert, 0.6), 1, [3, 3]);
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) { const w = wMin + ((wMax - wMin) * i) / 200; i ? ctx.lineTo(X(w), Y(this.L(w))) : ctx.moveTo(X(w), Y(this.L(w))); }
    ctx.strokeStyle = P.texte2;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Trajectoire des pas précédents.
    for (let k = 1; k < this.traces.length - 1; k++) {
      ligne(ctx, X(this.traces[k - 1]), Y(this.L(this.traces[k - 1])), X(this.traces[k]), Y(this.L(this.traces[k])), alpha(P.orange, 0.45), 1, [3, 3]);
      disque(ctx, X(this.traces[k]), Y(this.L(this.traces[k])), 3, alpha(P.orange, 0.5));
    }
    const w = this.wAffiche;
    const y = Y(this.L(w)) - (this.saut || 0) * 40;
    // Tangente : la pente indique la direction de descente.
    const d = 0.9, s = this.pente(w);
    const Yt = (v) => interpole(y1, y0, v / Lmax);
    ligne(ctx, X(w - d), Yt(this.L(w) - s * d), X(w + d), Yt(this.L(w) + s * d), alpha(P.accent, 0.8), 1.5);
    disque(ctx, X(w), y, 8, P.orange, P.surface, 2);
    if (Math.abs(this.w - 1) > 4.6) etiquette(ctx, "divergence : le pas est trop grand", l / 2, y0 + 12, P.rouge);
  },
};

/* ============================================================
   6.3 Les courbes d'apprentissage réelles
   ============================================================ */

const Courbes = {
  vu: false,
  init() {
    this.lecteur = lecteur(3200, (t) => this.dessiner(t));
    $("courbes-rejouer").addEventListener("click", () => this.lecteur.jouer());
    quandVisible($("courbes-canvas"), () => { this.vu = true; this.lecteur.jouer(); }, 0.4);
  },
  maj() {
    const m = F.modele.graines[F.graine];
    const courbes = F.courbes.seeds[String(m.graine)]?.fold_average;
    if (!Array.isArray(courbes) || courbes.length === 0) {
      throw new Error(`Courbes d'apprentissage absentes pour la graine ${m.graine}.`);
    }
    const n = Math.max(...courbes.map((record) => record.fold_count));
    $("courbes-texte").textContent =
      `Graine ${m.graine} : moyenne des courbes sur ${n} plis internes. `
      + "Les plis s'arrêtent à des époques différentes ; les points tardifs "
      + "sont calculés sur les plis qui poursuivent encore l'apprentissage.";
    if (this.vu) this.lecteur.jouer(); else this.dessiner(Infinity);
  },

  dessiner(t) {
    const P = palette();
    const m = F.modele.graines[F.graine];
    const H = F.courbes.seeds[String(m.graine)].fold_average;
    const canvas = $("courbes-canvas");
    const empile = canvas.clientWidth < 560;
    const { ctx, l, h } = contexte(canvas, empile ? 460 : 280);
    const n = H.length;
    const visibles = t === Infinity ? n : Math.max(1, Math.ceil(prog(t, 0, 2600) * n));
    const panneaux = empile
      ? [{ x0: 48, x1: l - 14, y0: 26, y1: h / 2 - 34 }, { x0: 48, x1: l - 14, y0: h / 2 + 18, y1: h - 30 }]
      : [{ x0: 48, x1: l / 2 - 16, y0: 26, y1: h - 34 }, { x0: l / 2 + 40, x1: l - 14, y0: 26, y1: h - 34 }];
    // Axes fixes et communs aux graines : un axe resserré sur les données
    // grossirait l'écart entre train et validation. Repères : ln 2 ≈ 0,69 est
    // la perte d'un modèle qui répond au hasard, 50 % son exactitude.
    const series = [
      { titre: "Perte (entropie croisée)", echelle: 1, bornes: [0, 0.8], decimales: 1, courbes: [["train_loss_mean", P.accent, "entraînement"], ["val_loss_mean", P.orange, "validation"]] },
      { titre: "Exactitude (%)", echelle: 100, bornes: [50, 100], decimales: 0, courbes: [["train_accuracy_mean", P.accent, "entraînement"], ["val_accuracy_mean", P.orange, "validation"]] },
    ];
    series.forEach((serie, s) => {
      const z = panneaux[s];
      const toutes = serie.courbes.flatMap(([cle]) => H.map((point) => point[cle] * serie.echelle));
      const min = Math.min(serie.bornes[0], ...toutes), max = Math.max(serie.bornes[1], ...toutes);
      const X = (i) => interpole(z.x0, z.x1, i / Math.max(1, n - 1));
      const Y = (v) => interpole(z.y1, z.y0, (v - min) / (max - min));
      const { valeurs, decimales } = graduations(min, max, 4);
      valeurs.forEach((v) => {
        ligne(ctx, z.x0, Y(v), z.x1, Y(v), P.bord, 1);
        ecrire(ctx, fr(v, Math.max(decimales, serie.decimales)), z.x0 - 6, Y(v) + 0.5, { taille: 10, aligne: "right", base: "middle", couleur: P.texte3 });
      });
      const maxEpoch = H[n - 1].epoch;
      for (let e = 10; e <= maxEpoch; e += 10) {
        const index = H.findIndex((point) => point.epoch >= e);
        if (index >= 0) ecrire(ctx, String(e), X(index), z.y1 + 14, { taille: 10, aligne: "center", couleur: P.texte3 });
      }
      ecrire(ctx, "époque", z.x1, z.y1 + 27, { taille: 10, aligne: "right", couleur: P.texte3 });
      ecrire(ctx, serie.titre, z.x0, z.y0 - 10, { taille: 11.5, graisse: 650 });

      serie.courbes.forEach(([cle, couleur]) => {
        ctx.beginPath();
        for (let i = 0; i < visibles; i++) {
          const x = X(i), y = Y(H[i][cle] * serie.echelle);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.strokeStyle = couleur;
        ctx.lineWidth = 1.8;
        ctx.stroke();
      });
      if (s === 0) {
        ctx.font = `400 10.5px ${P.police}`;
        const largeurs = serie.courbes.map(([, , nom]) => 21 + ctx.measureText(nom).width + 14);
        let lx = z.x1 - largeurs.reduce((a, b) => a + b, 0);
        serie.courbes.forEach(([, couleur, nom], i) => {
          ligne(ctx, lx, z.y0 + 6, lx + 16, z.y0 + 6, couleur, 2.5);
          ecrire(ctx, nom, lx + 21, z.y0 + 6.5, { taille: 10.5, base: "middle", couleur: P.texte2 });
          lx += largeurs[i];
        });
      }
    });
  },
};

/* ============================================================
   6.4 La courbe ROC (résultats enregistrés du test)
   ============================================================ */

const Roc = {
  vu: false,
  init() {
    this.lecteur = lecteur(2200, (t) => this.dessiner(t));
    $("roc-rejouer").addEventListener("click", () => this.lecteur.jouer());
    quandVisible($("roc-canvas"), () => { this.vu = true; this.lecteur.jouer(); }, 0.4);
  },
  courbe() {
    const m = F.modele.graines[F.graine];
    const courbe = F.roc.seeds[String(m.graine)];
    if (!courbe) throw new Error(`Courbe ROC absente pour la graine ${m.graine}.`);
    return { graine: m.graine, ...courbe };
  },
  maj() {
    const c = this.courbe(), op = c.operating_point;
    $("roc-texte").textContent =
      `Graine ${c.graine} : AUROC ${fr(c.auroc, 3)} sur les ${entier(F.roc.n_ecg)} ECG du test. `
      + `Au seuil ${fr(c.threshold, 2)} choisi sur la validation (point orange), le modèle détecte `
      + `${fr(100 * op.tpr, 1)} % des ECG pathologiques et signale à tort ${fr(100 * op.fpr, 1)} % des ECG normaux.`;
    if (this.vu) this.lecteur.jouer(); else this.dessiner(Infinity);
  },

  dessiner(t) {
    const P = palette();
    const c = this.courbe();
    const canvas = $("roc-canvas");
    const { ctx, l, h } = contexte(canvas, 340);
    // Repère carré : les deux axes sont des taux de 0 à 1.
    const cote = Math.max(120, Math.min(l - 76, h - 64));
    const x0 = Math.max(56, (l - cote) / 2 + 18), y0 = 14;
    const z = { x0, y0, x1: x0 + cote, y1: y0 + cote };
    const X = (v) => interpole(z.x0, z.x1, v), Y = (v) => interpole(z.y1, z.y0, v);

    [0, 0.2, 0.4, 0.6, 0.8, 1].forEach((v) => {
      ligne(ctx, X(v), z.y0, X(v), z.y1, P.bord, 1);
      ligne(ctx, z.x0, Y(v), z.x1, Y(v), P.bord, 1);
      ecrire(ctx, fr(v, 1), X(v), z.y1 + 14, { taille: 10, aligne: "center", couleur: P.texte3 });
      ecrire(ctx, fr(v, 1), z.x0 - 6, Y(v) + 0.5, { taille: 10, aligne: "right", base: "middle", couleur: P.texte3 });
    });
    ecrire(ctx, "taux de faux positifs (1 − spécificité)", (z.x0 + z.x1) / 2, z.y1 + 30, { taille: 10.5, aligne: "center", couleur: P.texte2 });
    ctx.save();
    ctx.translate(z.x0 - 38, (z.y0 + z.y1) / 2);
    ctx.rotate(-Math.PI / 2);
    ecrire(ctx, "taux de vrais positifs (sensibilité)", 0, 0, { taille: 10.5, aligne: "center", base: "middle", couleur: P.texte2 });
    ctx.restore();

    // Diagonale : un modèle qui répond au hasard.
    ligne(ctx, X(0), Y(0), X(1), Y(1), P.texte3, 1.2, [5, 4]);
    ecrire(ctx, "hasard (AUROC 0,5)", X(0.6), Y(0.52), { taille: 10, couleur: P.texte3 });

    // Courbe tracée progressivement, avec l'aire qu'elle délimite.
    const n = c.fpr.length;
    const k = t === Infinity ? n : Math.max(2, Math.ceil(prog(t, 0, 1600) * n));
    ctx.beginPath();
    ctx.moveTo(X(c.fpr[0]), Y(0));
    for (let i = 0; i < k; i++) ctx.lineTo(X(c.fpr[i]), Y(c.tpr[i]));
    ctx.lineTo(X(c.fpr[k - 1]), Y(0));
    ctx.closePath();
    // Aire translucide : la grille et la diagonale restent visibles dessous.
    ctx.globalAlpha = 0.14;
    ctx.fillStyle = P.accent;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    for (let i = 0; i < k; i++) i ? ctx.lineTo(X(c.fpr[i]), Y(c.tpr[i])) : ctx.moveTo(X(c.fpr[i]), Y(c.tpr[i]));
    ctx.strokeStyle = P.accent;
    ctx.lineWidth = 2;
    ctx.stroke();

    if (k < n) return;
    ecrire(ctx, `AUROC = ${fr(c.auroc, 3)}`, X(0.62), Y(0.24), { taille: 13, graisse: 650, aligne: "center" });
    ecrire(ctx, "aire sous la courbe", X(0.62), Y(0.24) + 16, { taille: 10.5, aligne: "center", couleur: P.texte2 });
    const op = c.operating_point;
    disque(ctx, X(op.fpr), Y(op.tpr), 5.5, P.orange, P.surface, 2);
    etiquette(ctx, `seuil ${fr(c.threshold, 2)}`, X(op.fpr) + 10, Y(op.tpr) + 18, P.orange, { aligne: "left", taille: 10.5 });
  },
};

/* ============================================================
   Orchestration
   ============================================================ */

const CHAPITRES = [Fil, Vue, Neurone, Stats, Norm, Grille, Reseau, Sigmoide, Comparaison, Seuils, Mesures, Perte, Gradient, Courbes, Roc];

function toutMettreAJour() {
  recalculer();
  CHAPITRES.forEach((c) => c.maj());
}

function brancherSommaire() {
  const liens = [...document.querySelectorAll(".sommaire a")];
  const obs = new IntersectionObserver((entrees) => {
    entrees.forEach((e) => {
      if (!e.isIntersecting) return;
      liens.forEach((a) => a.classList.toggle("actif", a.getAttribute("href") === `#${e.target.id}`));
    });
  }, { rootMargin: "-35% 0px -60% 0px" });
  document.querySelectorAll(".chapitre").forEach((s) => obs.observe(s));
}

function redessiner() {
  PAL = null;
  Grille.maj();
  Norm.dessiner();
  if (Reseau.dropout) Reseau.dessinerDropout(); else if (!Reseau.lecteur.enCours) Reseau.dessiner(Infinity);
  if (!Sigmoide.lecteur.enCours) Sigmoide.dessiner(Infinity);
  Perte.dessiner();
  Gradient.dessiner();
  if (!Courbes.lecteur.enCours) Courbes.dessiner(Infinity);
  if (!Roc.lecteur.enCours) Roc.dessiner(Infinity);
}

async function demarrer() {
  try {
    [F.modele, F.distributions, F.courbes, F.roc] = await Promise.all([
      api("/api/modele"),
      api("/api/distributions"),
      api("/api/courbes-apprentissage"),
      api("/api/roc"),
    ]);
    verifierModeleBinaire(F.modele);
    try { F.graine = limite(Number(localStorage.getItem(CLE_GRAINE)) || 0, 0, 4); } catch (e) { F.graine = 0; }
    const demande = new URLSearchParams(location.search).get("ecg");
    const derniere = relireAnalyse();
    if (demande) F.analyse = await api(`/api/ecg/${demande}`);
    else if (derniere && derniere.graines) F.analyse = derniere;
    else {
      F.catalogue = await api("/api/catalogue");
      if (!F.catalogue.donnees_brutes) throw new Error("Aucun ECG à suivre : analysez d'abord un fichier dans la fenêtre 1.");
      F.analyse = await api(`/api/ecg/${F.catalogue.ecg[0][0]}`);
    }
    await chargerPoids(F.graine);
  } catch (e) {
    $("fil-ecg").textContent = "indisponible";
    erreurFil(`Chargement impossible : ${e.message}`);
    return;
  }
  recalculer();
  CHAPITRES.forEach((c) => c.init());
  CHAPITRES.forEach((c) => c.maj());
  brancherSommaire();
  surRedimension(redessiner);
}

demarrer();
