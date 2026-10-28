/* Fenêtre 1 : utilisation du classifieur binaire. */
"use strict";

const etat = {
  modele: null,
  catalogue: null,
  analyse: null,
  graine: 0,
  format: "clinique",
};

const $ = (id) => document.getElementById(id);
const CLE_GRAINE = "ecg-mlp:binaire:graine";

/* ============================================================
   Démarrage
   ============================================================ */

async function demarrer() {
  brancherOnglets();
  brancherDepot();
  brancherFormat();
  surRedimension(() => { if (etat.analyse) { dessinerTrace(); } });

  try {
    [etat.modele, etat.catalogue] = await Promise.all([api("/api/modele"), api("/api/catalogue")]);
    verifierModeleBinaire(etat.modele);
  } catch (e) {
    const message = `Chargement impossible : ${e.message}`;
    $("carte-resultat").replaceChildren(el("h2", {}, "Résultat"),
      el("p", { class: "erreur", role: "alert" }, message));
    $("fiche-modele").replaceChildren(el("h2", {}, "Le modèle"),
      el("p", { class: "erreur", role: "alert" }, message));
    return;
  }

  try { etat.graine = limite(Number(localStorage.getItem(CLE_GRAINE)) || 0, 0, 4); } catch (e) { etat.graine = 0; }
  construireGraines();
  construireFiche();
  construireFiltre();
  afficherListe();

  const demande = new URLSearchParams(location.search).get("ecg");
  const derniere = relireAnalyse();
  if (demande) analyserBase(Number(demande));
  else if (derniere && derniere.graines) afficherAnalyse(derniere);
  else if (etat.catalogue.donnees_brutes) analyserBase(etat.catalogue.ecg[0][0]);
  else afficherVide();
}

/* ============================================================
   Choix de l'ECG
   ============================================================ */

function brancherOnglets() {
  const boutons = document.querySelectorAll("[data-onglet]");
  boutons.forEach((b) => b.addEventListener("click", () => {
    boutons.forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    $("onglet-base").hidden = b.dataset.onglet !== "base";
    $("onglet-fichier").hidden = b.dataset.onglet !== "fichier";
  }));
}

function construireFiltre() {
  for (const c of CLASSES) $("filtre-classe").append(el("option", { value: c }, `${c} · ${CLASSE_INFO[c].court}`));
  $("filtre-classe").addEventListener("change", afficherListe);
  $("recherche").addEventListener("input", afficherListe);
  $("hasard").addEventListener("click", () => {
    const liste = ecgFiltres();
    if (liste.length) analyserBase(liste[Math.floor(Math.random() * liste.length)][0]);
  });
}

function ecgFiltres() {
  const classe = $("filtre-classe").value;
  const texte = $("recherche").value.trim();
  return etat.catalogue.ecg.filter(([id, classes]) =>
    (!classe || classes.includes(classe)) && (!texte || String(id).startsWith(texte)));
}

const MAX_LISTE = 250;

function afficherListe() {
  const liste = $("liste-ecg");
  if (!etat.catalogue.donnees_brutes) {
    liste.replaceChildren(el("p", { class: "plus" },
      "Signaux bruts de PTB-XL introuvables (data/raw/ptb-xl). Importez un fichier ou téléchargez la base (voir README)."));
    $("hasard").disabled = true;
    return;
  }
  const filtres = ecgFiltres();
  const courant = etat.analyse?.source?.ecg_id;
  liste.replaceChildren(...filtres.slice(0, MAX_LISTE).map(([id, classes]) =>
    el("button", { type: "button", "aria-current": String(id === courant), onclick: () => analyserBase(id) },
      el("span", { class: "id" }, `ECG ${id}`),
      ...classes.map((c) => pastilleClasse(c)))));
  if (filtres.length > MAX_LISTE) {
    liste.append(el("p", { class: "plus" }, `… ${entier(filtres.length - MAX_LISTE)} autres : précisez le numéro.`));
  }
  if (!filtres.length) liste.append(el("p", { class: "plus" }, "Aucun ECG ne correspond."));
  $("compte").textContent = `${entier(filtres.length)} ECG sur ${entier(etat.catalogue.n_ecg)}`;
  $("hasard").disabled = !filtres.length;
}

function brancherDepot() {
  const depot = $("depot");
  const entree = $("fichiers");
  entree.addEventListener("change", () => analyserFichiers([...entree.files]).finally(() => { entree.value = ""; }));
  ["dragenter", "dragover"].forEach((t) => depot.addEventListener(t, (e) => { e.preventDefault(); depot.classList.add("survol"); }));
  ["dragleave", "drop"].forEach((t) => depot.addEventListener(t, (e) => { e.preventDefault(); depot.classList.remove("survol"); }));
  depot.addEventListener("drop", (e) => analyserFichiers([...e.dataTransfer.files]));
}

function enBase64(tampon) {
  const octets = new Uint8Array(tampon);
  let binaire = "";
  for (let i = 0; i < octets.length; i += 0x8000) binaire += String.fromCharCode(...octets.subarray(i, i + 0x8000));
  return btoa(binaire);
}

async function analyserFichiers(fichiers) {
  const hea = fichiers.find((f) => f.name.toLowerCase().endsWith(".hea"));
  const dat = fichiers.find((f) => f.name.toLowerCase().endsWith(".dat"));
  if (!hea || !dat) {
    afficherErreur("Sélectionnez ensemble les deux fichiers de l'enregistrement : l'en-tête .hea et le signal .dat.");
    return;
  }
  afficherChargement(`Analyse de ${hea.name.replace(/\.hea$/i, "")}…`);
  try {
    const corps = JSON.stringify({ hea: enBase64(await hea.arrayBuffer()), dat: enBase64(await dat.arrayBuffer()) });
    afficherAnalyse(await api("/api/analyser", { method: "POST", headers: { "Content-Type": "application/json" }, body: corps }));
  } catch (e) {
    afficherErreur(e.message);
  }
}

async function analyserBase(id) {
  afficherChargement(`Analyse de l'ECG ${id}…`);
  try {
    afficherAnalyse(await api(`/api/ecg/${id}`));
  } catch (e) {
    afficherErreur(e.message);
  }
}

/* ============================================================
   Choix du modèle et fiche
   ============================================================ */

function construireGraines() {
  const groupe = $("choix-graine");
  groupe.replaceChildren(...etat.modele.graines.map((g, i) =>
    el("button", { type: "button", "aria-pressed": String(i === etat.graine), "aria-label": `Graine ${g.graine}`,
      onclick: () => choisirGraine(i) }, String(g.graine))));
}

function choisirGraine(i) {
  etat.graine = i;
  try { localStorage.setItem(CLE_GRAINE, String(i)); } catch (e) { /* stockage indisponible */ }
  $("choix-graine").querySelectorAll("button").forEach((b, k) => b.setAttribute("aria-pressed", String(k === i)));
  majFicheGraine();
  if (etat.analyse) afficherResultat();
}

function construireFiche() {
  const m = etat.modele;
  const t = m.test;
  const pc = t.par_classe, g = t.globales;
  const ligne = (entete, valeurs, classe) => el("tr", { class: classe },
    el("th", { scope: "row" }, entete), ...valeurs.map((v) => el("td", {}, fr(v, 3))));
  $("fiche-modele").replaceChildren(
    el("h2", {}, "Le modèle"),
    el("dl", {},
      el("dt", {}, "Architecture"), el("dd", {}, el("b", {}, m.architecture.tailles.join(" → "))),
      el("dt", {}, "Paramètres"), el("dd", {}, entier(m.architecture.parametres)),
      el("dt", {}, "Activations"), el("dd", {}, `ReLU, dropout ${fr(m.architecture.dropout, 1)} à l'entraînement, softmax en sortie`),
      el("dt", {}, "Données"), el("dd", {}, "train 10 692 ECG (folds 1 à 5), validation 2 158 (fold 10)")),
    el("h3", { class: "fiche-titre" }, "Performances sur le test"),
    el("p", { class: "discret fiche-sous-titre" }, `${entier(t.n_ecg)} ECG, évalué une seule fois ; moyenne des 5 graines`),
    el("table", { class: "tableau-perf" },
      el("thead", {}, el("tr", {}, el("th", {}, ""), el("th", {}, "AUROC"),
        el("th", { title: "Parmi les ECG prédits dans la classe, part qui y appartient réellement" }, "Précision"),
        el("th", { title: "Parmi les ECG de la classe, part correctement reconnue" }, "Rappel"),
        el("th", { title: "Moyenne harmonique de la précision et du rappel" }, "F1"))),
      el("tbody", {},
        ...CLASSES.map((c) => ligne(el("span", { class: "point-classe", style: { "--couleur": couleurClasse(c) } }, c),
          [pc[c].auroc.mean, pc[c].precision.mean, pc[c].recall.mean, pc[c].f1.mean])),
        ligne("Macro", [t.auroc_macro.mean, g.precision_macro.mean, g.recall_macro.mean, g.f1_macro.mean], "macro"))),
    el("p", { class: "fiche-note" }, `IC 95 % (bootstrap sur les patients) : AUROC macro ${fr(t.ic95.auroc_macro[0], 3)} à `
      + `${fr(t.ic95.auroc_macro[1], 3)} ; F1 macro ${fr(t.ic95.f1_macro[0], 3)} à ${fr(t.ic95.f1_macro[1], 3)}.`),
    el("p", { class: "fiche-note", id: "fiche-graine" }),
    el("details", { class: "definitions" },
      el("summary", {}, "Que mesurent ces nombres ?"),
        el("p", {}, el("b", {}, "Précision"), " : parmi les ECG prédits dans une classe, part de ceux qui l'ont réellement."),
        el("p", {}, el("b", {}, "Rappel"), " (sensibilité) : parmi les ECG d'une classe, part correctement reconnue."),
      el("p", {}, el("b", {}, "F1"), " : moyenne harmonique de la précision et du rappel ; il n'est élevé que si les deux le sont."),
        el("p", {}, el("b", {}, "AUROC"), " : capacité à classer les ECG d'une classe au-dessus des autres, quel que soit le seuil."),
      el("p", {}, el("b", {}, "Macro"), " : moyenne des deux classes, qui pèsent chacune autant. Le F1 macro est la moyenne des F1, "
        + "pas le F1 de la précision et du rappel macro."),
      el("p", {}, "La fenêtre 2 (chapitre 5) détaille ces calculs sur les matrices de confusion du test.")),
    el("p", { class: "discret", style: { marginTop: "10px", marginBottom: 0 } },
      "Ces mesures ne valent que pour PTB-XL et ne constituent pas une validation clinique."));
  majFicheGraine();
}

/* Résultats du test pour le modèle choisi (relus, jamais recalculés). */
function majFicheGraine() {
  const graine = etat.modele.graines[etat.graine].graine;
  const g = etat.modele.test.par_graine.find((x) => x.graine === graine);
  const noeud = $("fiche-graine");
  if (!g || !noeud) return;
  noeud.replaceChildren(el("b", {}, `Modèle de graine ${graine} seul`), ` : précision macro ${fr(g.precision_macro, 3)}, `
    + `rappel macro ${fr(g.recall_macro, 3)}, F1 macro ${fr(g.f1_macro, 3)}.`);
}

/* ============================================================
   Affichage d'une analyse
   ============================================================ */

function afficherVide() {
  $("carte-resultat").replaceChildren(el("h2", {}, el("span", { class: "etape" }, "3"), " Résultat"),
    el("p", { class: "vide" }, "Choisissez un ECG pour lancer l'analyse."));
}

/* Message placé en tête de la carte, au-dessus du résultat précédent. */
function messageResultat(noeud) {
  const carte = $("carte-resultat");
  carte.querySelectorAll(".chargement, .erreur").forEach((n) => n.remove());
  carte.querySelector("h2").after(noeud);
  carte.scrollIntoView({ block: "nearest", behavior: "smooth" });
}
const afficherChargement = (texte) => messageResultat(el("div", { class: "chargement", style: { marginBottom: "12px" } }, texte));
const afficherErreur = (message) => messageResultat(el("p", { class: "erreur", role: "alert" }, message));

function afficherAnalyse(analyse) {
  etat.analyse = analyse;
  memoriserAnalyse(analyse);
  afficherResultat();
  dessinerTrace();
  afficherChaleur();
  afficherListe();
  const s = analyse.source;
  $("lien-fonctionnement").href = s.type === "base" ? `/fonctionnement?ecg=${s.ecg_id}` : "/fonctionnement";
}

function titreSource(s) {
  return s.ecg_id ? `ECG ${s.ecg_id}` : `Fichier ${s.nom}`;
}

function badgesSource(s) {
  const badges = [el("span", { class: "pastille info sans-point" }, titreSource(s))];
  if (s.partition === "validation") badges.push(el("span", { class: "pastille sans-point" }, "Validation (fold 10)"));
  else if (s.partition === "train") badges.push(el("span", { class: "pastille attention sans-point" }, "Train : ECG vu pendant l'apprentissage, résultat optimiste"));
  else badges.push(el("span", { class: "pastille sans-point" }, "Hors des partitions du projet : pas d'annotation"));
  if (s.etiquettes) {
    const annotees = CLASSES.filter((c) => s.etiquettes[c] === 1);
    badges.push(el("span", { class: "pastille sans-point" }, "Annotation : "), ...annotees.map((c) => pastilleClasse(c)));
  }
  return badges;
}

function afficherResultat() {
  const a = etat.analyse;
  const g = a.graines[etat.graine];
  const s = a.source;
  const lignes = CLASSES.map((c, k) => ligneSortie(c, k, a, g));

  $("carte-resultat").replaceChildren(
    el("h2", {}, el("span", { class: "etape" }, "3"), " Résultat",
      el("span", { class: "sous-titre" }, `${titreSource(s)} · modèle de graine ${g.graine}`)),
    el("div", { class: "badges" }, ...badgesSource(s)),
    el("div", { class: "entete-sorties", "aria-hidden": "true" },
      el("span", {}, "Classe"), el("span", {}, "Probabilité et seuil"), el("span", {}, ""),
      el("span", {}, "Décision"), el("span", {}, "Annotation"), el("span", {}, "5 modèles")),
    el("div", { class: "sorties" }, ...lignes),
    synthese(a, g),
    el("div", { class: "actions" },
      el("a", { class: "bouton principal", href: s.type === "base" ? `/fonctionnement?ecg=${s.ecg_id}` : "/fonctionnement" },
        "Voir cet ECG traverser le réseau →")));

}

function ligneSortie(c, k, a, g) {
  const p = g.probabilites[k];
  const retenue = g.decisions[k] === 1;
  const etiquette = a.source.etiquettes?.[c];
  let annotation = el("span", { class: "discret" }, "—");
  if (etiquette !== undefined) {
    const accord = etiquette === g.decisions[k];
    annotation = el("span", { class: `pastille sans-point ${accord ? "ok" : "ko"}`,
      title: accord ? "Décision conforme à l'annotation" : "Décision différente de l'annotation" },
      `${accord ? "✓" : "✗"} ${etiquette ? "annotée" : "non annotée"}`);
  }
  const nbOui = a.graines.filter((x) => x.decisions[k] === 1).length;
  return el("div", { class: `sortie ${retenue ? "retenue" : ""}`, style: { "--couleur": couleurClasse(c) } },
    el("div", { class: "sortie-nom" }, el("b", {}, c), el("span", {}, CLASSE_INFO[c].nom)),
    el("div", { class: "jauge", role: "img", "aria-label": `Probabilité ${fr(p, 2)}, seuil ${fr(g.seuils[k], 2)}` },
      el("div", { class: "remplissage", style: { width: `${(100 * p).toFixed(1)}%` } }),
      el("div", { class: "seuil", style: { left: `${100 * g.seuils[k]}%` }, "data-texte": `seuil ${fr(g.seuils[k], 2)}` })),
    el("div", { class: "sortie-proba" }, fr(p, 2)),
    el("div", { class: "sortie-decision" },
      retenue ? el("span", { class: "pastille", style: { "--couleur": couleurClasse(c) } }, "Retenue")
        : el("span", { class: "pastille sans-point" }, "Non retenue")),
    el("div", { class: "sortie-annotation" }, annotation),
    el("div", { class: "accord", title: `Retenue par ${nbOui} modèle${nbOui > 1 ? "s" : ""} sur 5` },
      ...a.graines.map((x, i) => el("i", { class: `${x.decisions[k] ? "oui" : ""} ${i === etat.graine ? "courant" : ""}` })),
      el("span", { class: "discret", style: { marginLeft: "4px" } }, `${nbOui}/5`)));
}

function listeClasses(classes) {
  // Minuscule initiale, sauf pour un sigle (« ECG normal »).
  const nom = (c) => { const n = CLASSE_INFO[c].nom; return /^[A-Z]{2}/.test(n) ? n : n[0].toLowerCase() + n.slice(1); };
  return classes.length ? classes.map((c) => `${c} (${nom(c)})`).join(", ") : "aucune";
}

function synthese(a, g) {
  const retenues = CLASSES.filter((c, k) => g.decisions[k] === 1);
  const paragraphes = [];
  paragraphes.push(el("p", {}, `Le modèle de graine ${g.graine} retient : `, el("b", {}, listeClasses(retenues)), "."));
  if (g.regle_vide) {
    paragraphes.push(el("p", {}, "Aucune probabilité n'atteignait son seuil : la règle de secours retient la classe la plus probable."));
  }
  const e = a.source.etiquettes;
  if (e) {
    const manquees = CLASSES.filter((c, k) => e[c] === 1 && g.decisions[k] === 0);
    const ajoutees = CLASSES.filter((c, k) => e[c] === 0 && g.decisions[k] === 1);
    if (!manquees.length && !ajoutees.length) paragraphes.push(el("p", {}, "La classe prédite coïncide avec l'annotation binaire."));
    else {
      const parties = [];
      if (manquees.length) parties.push(`annotée${manquees.length > 1 ? "s" : ""} mais non retenue${manquees.length > 1 ? "s" : ""} : ${manquees.join(", ")}`);
      if (ajoutees.length) parties.push(`retenue${ajoutees.length > 1 ? "s" : ""} mais non annotée${ajoutees.length > 1 ? "s" : ""} : ${ajoutees.join(", ")}`);
      paragraphes.push(el("p", {}, `Écart avec l'annotation (${parties.join(" ; ")}).`));
    }
  }
  const desaccords = CLASSES.filter((c, k) => a.accord[k] > 0 && a.accord[k] < 5);
  paragraphes.push(el("p", {}, desaccords.length
    ? `Les 5 modèles ne s'accordent pas sur : ${desaccords.map((c) => `${c} (${a.accord[CLASSES.indexOf(c)]}/5)`).join(", ")}.`
    : "Les 5 modèles prennent la même décision binaire."));
  paragraphes.push(el("p", { class: "discret" },
    "Les probabilités sont les sorties de la softmax. Elles ne sont pas calibrées et ne doivent pas être lues comme un risque."));
  return el("div", { class: "synthese" }, ...paragraphes);
}

/* ============================================================
   Tracé ECG sur papier millimétré
   ============================================================ */

const DISPOSITION_CLINIQUE = [
  ["I", "aVR", "V1", "V4"],
  ["II", "aVL", "V2", "V5"],
  ["III", "aVF", "V3", "V6"],
];

function brancherFormat() {
  const boutons = document.querySelectorAll("[data-format]");
  boutons.forEach((b) => b.addEventListener("click", () => {
    etat.format = b.dataset.format;
    boutons.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    if (etat.analyse) dessinerTrace();
  }));
  const canvas = $("trace");
  canvas.addEventListener("mousemove", survolTrace);
  canvas.addEventListener("mouseleave", () => { infobulle.cacher(); surlignerDerivation(null); });
}

/* Panneaux : { derivation, t0, t1, x, y, l, h } en pixels. */
function panneauxTrace(l) {
  const pxMm = l / 250; // 10 s à 25 mm/s sur toute la largeur
  const d = etat.analyse.signal.derivations;
  const panneaux = [];
  if (etat.format === "clinique") {
    const hLigne = 30 * pxMm;
    DISPOSITION_CLINIQUE.forEach((ligne, r) => ligne.forEach((nom, c) => panneaux.push({
      derivation: d.indexOf(nom), t0: 2.5 * c, t1: 2.5 * (c + 1), x: (c * l) / 4, y: r * hLigne, l: l / 4, h: hLigne,
    })));
    panneaux.push({ derivation: d.indexOf("II"), t0: 0, t1: 10, x: 0, y: 3 * hLigne, l, h: hLigne, rythme: true });
    return { panneaux, h: 4 * hLigne, pxMm, gainMm: 10 };
  }
  const hLigne = 16 * pxMm;
  d.forEach((nom, r) => panneaux.push({ derivation: r, t0: 0, t1: 10, x: 0, y: r * hLigne, l, h: hLigne }));
  return { panneaux, h: 12 * hLigne, pxMm, gainMm: 5 };
}

function dessinerTrace() {
  const a = etat.analyse;
  const canvas = $("trace");
  const largeur = canvas.parentElement.clientWidth;
  if (!largeur) return; // fenêtre en cours de redimensionnement
  const { panneaux, h, pxMm, gainMm } = panneauxTrace(largeur);
  const { ctx, l } = preparerCanvas(canvas, Math.round(h));
  etat.geometrie = { panneaux, pxMm, gainMm };

  ctx.fillStyle = cssVar("--papier");
  ctx.fillRect(0, 0, l, h);
  // Quadrillage : 1 mm (0,04 s ; 0,1 mV au gain standard) et 5 mm.
  const fin = cssVar("--papier-fin"), epais = cssVar("--papier-epais");
  for (const [pas, couleur, largeurTrait] of [[1, fin, 0.5], [5, epais, 0.8]]) {
    if (pas * pxMm < 2.2) continue;
    ctx.strokeStyle = couleur;
    ctx.lineWidth = largeurTrait;
    ctx.beginPath();
    for (let x = 0; x <= l + 0.5; x += pas * pxMm) { ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, h); }
    for (let y = 0; y <= h + 0.5; y += pas * pxMm) { ctx.moveTo(0, Math.round(y) + 0.5); ctx.lineTo(l, Math.round(y) + 0.5); }
    ctx.stroke();
  }

  const fs = a.signal.fs;
  const pxMv = gainMm * pxMm;
  const trace = cssVar("--trace");
  for (const p of panneaux) {
    const valeurs = a.signal.valeurs[p.derivation];
    const i0 = Math.round(p.t0 * fs), i1 = Math.min(valeurs.length, Math.round(p.t1 * fs));
    const segment = valeurs.slice(i0, i1);
    const mediane = [...segment].sort((x, y) => x - y)[Math.floor(segment.length / 2)];
    const yCentre = p.y + p.h * 0.55;
    ctx.save();
    ctx.beginPath();
    ctx.rect(p.x, p.y, p.l, p.h);
    ctx.clip();
    ctx.strokeStyle = trace;
    ctx.lineWidth = 1.15;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = i0; i < i1; i++) {
      const x = p.x + ((i - i0) / fs) * (p.l / (p.t1 - p.t0));
      const y = yCentre - (valeurs[i] - mediane) * pxMv;
      i === i0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
    texteCanvas(ctx, a.signal.derivations[p.derivation] + (p.rythme ? " (rythme, 10 s)" : ""), p.x + 6, p.y + 15,
      { taille: 12, graisse: 650, couleur: trace });
    if (p.x > 0) {
      ctx.strokeStyle = trace;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(p.x + 0.5, yCentre - 6);
      ctx.lineTo(p.x + 0.5, yCentre + 6);
      ctx.stroke();
    }
  }

  $("trace-titre").textContent = titreSource(a.source);
  $("echelle-trace").textContent = `25 mm/s · ${gainMm} mm/mV${gainMm === 5 ? " (gain réduit de moitié)" : ""} · grand carreau = 0,2 s`;
}

function survolTrace(e) {
  if (!etat.geometrie) return;
  const r = e.currentTarget.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const p = etat.geometrie.panneaux.find((q) => x >= q.x && x < q.x + q.l && y >= q.y && y < q.y + q.h);
  if (!p) { infobulle.cacher(); surlignerDerivation(null); return; }
  const t = p.t0 + ((x - p.x) / p.l) * (p.t1 - p.t0);
  const i = limite(Math.round(t * etat.analyse.signal.fs), 0, 4999);
  const nom = etat.analyse.signal.derivations[p.derivation];
  infobulle.montrer(`<b>${nom}</b> · t = ${fr(t, 2)} s<br>amplitude : ${fr(etat.analyse.signal.valeurs[p.derivation][i], 3)} mV`, e.clientX, e.clientY);
  surlignerDerivation(p.derivation);
}

/* ============================================================
   Carte de chaleur des 120 entrées
   ============================================================ */

const STATS_MODELE = ["mean", "std", "rms", "median", "iqr", "min", "max", "skewness", "kurtosis", "line_length"];

function frAuto(x) {
  const v = Math.abs(x);
  return fr(x, v >= 100 ? 1 : v >= 10 ? 2 : 3);
}

function afficherChaleur() {
  const a = etat.analyse;
  const e = a.entrees;
  const grille = $("carte-chaleur");
  const cellules = [el("div", {})];
  for (const s of STATS_MODELE) cellules.push(el("div", { class: "tete", title: STAT_INFO[s].nom }, STAT_INFO[s].court));
  a.signal.derivations.forEach((d, r) => {
    cellules.push(el("div", { class: "derivation", "data-ligne": r }, d));
    STATS_MODELE.forEach((s, c) => {
      const k = r * STATS_MODELE.length + c;
      const z = e.normalisees[k];
      const unite = STAT_INFO[s].unite ? " " + STAT_INFO[s].unite : "";
      const detail = `<b>${d} · ${STAT_INFO[s].nom}</b><br>valeur : ${frAuto(e.brutes[k])}${unite}<br>` +
        `train : moyenne ${frAuto(e.moyennes[k])}, écart-type ${frAuto(e.ecarts_types[k])}<br>` +
        `z = (${frAuto(e.brutes[k])} − ${frAuto(e.moyennes[k])}) / ${frAuto(e.ecarts_types[k])} = <b>${fr(z, 2)}</b>`;
      cellules.push(el("div", {
        class: "case", "data-ligne": r, style: { background: couleurZ(z), color: texteSurZ(z) },
        onmousemove: (ev) => infobulle.montrer(detail, ev.clientX, ev.clientY),
        onmouseleave: () => infobulle.cacher(),
      }, el("span", {}, fr(z, 1))));
    });
  });
  grille.replaceChildren(...cellules);
}

function surlignerDerivation(r) {
  document.querySelectorAll("#carte-chaleur [data-ligne]").forEach((n) =>
    n.classList.toggle("ligne-active", r !== null && Number(n.dataset.ligne) === r && n.classList.contains("case")));
}

demarrer();
