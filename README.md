# Classification binaire d'ECG sur PTB-XL

Le projet distingue les ECG `NORMAL` des ECG `PATHOLOGICAL` à partir de
signaux 12 dérivations échantillonnés à 500 Hz. Le MLP est entraîné sur
120 caractéristiques statistiques globales, extraites directement sans
caractéristique crête-à-crête.

> Démonstration pédagogique uniquement : ces résultats ne constituent pas
> une validation clinique et le modèle ne doit pas être utilisé pour poser
> un diagnostic.

## Règle d'étiquetage

Les étiquettes source PTB-XL `NORM`, `MI`, `STTC`, `CD` et `HYP` sont
converties en deux classes exclusives :

- `PATHOLOGICAL` dès qu'au moins une des étiquettes `MI`, `STTC`, `CD` ou
  `HYP` est présente ;
- `NORMAL` si `NORM` est présente et qu'aucune de ces quatre étiquettes
  pathologiques n'est présente.

La pathologie est prioritaire : un ECG co-étiqueté `NORM` et pathologie est
classé `PATHOLOGICAL`. Les partitions préétablies sont inchangées et séparées
par patient.

| Partition | NORMAL | PATHOLOGICAL | Total |
|---|---:|---:|---:|
| Entraînement | 4 542 | 6 150 | 10 692 |
| Validation | 912 | 1 246 | 2 158 |
| Test | 890 | 1 239 | 2 129 |

## Données

Les données viennent de PTB-XL v1.0.3 (21 799 ECG de 18 869 patients),
publiée par PhysioNet sous licence CC BY 4.0 :

- page officielle : <https://physionet.org/content/ptb-xl/1.0.3/>
- archive ZIP de la base complète (1,7 Go ; 3,0 Go décompressée) :
  <https://physionet.org/content/ptb-xl/get-zip/1.0.3/>
- DOI de cette version : <https://doi.org/10.13026/kfzx-aw45>

La base n'est pas versionnée : le pipeline la lit dans `data/raw/ptb-xl/`
(voir le guide de reproduction), et le notebook peut télécharger lui-même les
seuls fichiers dont il a besoin.

## Caractéristiques et modèle

Pour chacune des 12 dérivations, le pipeline calcule moyenne, écart-type,
RMS, médiane, IQR, minimum, maximum, asymétrie, kurtosis et longueur de
ligne, soit \(12 \times 10 = 120\) valeurs. 
 Les fichiers CSV de caractéristiques ont
121 colonnes au total : `ecg_id` et les 120 valeurs numériques.

Le réseau comporte 120 entrées, des couches cachées de 128 et 64 neurones
(ReLU, dropout 0,3), puis deux sorties. Il est entraîné avec
`CrossEntropyLoss` et AdamW. Cinq graines sont entraînées ; le seuil
pathologique de chaque graine est choisi sur la validation uniquement.

## Résultats sur le test

Moyenne ± écart-type sur cinq graines ; test indépendant de 2 129 ECG :

| Métrique | Résultat |
|---|---:|
| AUROC macro | 0,9050 ± 0,0008 |
| F1 macro | 0,8061 ± 0,0089 |
| Exactitude | 0,8149 ± 0,0073 |
| Précision PATHOLOGICAL | 0,8150 ± 0,0129 |
| Rappel PATHOLOGICAL | 0,8826 ± 0,0086 |

Les intervalles bootstrap à 95 % sont 0,891--0,919 pour l'AUROC macro et
0,790--0,824 pour le F1 macro. Les artefacts détaillés sont dans
`results/binary/`.

## Notebook du projet

[`projet_ecg.ipynb`](projet_ecg.ipynb) déroule tout le pipeline de façon
autonome, des fichiers bruts PTB-XL à l'évaluation, sans importer `src/` ni
lire `data/processed/` ou `results/`. Il enchaîne les
étiquettes SCP et les partitions par patient, l'extraction des 120
caractéristiques, l'exploration, le prétraitement dans un `Pipeline`, le
réglage et l'entraînement du MLP sur cinq graines, la validation croisée
imbriquée, l'évaluation unique sur le test (bootstrap par patient,
références régression logistique et classe majoritaire), l'analyse des
erreurs, la prédiction sur des ECG des folds 7 à 9 et la comparaison avec la
base complète. Il est enregistré exécuté : tableaux et figures sont lisibles
sans rien relancer.

Par défaut, il travaille sur un sous-échantillon stratifié de 3 000 ECG
(paramètre `N_ECG`, une dizaine de minutes sur CPU) lus dans
`data/raw/ptb-xl/` ; `N_ECG = None` refait le protocole sur les 14 979 ECG
(une vingtaine de minutes) et retrouve les résultats ci-dessus à quelques
millièmes près. S'il manque des fichiers dans ce dossier (`DOSSIER_PTBXL`),
il télécharge uniquement ceux dont il a besoin depuis le miroir officiel de
PhysioNet, environ 360 Mo pour 3 000 ECG ; il n'écrit rien d'autre. Pour le
réexécuter, l'ouvrir avec le noyau de `.venv`, puis *Restart & Run All* :

```powershell
.\.venv\Scripts\jupyter-lab.exe projet_ecg.ipynb
```

## Guide de reproduction : des données brutes aux résultats

Les étapes ci-dessous partent de PTB-XL v1.0.3 et recréent les partitions,
caractéristiques, matrices d'entrée, modèles et évaluations. Exécuter les
commandes depuis la racine du dépôt, dans PowerShell.

### 1. Préparer l'environnement et PTB-XL

Environnement ciblé : Windows et Python 3.14. Les dépendances versionnées sont
dans `requirements.txt`.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.check_env
```

Télécharger PTB-XL v1.0.3 depuis PhysioNet
(<https://physionet.org/content/ptb-xl/get-zip/1.0.3/>), décompresser
l'archive et placer sous `data/raw/ptb-xl/` les fichiers
`ptbxl_database.csv`, `scp_statements.csv` et le dossier complet
`records500/` avec les enregistrements WFDB (`.hea` et `.dat`). Le pipeline lit les signaux 500 Hz référencés par les métadonnées ;
il ne télécharge pas la base lui-même.

### 2. Construire ou vérifier les partitions figées

```powershell
.\.venv\Scripts\python.exe -m src.data.build_splits
```

Le script traduit les codes diagnostiques SCP en cinq indicateurs source
(`NORM`, `MI`, `STTC`, `CD`, `HYP`), écarte les ECG sans ces annotations,
sélectionne les folds PTB-XL déjà définis dans `src/config.py` et contrôle
qu'aucun patient n'est partagé entre partitions. Les folds sont fixes :
train = 1--5, validation = 10, test = 6 ; les folds 7--9 sont exclus.
Si les fichiers de partition existent, la commande les compare aux partitions
reconstruites et s'arrête si elles diffèrent ; elle ne les réécrit pas sans
option explicite.

Sorties : `data/processed/splits/{train,validation,test}_metadata.csv` et
`split_summary.csv`.

### 3. Extraire les 120 caractéristiques par ECG

```powershell
.\.venv\Scripts\python.exe -m src.features.extract_statistical_features_full --split all
```

Pour chaque ECG, `src/features/extract_statistical_features_full.py` lit les
12 dérivations à 500 Hz, vérifie la forme et les unités du signal, puis calcule
sur chaque dérivation moyenne, écart-type, RMS, médiane, IQR, minimum,
maximum, asymétrie, kurtosis et longueur de ligne. Cela produit 120 valeurs
par ECG, sans caractéristique crête-à-crête. L'extraction est reprenable :
les ECG déjà écrits dans les CSV ne sont pas recalculés. Consulter
`data/processed/features_statistical/*_errors.csv` et ne poursuivre que si
l'extraction est complète et que la validation suivante réussit.

Sorties : `data/processed/features_statistical/{train,validation,test}_features.csv`.

### 4. Valider et assembler les matrices

```powershell
.\.venv\Scripts\python.exe -m src.features.validate_statistical_features
.\.venv\Scripts\python.exe -m src.data.build_model_input
```

Le premier script vérifie les effectifs, colonnes, identifiants, valeurs
finies, résultats d'extraction et absence de recouvrement entre partitions.
S'il échoue, corriger ou reprendre l'extraction avant de continuer.

Le second joint caractéristiques et métadonnées par `ecg_id`, vérifie leurs
dimensions et leur alignement, puis écrit les matrices compressées. `X`
contient les 120 caractéristiques ; `Y` conserve les cinq indicateurs source
et les identifiants ECG/patient sont conservés pour les contrôles de séparation.
La conversion en cible binaire est faite ensuite par le chargeur d'entraînement,
avec priorité à `PATHOLOGICAL`.

Sorties : `data/processed/model_input/{train,validation,test}.npz` et
`manifest.json`. Les contrôles globaux s'assurent également qu'il n'y a pas
d'ECG ni de patient commun entre les partitions.

### 5. Ajuster le prétraitement

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.preprocess
```

Le `StandardScaler` est ajusté uniquement sur `X_train`, puis appliqué au
train et à la validation pour vérifier la standardisation. Le test n'est pas
utilisé pour ajuster le prétraitement. Les paramètres sont écrits dans
`results/binary/preprocessing/P1.joblib` et `P1_parameters.csv`.

### 6. Entraîner puis évaluer le test une seule fois

```powershell
.\.venv\Scripts\python.exe -m src.models.train_binary
.\.venv\Scripts\python.exe -m src.models.train_binary --evaluate-test
.\.venv\Scripts\python.exe -m src.models.train_binary --export-roc
```

`configs/binary.json` fixe l'architecture 120--128--64--2, le dropout,
AdamW, les mini-lots et les graines 0 à 4. L'entraînement utilise la
validation pour l'arrêt précoce sur AUROC pathologique et pour choisir le
seuil maximisant le F1 pathologique. Le test n'intervient ni dans la sélection
de l'époque, ni dans celle du seuil ; la commande `--evaluate-test` l'évalue
séparément après l'entraînement.

Sorties : `results/binary/runs/` contient les poids, configurations,
historiques et métriques de validation de chaque graine ; `protocol.json`
enregistre le protocole et les seuils ; `test_summary.json` contient
l'évaluation finale et le bootstrap patient-level. `--export-roc` recalcule
avec les mêmes modèles les courbes ROC du test (`test_roc.json`), en vérifiant
que les AUROC coïncident avec `test_summary.json` ; l'interface les affiche
sans jamais lire les données du test.

### 7. Reproduire la validation croisée et les courbes d'apprentissage

```powershell
.\.venv\Scripts\python.exe -m src.models.cross_validate_binary
.\.venv\Scripts\python.exe -m src.models.cross_validate_binary --learning-curves
```

La validation croisée imbriquée évalue les cinq graines sur les folds 1--5.
À chaque pli externe, un fold est tenu à l'écart ; un autre des folds
d'entraînement sert à l'arrêt précoce et au choix du seuil. La standardisation
est réajustée dans chaque pli, et les patients restent séparés. Le fold 6
demeure réservé au test final.

La deuxième commande entraîne de nouveau les modèles internes pour mesurer,
à chaque époque, l'entropie croisée et l'exactitude sur train et validation.
Elle génère les données des courbes d'apprentissage affichées par l'interface.

Sorties : `results/binary/cross_validation/summary.json`, les sous-dossiers
par graine et pli, `learning_curves.json` et `learning_curves_report.csv`.



### Notes sur les sorties existantes

Les scripts d'entraînement refusent d'écraser un dossier d'exécution déjà
présent ; l'évaluation refuse également de remplacer un `test_summary.json`
existant. La génération des courbes refuse de remplacer les fichiers de
courbes déjà créés. Pour reproduire une expérience sans toucher aux artefacts
actuels, utiliser une copie propre du projet ou archiver les sorties dans une
destination séparée avant de relancer les étapes concernées.

## Interface de démonstration

```powershell
.venv\Scripts\python.exe -m interface.app
```

Ouvrir `http://127.0.0.1:8050/utilisation`. L'application peut présenter un
ECG disponible ou importer un enregistrement WFDB compatible, affiche les
deux classes et leurs probabilités, puis permet de parcourir l'architecture
réelle 120–128–64–2. La couche de sortie est représentée au centre
verticalement. La page de fonctionnement présente aussi les courbes moyennes
de perte et d'exactitude, ainsi que la courbe ROC du test et son AUROC,
sélectionnables par graine. `tools\verify_interface.py` vérifie l'accord de l'inférence
de l'application avec les artefacts du modèle sur la validation.



## Structure du dépôt

```text
projet_ecg.ipynb  notebook autonome : pipeline complet, des fichiers bruts à l'évaluation
configs/          configuration de l'entraînement binaire
data/processed/   partitions, caractéristiques, matrices et manifeste
interface/        application locale et fichiers statiques
results/binary/   prétraitement, modèles, historiques, seuils et métriques binaires
src/              extraction, préparation, modèle et entraînement
tools/            vérification de l'interface
```

Les étiquettes source PTB-XL sont converties en deux classes pendant
l'entraînement ; le manifeste des matrices décrit donc toujours ces cinq
étiquettes sources.
