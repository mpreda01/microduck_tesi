# Struttura della repository della tesi

*Documento di progetto, 19 ottobre 2026. Descrive come organizzare codice, configurazioni, dati e risultati per tutte le fasi del [piano di lavoro](docs/Piano_lavoro_tesi_Preda.md). Non contiene codice: fissa cosa va scritto, dove, e con quale interfaccia.*

## Obiettivi

1. **Nessun codice ripetuto.** Ogni operazione (simulare un episodio, calcolare una metrica, distillare un adapter, addestrare con PPO) esiste in un solo punto.
2. **Uno script di valutazione unico** per tutte le policy e tutti i metodi: policy ufficiale, varianti della comunità, specialiste, policy con adapter, sistema completo, baseline.
3. **Uno script di training unico** per tutte le policy specialiste: cambia solo il file di configurazione della situazione.
4. **Risultati riproducibili e non ambigui.** Ogni numero della tesi si rigenera con un comando, a partire da run salvati e mai sovrascritti.

---

## 1. Il principio di progetto: un'unica interfaccia Controller

Tutti i metodi confrontati nella tesi si presentano allo stesso modo: un **Controller**, cioè un oggetto con due metodi.

| Metodo | Cosa fa |
|---|---|
| `reset(n_env)` | Azzera lo stato interno all'inizio di un episodio (memoria dell'allievo, finestra del riconoscitore, stato del router). |
| `act(obs) → azione` | Riceve le 61 osservazioni e restituisce le 14 posizioni target dei motori. Facoltativamente espone le activation e le grandezze interne (probabilità del riconoscitore, vettore applicato) per la registrazione. |

Sono Controller:

| Controller | Composizione |
|---|---|
| `alpha`, `rough-walk-g` | Policy ONNX o MLP PyTorch frozen |
| `specialist_<situazione>` | MLP PyTorch addestrato con PPO |
| `alpha+<adapter>@α` | MLP frozen + un adapter di una delle quattro forme, intensità fissa |
| `system` | MLP frozen + libreria di adapter + riconoscitore + router |
| `system_oracle` | Come `system`, ma con la situazione vera letta dal simulatore |
| `switch` / `switch_oracle` | Sostituzione tra specialiste guidata dal riconoscitore o dalla situazione vera |
| `clae`, `residual` | MLP frozen + controllore addestrato con PPO |
| `student` | Policy allievo con memoria |

Ne segue che:
- **la valutazione** è un solo ciclo "simula N episodi con un Controller";
- **la raccolta dei dati** è lo stesso ciclo con la registrazione attiva;
- **la distillazione** è un solo ciclo DAgger, qualunque sia la cosa che impara;
- **le differenze tra esperimenti** stanno nei file YAML di `configs/`, non in codice copiato.

---

## 2. Albero completo

```
tesi/
├── pyproject.toml
├── README.md
├── REPO_STRUCTURE.md
├── docs/
├── external/
│   └── microduck_rl/
│
├── configs/
│   ├── situations.yaml
│   ├── policies.yaml
│   ├── sim.yaml
│   ├── eval/
│   │   └── fase1.yaml
│   ├── ppo/
│   │   ├── _base.yaml
│   │   ├── specialist_<situazione>.yaml
│   │   ├── clae.yaml
│   │   └── residual.yaml
│   ├── distill/
│   │   ├── _base.yaml
│   │   ├── const.yaml
│   │   ├── lora1.yaml
│   │   ├── smallnet.yaml
│   │   ├── sae_guided.yaml
│   │   └── student.yaml
│   ├── sae.yaml
│   └── recognizer.yaml
│
├── src/steerduck/
│   ├── __init__.py
│   ├── paths.py
│   ├── io.py
│   ├── sim/
│   │   ├── __init__.py
│   │   ├── terrains.py
│   │   ├── scene.py
│   │   ├── robot.py
│   │   ├── conditions.py
│   │   ├── rollout.py
│   │   ├── metrics.py
│   │   ├── recording.py
│   │   └── render.py
│   ├── mjlab_tasks/
│   │   ├── __init__.py
│   │   ├── terrains.py
│   │   ├── events.py
│   │   └── register.py
│   ├── policies/
│   │   ├── __init__.py
│   │   ├── controller.py
│   │   ├── mlp.py
│   │   └── registry.py
│   ├── steering/
│   │   ├── __init__.py
│   │   ├── hooks.py
│   │   ├── adapters.py
│   │   ├── router.py
│   │   └── distill.py
│   ├── sae/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── interpret.py
│   ├── recognizer/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── model.py
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── frozen_env.py
│   │   └── student.py
│   └── analysis/
│       ├── __init__.py
│       ├── aggregate.py
│       ├── probes.py
│       └── plots.py
│
├── scripts/
│   ├── evaluate.py
│   ├── preview_situations.py
│   ├── train_ppo.py
│   ├── distill.py
│   ├── train_sae.py
│   ├── train_recognizer.py
│   ├── import_policy.py
│   ├── export_robot.py
│   └── make_report.py
│
├── tests/
│   ├── test_policy_equivalence.py
│   ├── test_adapters.py
│   └── test_conditions.py
│
├── notebooks/
│
├── models/        (non versionata)
├── data/          (non versionata)
├── runs/          (non versionata)
└── results/       (versionata)
```

In totale 38 file Python di libreria, script e test (gli `__init__.py` restano vuoti o quasi).

**Stato (19 ottobre 2026).** Esistono i file che servono alla fase 1, migrati da `notebooks/fase_1.ipynb`: `configs/situations.yaml`, `policies.yaml`, `sim.yaml`, `eval/fase1.yaml`; `paths.py`, `io.py`; tutta `sim/`; `policies/controller.py`, `mlp.py`, `registry.py`; `steering/hooks.py`; `analysis/aggregate.py`; `scripts/evaluate.py`, `preview_situations.py`, `make_report.py`; `tests/test_policy_equivalence.py`; `pyproject.toml`, `README.md`. Gli altri si creano quando serve la fase corrispondente, non prima.

---

## 3. Cartelle di primo livello

| Cartella | Contenuto | Versionata con git |
|---|---|---|
| `docs/` | Piano di lavoro, note, bozze del paper e della tesi. | Sì |
| `external/microduck_rl/` | Repository ufficiale, come submodule o clone, a un commit fissato (`cfe1c2a` o successivo, da annotare nel README). **Non si modifica mai**: i task e i terreni nuovi si registrano dal nostro pacchetto. | Solo il riferimento al commit |
| `configs/` | Tutti i parametri degli esperimenti in YAML. È l'unica fonte di verità. | Sì |
| `src/steerduck/` | La libreria: tutta la logica, scritta una volta sola. | Sì |
| `scripts/` | Punti d'ingresso sottili da riga di comando: leggono una config, chiamano la libreria, salvano un run. | Sì |
| `tests/` | Controlli automatici sui punti in cui un errore renderebbe falsi tutti i risultati. | Sì |
| `notebooks/` | Esplorazione e visualizzazione. Importano `steerduck`, non contengono logica né `%%writefile`. | Sì |
| `models/` | Modelli "promossi", gli unici citati da `policies.yaml`. | No (o Git LFS) |
| `data/` | Dataset di passi registrati. | No |
| `runs/` | Un run = una cartella immutabile con config, manifest, log e risultati grezzi. | No (backup su disco o cloud) |
| `results/` | Solo tabelle e figure finali, generate da `make_report.py`. | Sì |

---

## 4. `configs/`: gli esperimenti come dati

### 4.1 `situations.yaml`

Elenco unico di tutte le situazioni del benchmark: le 68 condizioni della fase 1 più quelle che si aggiungeranno. Ogni voce ha:
- `name`: identificativo usato in tutti i file e i nomi di cartella (per esempio `slope_down_10deg`, `friction_0.2`, `payload_20pct_com`, `trans_flat_down_flat`);
- `group`: il gruppo del benchmark (`reference`, `uneven`, `steps`, `slope`, `friction`, `payload`, `transition`). Il criterio della fase 1 conta i gruppi peggiori del riferimento. Dalla fase 2 si aggiungerà `family`, la situazione della libreria e l'etichetta del riconoscitore (per esempio `slope_up` e `slope_down` separati);
- `terrain`: tipo e parametri del terreno;
- `friction`, `payload`: attrito di piedi e pavimento, carico sul baricentro o spostato;
- `event`: cambio durante l'episodio (attrito che cala, carico che compare) con il suo istante;
- `episode_s`: durata dell'episodio, se diversa da quella di `sim.yaml`.

La posizione di una voce nell'elenco ne fissa il seme (terreno ed episodi). Le situazioni nuove si aggiungono **in fondo**, così gli episodi di quelle esistenti non cambiano e un run su un sottoinsieme produce gli stessi episodi del run completo.

In fondo al file si definiscono le **suite**, cioè gruppi di situazioni usati insieme:

| Suite | Contenuto | Usata in |
|---|---|---|
| `card110` | Le 22 condizioni × 5 direzioni delle schede di Hugging Face | Fase 0 (da aggiungere con la migrazione di `fase_0.ipynb`) |
| `flat` | Solo il piano | Degrado sul piano, ogni fase |
| `fixed` | Situazioni a condizione fissa | Fasi 1-5 |
| `transitions` | Episodi con cambi durante la camminata | Fasi 1, 4, 5 |
| `combined` | Situazioni sovrapposte (discesa + carico) | Fasi 4, 5 (da aggiungere) |
| `library` | Le situazioni selezionate al termine della fase 1 | Fasi 2-5 (da aggiungere) |
| `all` | Tutto, predefinita | Fase 1 |

Una suite è un elenco di nomi oppure `{groups: [...]}`, cioè tutte le situazioni di quei gruppi.

### 4.2 `policies.yaml`

Registro dei Controller: a ogni nome corrisponde un tipo (`onnx`, `mlp`, `adapter`, `system`, `switch`, `clae`, `residual`, `student`) e i file in `models/` da cui costruirlo. Gli script ricevono solo il nome (`--controller system`) e non conoscono mai un percorso.

### 4.3 `sim.yaml` ed `eval/`

- `sim.yaml`: parametri del simulatore e del protocollo degli episodi, comuni a ogni valutazione: passo fisico, decimazione, modello dei motori bam, soglia di caduta, dimensione dei patch, comando di velocità, heading hold, intervallo di tensione della batteria, rumore iniziale, warm-up, orizzonte dell'etichetta di caduta.
- `eval/<fase>.yaml`: una valutazione. Eredita `sim.yaml` e aggiunge fase, controller, suite, episodi per situazione, seme, cosa registrare, video e parametri del report (riferimento, criterio "peggiore di"). Le opzioni da riga di comando di `evaluate.py` sovrascrivono controller, suite, episodi, registrazione e processi.

### 4.4 `ppo/`

- `_base.yaml`: iperparametri PPO comuni (learning rate, numero di ambienti, iterazioni, rete, domain randomization di base), task mjlab di partenza, checkpoint iniziale (`alpha` per il fine-tuning).
- `specialist_<situazione>.yaml`: un file per situazione della libreria. Contiene **solo** il riferimento alla voce di `situations.yaml` su cui addestrare e gli eventuali override (per esempio più iterazioni, oppure training da zero invece del fine-tuning). Aggiungere una situazione nuova significa aggiungere questo file, non codice.
- `clae.yaml`, `residual.yaml`: le baseline addestrate con PPO. Dichiarano il wrapper di `baselines/frozen_env.py` e la sua modalità.

### 4.5 `distill/`

- `_base.yaml`: parametri DAgger comuni: coppie raccolte dalla specialista, coppie per ciclo, numero massimo di cicli, criterio di stop (miglioramento di cadute ed errore di velocità sotto una soglia), ottimizzatore, batch.
- `const.yaml`, `lora1.yaml`, `smallnet.yaml`, `sae_guided.yaml`: le quattro forme di adapter (sezioni 3.3 e 3.6 del piano). Specificano la forma, il layer (`h1`, `h2`, `h3`, `all`) e i suoi iperparametri (larghezza della piccola rete, numero di feature SAE).
- `student.yaml`: la policy allievo con memoria della fase 5. Usa lo stesso ciclo DAgger, con maestro "la specialista della situazione corrente".

### 4.6 `sae.yaml` e `recognizer.yaml`

- `sae.yaml`: layer, dimensione del dictionary, tipo (TopK, BatchTopK), sparsità, dataset di activation da usare.
- `recognizer.yaml`: finestra (0,5 / 1 / 2 / 4 s), input (solo osservazioni oppure osservazioni + feature SAE), situazioni da classificare, grandezze continue da stimare (pendenza, attrito), soglia di incertezza sotto cui il router non applica nulla.

### 4.7 Convenzione

Ogni file può dichiarare `inherit: _base.yaml` e sovrascrivere solo ciò che cambia. Gli script salvano nel run la **config risolta**, cioè quella completa effettivamente usata.

---

## 5. `src/steerduck/`: la libreria

### 5.1 File di base

| File | Cosa fa |
|---|---|
| `paths.py` | Percorsi canonici di `models/`, `data/`, `runs/`, `results/`. Genera i nomi dei run (sezione 8.1). Nessun altro file scrive percorsi a mano. |
| `io.py` | Lettura e scrittura di Parquet, npz/Zarr a blocchi e JSON. Scrive il manifest `run.json` (commit git, semi, hardware, versioni, tempi). Carica e risolve le config YAML con ereditarietà. |

### 5.2 `sim/`: simulatore di valutazione e raccolta

Stack CPU MuJoCo + modello dei motori bam, lo stesso delle fasi 0 e 1 e del protocollo delle schede di Hugging Face. Raccoglie il codice oggi in `notebooks/_build/eval_scenario.py` e in `fase1_bench.py`.

| File | Cosa fa |
|---|---|
| `terrains.py` | Il terreno di una situazione come sub-terrain di mjlab, compresi i due heightfield scritti per la tesi: piramide a facce piane (pendenze direzionali) e striscia con profilo (transizioni piano → pendenza → piano). Codici dei tipi di terreno per i dati per passo. |
| `scene.py` | Costruisce il modello MuJoCo di una situazione: robot, terreno generato, carico saldato al tronco, motori bam. Posa di default, ordine dei giunti, attrito di piedi **e** pavimento, altezza del terreno. |
| `robot.py` | Il robot simulato: reset, comando dei servomotori, le 61 osservazioni e lo stato misurato (velocità del tronco, contatti e altezza dei piedi, pendenza del terreno sotto il robot). |
| `conditions.py` | Legge `situations.yaml` e le suite, sceglie la partenza di un episodio e tiene lo stato della situazione durante l'episodio: attrito iniziale, evento all'istante previsto, campi dello scenario per ogni passo. |
| `rollout.py` | Il ciclo di simulazione: esegue gli episodi di **qualsiasi** Controller in una situazione, uno o più processi in parallelo, e restituisce una riga per episodio. Con la registrazione attiva salva anche ogni passo. Lo usano la valutazione, la raccolta dei dataset e, più avanti, i cicli DAgger. |
| `metrics.py` | Le metriche di un episodio, definite in un solo punto: errore di velocità (istantaneo e su finestre di 1 s), velocità, velocità di rotazione, potenza dei motori, le stesse dopo un evento; etichetta "cade entro 2 s". Le medie su episodi e semi stanno in `analysis/aggregate.py`. |
| `recording.py` | I dati per passo (sezione 4.8 del piano): elenco dei campi con tipo e significato, raccolta per colonne, salvataggio in `.npz` compresso e `fields.json`. |
| `render.py` | Rendering offscreen: video degli episodi e immagini della partenza per l'anteprima delle situazioni. |

### 5.3 `mjlab_tasks/`: le stesse situazioni per il training su GPU

Il PPO di microduck_rl gira su mjlab. Questa cartella traduce **le stesse voci** di `situations.yaml` in configurazioni mjlab, così le situazioni si definiscono una volta sola.

| File | Cosa fa |
|---|---|
| `terrains.py` | Costruisce i `TerrainGeneratorCfg` di mjlab dalle voci YAML, con `curriculum=False` e mantenendo `_soften_terrain_contacts`. |
| `events.py` | Eventi mjlab in modalità `"step"`: attrito che cala durante l'episodio (`dr.geom_friction`), carico che compare (`dr.body_mass`), piattaforma mocap che vibra (opzionale). |
| `register.py` | Registra i task `Mjlab-Spec-<situazione>` e i task delle baseline con il wrapper di `baselines/frozen_env.py`, senza toccare `external/microduck_rl`. |

### 5.4 `policies/`

| File | Cosa fa |
|---|---|
| `controller.py` | Definisce l'interfaccia Controller (sezione 1) e il Controller per i file ONNX (onnxruntime). |
| `mlp.py` | L'MLP 61 → 512 → 256 → 128 → 14 con ELU in PyTorch, con l'eventuale normalizzazione degli input. Caricamento da ONNX (copia dei pesi) e da checkpoint rsl_rl. Espone i tre hidden layer con nomi stabili (`h1`, `h2`, `h3`) per hook e adapter. |
| `registry.py` | `build_controller(nome)`: legge `policies.yaml` e assembla il Controller richiesto componendo MLP, adapter, router, riconoscitore e baseline. È l'unico punto in cui i pezzi vengono combinati. |

### 5.5 `steering/`: il cuore del metodo

| File | Cosa fa |
|---|---|
| `hooks.py` | Forward hook di PyTorch: registra le activation dei tre layer e inietta una correzione. Gestisce aggancio e rimozione. |
| `adapters.py` | Le quattro forme di correzione, con **la stessa API**: parametri da imparare, `apply(h, α)`, layer di destinazione, limite di sicurezza. 1) vettore costante α·v; 2) LoRA di rango 1 α·(w·x)·u; 3) piccola rete α·g(h); 4) forma guidata dal SAE (scala e spostamento di poche feature del dictionary). Con α = 0 ogni forma restituisce esattamente la policy originale. |
| `router.py` | Il Controller del sistema completo: a ogni passo legge le probabilità del riconoscitore, somma gli adapter pesati, limita ciascuno al proprio limite di sicurezza, non applica nulla se il riconoscitore è incerto. Modalità alternative: `oracle` (situazione vera dal simulatore) e `switch` (sostituzione tra specialiste invece della somma), così anche la baseline delle specialiste usa lo stesso codice. |
| `distill.py` | Il ciclo di distillazione con DAgger (sezione 3.2 del piano), generico rispetto a chi impara: 1) dataset iniziale dai rollout della specialista; 2) ottimizzazione dei soli parametri dell'allievo con i pesi della policy frozen; 3) rollout dell'allievo con `sim/rollout.py`, etichettatura con la specialista, aggregazione; 4) ripetizione fino al criterio di stop. Lo stesso codice addestra le quattro forme di adapter e la policy allievo della fase 5. |

### 5.6 `sae/`

| File | Cosa fa |
|---|---|
| `model.py` | Il SAE (TopK o BatchTopK), scritto a mano o come wrapper di overcomplete; gestione degli elementi del dictionary che smettono di attivarsi. |
| `interpret.py` | Dà un nome agli elementi del dictionary confrontando i loro pesi con il ground truth (contatto dei piedi, inclinazione, velocità...). Sceglie le feature rilevanti per ogni situazione (uso 1, sezione 3.6). Scompone un vettore costante negli elementi del dictionary (uso 2). Estrae le feature come input del riconoscitore (uso 3). |

### 5.7 `recognizer/`

| File | Cosa fa |
|---|---|
| `dataset.py` | Costruisce le finestre temporali (ultimi 0,5-4 s di osservazioni, eventualmente con feature SAE) dai passi registrati in `data/`, con l'etichetta della situazione vera e i valori continui. |
| `model.py` | Piccola rete con due uscite: probabilità delle situazioni (classificazione) e stima di pendenza e attrito (regressione). Mantiene la finestra interna durante l'esecuzione, così si usa dentro il router. |

### 5.8 `baselines/`

| File | Cosa fa |
|---|---|
| `frozen_env.py` | Wrapper di ambiente mjlab con dentro la policy frozen, per le baseline addestrate con PPO. Modalità `activation`: l'azione del controllore sono le intensità delle feature SAE (controllore in stile CLAE). Modalità `action`: l'azione è un residuo sommato alle azioni della policy (action residual). |
| `student.py` | La rete allievo con memoria (finestra di osservazioni in ingresso), addestrata da `steering/distill.py` con la config `distill/student.yaml`. |

La baseline "sostituzione tra specialiste" non ha un file: è la modalità `switch` di `steering/router.py`.

### 5.9 `analysis/`

| File | Cosa fa |
|---|---|
| `aggregate.py` | Dai `episodes.parquet` dei run calcola le tabelle: media, minimo e massimo sui semi; vantaggio recuperato; degrado sul piano; costo di una situazione nuova (dai tempi di `run.json`). |
| `probes.py` | Probe lineari sulle activation per le grandezze fisiche, con dati di training e di test separati per episodio. |
| `plots.py` | Tutte le figure della tesi con stile, colori e nomi dei metodi uniformi: curve della prova di steering, matrici di confusione, tracce temporali del sistema sulle transizioni, lettura delle feature. |

---

## 6. `scripts/`: i punti d'ingresso

Ogni script: 1) legge una config e gli argomenti; 2) crea la cartella del run; 3) chiama la libreria; 4) salva risultati e manifest. Nessuno script contiene logica che serve anche altrove.

| Script | Cosa fa | Argomenti principali |
|---|---|---|
| `evaluate.py` | **Unico script di valutazione e di raccolta dati** per tutte le policy e tutti i metodi. Costruisce il Controller dal registro, esegue la suite, scrive `episodes.parquet`. Registra i passi in `data/` se la config lo chiede (`record.steps`). Più avanti, con `--alpha`, eseguirà la prova di steering (sezione 3.5 del piano). | `config`, `--controller`, `--suite`, `--only`, `--episodes`, `--workers`, `--no-steps`, `--no-activations`, `--no-video` |
| `preview_situations.py` | Un'immagine per situazione con il robot alla partenza, per controllare a occhio terreni e carichi. | `config`, `--suite`, `--only` |
| `train_ppo.py` | **Unico script PPO**: tutte le specialiste e le baseline RL. Registra il task da `mjlab_tasks/`, lancia il training di microduck_rl, esporta `.pt` e `.onnx` nel run e infine chiama la valutazione sulla propria situazione e sul piano (criterio della fase 2). | `--config ppo/specialist_<situazione>.yaml` oppure `ppo/clae.yaml` |
| `distill.py` | **Unico script di distillazione**: qualunque forma di adapter, qualunque layer, qualunque situazione, e la policy allievo. Salva l'adapter e la curva DAgger. | `--config distill/<forma>.yaml`, `--situation`, `--layer` |
| `train_sae.py` | Addestra il SAE sulle activation registrate, verifica che la policy con l'activation ricostruita cammini come l'originale e salva il dictionary con i nomi degli elementi. | `--config sae.yaml` |
| `train_recognizer.py` | Addestra il riconoscitore e misura accuratezza e tempo di riconoscimento per ogni finestra e tipo di input. | `--config recognizer.yaml`, `--window` |
| `import_policy.py` | Converte una policy ONNX in PyTorch e verifica che su 10.000 input casuali la differenza massima sia sotto 10⁻⁵. Da eseguire una volta per ogni policy di partenza. | `--onnx`, `--name` |
| `export_robot.py` | Esporta in un unico grafo ONNX policy, riconoscitore, router e libreria; tra le uscite aggiunge activation e probabilità del riconoscitore per le prove sul robot reale. | `--controller system` |
| `make_report.py` | Rigenera **tutte** le tabelle e le figure di `results/` dai run indicati in un elenco (vedi sezione 8.3). | `--phase`, `--all` |

### Corrispondenza con le fasi del piano

| Fase | Comandi |
|---|---|
| 0. Setup e riproduzione | `import_policy` per alpha e rough-walk-g · `evaluate --controller alpha,rough-walk-g --suite card110` · `train_ppo` breve per misurare i tempi |
| 1. Benchmark | `preview_situations configs/eval/fase1.yaml` · `evaluate configs/eval/fase1.yaml [--controller alpha_walking,rough_walk_g]` · `make_report fase1` |
| 2. Specialiste | `train_ppo --config ppo/specialist_<s>` per ogni situazione · `evaluate --controller specialist_<s> --record` |
| 3. Libreria | `train_sae` · `distill --config distill/{const,lora1,smallnet,sae_guided} --layer {h1,h2,h3,all}` · `evaluate --controller alpha+<adapter> --alpha 0,…,1.5` |
| 4. Sistema completo | `train_recognizer` · `evaluate --controller system,system_oracle --suite fixed,transitions,combined` |
| 5. Confronti | `train_ppo --config ppo/{clae,residual}` · `distill --config distill/student` · `evaluate` con tutti i Controller sulla stessa suite e con gli stessi semi |
| 6. Robot reale | `export_robot` |
| 7. Scrittura | `make_report --all` |

---

## 7. `tests/`

| File | Cosa verifica | Perché |
|---|---|---|
| `test_policy_equivalence.py` | Le uscite PyTorch e ONNX coincidono (differenza < 10⁻⁵), normalizzazione inclusa. | Senza questa verifica ogni risultato successivo è inaffidabile (sezione 4.5 del piano). |
| `test_adapters.py` | Con α = 0 ogni forma di adapter, e il router con riconoscitore incerto, restituisce esattamente le azioni della policy originale. | Garantisce che "nessuna correzione" significhi davvero la policy validata. |
| `test_conditions.py` | Il simulatore CPU e mjlab producono, dalla stessa voce di `situations.yaml`, la stessa geometria del terreno, lo stesso attrito e la stessa massa. | È l'unico punto in cui una situazione viene tradotta due volte (sezione 10). |

---

## 8. Come salvare i risultati

### 8.1 I run

Ogni esecuzione di uno script crea una cartella nuova, che dopo non si modifica più:

```
runs/<fase>/<AAAAMMGG-HHMM>_<script>_<nome>/
├── config.yaml          config completa risolta (base + override + argomenti)
├── run.json             manifest: commit git, stato "dirty", semi, CPU/GPU, versioni
│                        (mjlab, torch, bam), durata, ore di GPU
├── log.txt              log testuale
├── episodes.parquet     risultati della valutazione (se c'è)
├── model.pt / .onnx     modello prodotto (se è un training)
├── model.json           metadati del modello
└── curves.parquet       curve di training o cicli DAgger (se è un training)
```

Esempi: `runs/fase2/20261110-1430_train_ppo_specialist_slope_down/`, `runs/fase3/20261201-0915_evaluate_alpha+const_h2_slope_down/`.

Le curve di training possono andare anche su Weights & Biases; il run locale resta comunque il riferimento.

### 8.2 Formato per tipo di dato

| Dato | Formato | Dove | Motivazione |
|---|---|---|---|
| Risultati per episodio | **Parquet** | `runs/.../episodes.parquet` | Tipizzato, compresso, filtrabile in una riga con pandas. È la base di tutte le tabelle. |
| Passi registrati (osservazioni, activation, azioni, ground truth, scenario) | **npz compresso**, un blocco per (controller, situazione), activation in float16; **Zarr** se il dataset supera qualche GB | `data/steps/<run_id>/<controller>/<situazione>.npz` | Array densi, letti in modo efficiente per SAE, probe, riconoscitore e DAgger. La cartella porta il nome del run che li ha prodotti, quindi due raccolte non si mescolano mai. Zarr permette di aggiungere blocchi e di leggerne una parte senza caricare tutto. |
| Descrizione dei campi di un dataset | JSON | `data/steps/<run_id>/fields.json` | Nomi, tipi e unità delle colonne, ordine delle osservazioni e dei giunti, codici dei terreni. |
| Dataset DAgger | npz a blocchi, uno per ciclo | Nel run di `distill.py` | Si può ricostruire cosa ha visto l'adapter a ogni ciclo. |
| Modelli | `.pt` (solo state_dict) + `.json` di metadati; per specialiste e sistema anche `.onnx` | Nel run, poi copiati in `models/` quando promossi | Il JSON dice forma, layer, situazione, limite di sicurezza e run di origine. |
| Tabelle finali | CSV + LaTeX | `results/tables/` | CSV per controllare, LaTeX da includere nella tesi. |
| Figure | PDF (vettoriale) + PNG | `results/figures/` | PDF per LaTeX, PNG per slide e note. |

### 8.3 Schema di `episodes.parquet`

Una riga per episodio, con colonne sempre uguali per tutti i Controller:

| Colonna | Contenuto |
|---|---|
| `run_id` | Nome della cartella del run |
| `controller` | Nome nel registro (`alpha_walking`, `system`, `alpha+const_h2_slope_down`...) |
| `situation`, `group`, `situation_index` | Situazione da `situations.yaml` e sua posizione nel file |
| `seed`, `episode` | Seme e indice dell'episodio |
| `outcome`, `duration_s`, `fall_time_s` | `completed`, `fell` o `nan`; durata effettiva; istante della caduta |
| `vel_error_m_s`, `vel_error_1s_m_s` | Errore di velocità istantaneo e su finestre di 1 s (m/s) |
| `path_speed_m_s`, `yaw_rate_rad_s`, `power_w`, `max_tilt_deg`, `distance_m` | Velocità in avanti, velocità di rotazione, potenza media dei motori, inclinazione massima, distanza percorsa |
| `fell_after_event`, `vel_error_after_event_m_s`, `power_after_event_w` | Solo per le situazioni con evento: caduta dopo il cambio e metriche da 1 s dopo il cambio |
| `start_x`, `start_y`, `end_x`, `end_y`, `heading_deg`, `vin_v` | Partenza, arrivo, direzione e tensione della batteria dell'episodio |
| `torch_onnx_max_diff` | Se si registrano le activation: massima differenza tra le azioni ONNX e la copia PyTorch |
| `alpha`, `recog_accuracy`, `recog_delay` | Dalle fasi 3-4: intensità applicata, accuratezza e tempo di riconoscimento |

Le medie, i minimi, i massimi e il vantaggio recuperato **non** si calcolano nello script di valutazione ma in `analysis/aggregate.py`: così cambiare un'aggregazione non richiede di ripetere le simulazioni.

### 8.4 Regole

1. **Un run non si sovrascrive mai.** Se un esperimento va ripetuto, si crea un run nuovo.
2. **`results/` contiene solo ciò che genera `make_report.py`.** Ogni numero della tesi deve potersi rigenerare dai run con un comando.
3. **L'elenco dei run usati nella tesi sta in un file versionato**, per esempio `results/runs_used.yaml`, con un run per ogni cella delle tabelle. `make_report.py` legge da lì: i run falliti o di prova restano in `runs/` ma non entrano nelle tabelle.
4. **Un modello entra in `models/` solo quando è promosso** (per esempio una specialista che supera il criterio della fase 2). `policies.yaml` cita solo `models/`; il `model.json` copiato conserva il run di origine.
5. **Niente `.txt` o `.xlsx` come risultato.** Un Excel si può esportare dal Parquet per consultarlo, ma non è la fonte.
6. **Ogni run registra il commit git.** Se il codice ha modifiche non salvate, il manifest lo segnala.

---

## 9. Convenzioni

- **Nomi delle situazioni**: minuscolo con underscore, identici in `situations.yaml`, nei nomi dei file di config, nelle cartelle di `data/` e nella colonna `situation`.
- **Nomi dei Controller con adapter**: `<base>+<forma>_<layer>_<situazione>`, per esempio `alpha+lora1_h2_slippery`.
- **Layer**: sempre `h1`, `h2`, `h3` (512, 256, 128 neuroni), oppure `all`.
- **Semi**: elenco fisso in `configs/` (per esempio 5 semi), uguale per tutti i metodi del confronto.
- **Notebook**: solo per guardare dati e risultati; importano `steerduck` e leggono da `runs/` o `results/`. Una funzione che serve in due notebook va spostata nella libreria.

---

## 10. Decisioni aperte e rischi

1. **Due simulatori.** La valutazione resta sullo stack CPU MuJoCo + bam delle fasi 0 e 1, coerente con le schede di Hugging Face. Il PPO delle specialiste e delle baseline RL gira invece su mjlab GPU. Per non duplicare le situazioni, entrambi le leggono da `situations.yaml`, tramite `sim/conditions.py` e `mjlab_tasks/`, e `test_conditions.py` controlla che coincidano. Resta da verificare che una specialista addestrata su mjlab mantenga il proprio vantaggio quando viene valutata sullo stack CPU: è il primo controllo da fare nella fase 2.
2. **Rollout di DAgger.** I cicli DAgger usano `sim/rollout.py` sullo stack CPU, lo stesso della valutazione. Se si rivelassero troppo lenti, si può aggiungere a `rollout.py` un backend mjlab, senza cambiare `distill.py`.
3. **Zarr o npz.** Si parte con npz a blocchi, già in uso in `fase_1.ipynb`. Si passa a Zarr solo se i dataset della fase 2 superano qualche GB o se la lettura parziale diventa necessaria.

---

## 11. Migrazione dallo stato attuale

| Oggi | Diventa |
|---|---|
| `notebooks/_build/eval_scenario.py` (fase 0) | Da confrontare con `src/steerduck/sim/` e ricondurre lì; suite `card110` in `situations.yaml` |
| `fase1_bench.py` scritto con `%%writefile` in `fase_1.ipynb` | **Fatto:** `src/steerduck/sim/`, `policies/`, `analysis/aggregate.py`, `scripts/evaluate.py`, `preview_situations.py`, `make_report.py`; le 68 condizioni in `configs/situations.yaml`, i parametri in `configs/sim.yaml` e `configs/eval/fase1.yaml` |
| Conversione ONNX → PyTorch in `fase_0.ipynb` | `scripts/import_policy.py` + `policies/mlp.py` + `tests/test_policy_equivalence.py` |
| `notebooks/_build/build_notebook_en.py`, `check_notebook.py` | Non più necessari se i notebook non generano codice |
| `notebooks/fase_0.ipynb`, `fase_1.ipynb` | Notebook che lanciano gli script (comandi nel README) e mostrano i risultati: tabella, video, dati per passo |
| `notebooks/valutazione_terreni_colab_mod_piramide.ipynb` | Archivio della vecchia suite mjlab GPU, da spostare in `notebooks/archive/` |
| `notebooks/cammina_piano_colab.ipynb`, `cammina_pendenza_colab.ipynb` | Da valutare: esplorazione, oppure da spostare in `archive/` |
| `results/fase1_*.csv`, `.txt`, `results_fase1.xlsx` | Rigenerati come `runs/fase1/.../episodes.parquet` e `results/tables/fase1_*.csv` |
| `microduck_rl/` nella radice | `external/microduck_rl/` a commit fissato |
