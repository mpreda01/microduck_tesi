# Struttura della repository della tesi

*Documento di progetto, 19 ottobre 2026. Versione semplificata. Descrive dove sta ogni cosa per tutte le fasi del [piano di lavoro](docs/Piano_lavoro_tesi_Preda.md). Non contiene codice.*

## L'idea in tre righe

1. **Tutti i metodi della tesi sono la stessa cosa:** un oggetto con `reset()` e `act(obs) → azione`. Lo chiamiamo **Controller**.
2. **Quindi serve un solo script per valutare** (esegui N episodi con un Controller) e **uno solo per addestrare** (cambia il file di configurazione, non il codice).
3. **Le differenze tra esperimenti stanno in 4 file YAML**, non in codice copiato.

Sono Controller, allo stesso modo: la policy ufficiale `alpha`, le varianti della comunità, le specialiste, la policy con un adapter, il sistema completo (adapter + riconoscitore + router), la sostituzione tra specialiste, le baseline RL, la policy allievo.

---

## 1. L'albero

```
tesi/
├── README.md              come si installa, i comandi fase per fase
├── CLAUDE.md              note per Claude Code
├── pyproject.toml
├── docs/                  piano di lavoro, bozze del paper e della tesi
├── external/microduck_rl/ repo ufficiale a commit fissato, mai modificato
│
├── configs/
│   ├── situations.yaml    le situazioni del benchmark e le suite
│   ├── policies.yaml      i Controller: nome → tipo + file in models/
│   ├── eval.yaml          parametri del simulatore e protocollo di valutazione
│   └── training.yaml      PPO, distillazione, SAE, riconoscitore
│
├── steerduck/             la libreria: 10 moduli, nessun sottopacchetto
│   ├── config.py          legge i YAML, percorsi, crea la cartella di un run
│   ├── scene.py           terreni, scena MuJoCo, robot, condizioni ed eventi
│   ├── rollout.py         esegue gli episodi, registra i passi, calcola le metriche
│   ├── policy.py          interfaccia Controller, ONNX, MLP PyTorch, registro
│   ├── steering.py        hook, le quattro forme di adapter, router
│   ├── distill.py         ciclo DAgger (adapter e policy allievo)
│   ├── interpret.py       SAE e probe lineari
│   ├── recognizer.py      dataset a finestre e rete del riconoscitore
│   ├── mjlab_tasks.py     le stesse situazioni per il training GPU
│   └── report.py          aggregazione, tabelle, figure
│
├── scripts/
│   ├── evaluate.py        valuta un Controller (e registra i dati)
│   ├── train.py           ppo | distill | sae | recognizer
│   ├── report.py          rigenera tabelle e figure
│   └── convert.py         ONNX ↔ PyTorch
│
├── tests/test_sanity.py   tre controlli, in un file solo
├── notebooks/             solo per guardare i dati: importano steerduck
│
├── models/   data/   runs/    non versionate
└── results/                   versionata: tabelle e figure finali
```

Circa **16 file Python** in tutto. I moduli e gli script si creano quando serve la fase corrispondente, non prima.

**Stato (20 ottobre 2026).**
- Fase 1, dalla migrazione di `notebooks/fase_1.ipynb`: `configs/situations.yaml`, `policies.yaml`, `eval.yaml`; `steerduck/config.py`, `scene.py`, `rollout.py`, `policy.py` (che contiene anche gli hook che *leggono* le activation), `report.py`; `scripts/evaluate.py`, `scripts/report.py`.
- Fase 2: `configs/training.yaml` (sezione `ppo`), `steerduck/mjlab_tasks.py`, `scripts/train.py ppo`, per ora con la sola famiglia `slope_down`.
- Test: `tests/test_sanity.py` con `test_policy_equivalence` e `test_rsl_rl_actor_equivalence`.
- Mancano ancora: `steering.py`, `distill.py`, `interpret.py`, `recognizer.py`, `convert.py`, le altre sezioni di `training.yaml` e gli altri due test, che arrivano con le fasi 3-6.

---

## 2. Le cartelle

| Cartella | Contenuto | Git |
|---|---|---|
| `configs/` | Tutti i parametri degli esperimenti. Unica fonte di verità. | Sì |
| `steerduck/` | La libreria: ogni operazione scritta una volta sola. | Sì |
| `scripts/` | Punti d'ingresso sottili: leggono una config, chiamano la libreria, salvano un run. | Sì |
| `tests/` | I tre controlli senza i quali i risultati sarebbero falsi. | Sì |
| `external/microduck_rl/` | Clone o submodule a commit fissato (`cfe1c2a` o successivo). I task nuovi si registrano dal nostro pacchetto. | Solo il commit |
| `models/` | Modelli promossi, gli unici citati da `policies.yaml`. | No |
| `data/` | Passi registrati. | No |
| `runs/` | Un run = una cartella immutabile. | No (backup) |
| `results/` | Solo ciò che genera `report.py`. | Sì |
| `docs/`, `notebooks/` | Testi ed esplorazione. | Sì |

---

## 3. `configs/`: quattro file

Nessuna sottocartella e nessuna ereditarietà tra file. Ogni parametro sta scritto nel YAML, non nascosto come valore predefinito nel codice. Ogni run salva la config completa che ha usato davvero.

**`situations.yaml`** — le situazioni, scritte per famiglie da modificare a mano. Ogni blocco ha un `name` con dei segnaposto (per esempio `face_{deg}deg_{direction}`), un `group` (`reference`, `uneven`, `steps`, `slope`, `friction`, `payload`, `transition`), il `terrain` con i suoi parametri e, sotto `sweep`, le **liste di valori**. Il blocco diventa una situazione per ogni combinazione delle liste. Cambiare le pendenze, gli attriti o i carichi vuol dire cambiare una lista, per esempio `sweep: {deg: [0, 2.5, 5, 7.5, 10, 12.5, 15], direction: [downhill, uphill]}`.
Chiavi di una situazione: `terrain` con `height_cm`, `amplitude_cm`, `num_waves`, `deg`, `direction` secondo il tipo, e per la striscia delle transizioni `flat_before_m`, `slope_m`, `flat_after_m`. Poi `friction` (piedi **e** pavimento), il carico con `mass_pct` (% della massa del robot) e `offset` (un nome di `payload_offsets_m`, definiti in cima al file), l'evento con `event` (`friction` | `payload`), `event_t_s` e `friction_after`, e infine `episode_s` se diverso da `eval.yaml`.
Il seme di terreno ed episodi dipende dal **nome** della situazione, non dalla sua posizione: modificare una lista non cambia mai gli episodi delle altre situazioni.
In fondo al file ci sono le **suite**, cioè gruppi di situazioni usati insieme — un elenco di nomi oppure `{groups: [...]}`. Per ora esistono `flat`, `fixed`, `transitions` e `all` (implicita); le altre si aggiungono con le fasi che le usano:

| Suite | Contenuto | Usata in |
|---|---|---|
| `card110` | 22 condizioni × 5 direzioni delle schede di Hugging Face | Fase 0 |
| `flat` | Solo il piano | Tutte (degrado sul piano) |
| `fixed` | Condizioni fisse | Fasi 1-5 |
| `transitions` | Cambi durante la camminata | Fasi 1, 4, 5 |
| `combined` | Situazioni sovrapposte (discesa + carico) | Fasi 4, 5 |
| `library` | Le situazioni scelte a fine fase 1 | Fasi 2-5 |
| `all` | Tutto, predefinita | Fase 1 |

**`policies.yaml`** — il registro dei Controller: a ogni nome corrisponde un tipo (`onnx`, `mlp`, `adapter`, `system`, `switch`, `clae`, `residual`, `student`) e i file di `models/` da cui costruirlo (`path`, o `hf: [repo, file]` da scaricare). Una policy appena addestrata si valuta prima di promuoverla con `run: <run_id>`, che punta all'ONNX esportato dal run. Gli script ricevono solo il nome (`--controller system`) e non conoscono mai un percorso.

**`eval.yaml`** — simulatore e protocollo degli episodi, comuni a ogni valutazione: passo fisico, decimazione, motori bam, soglia di caduta, dimensioni dei terreni, comando di velocità, heading hold, tensione della batteria, rumore iniziale, warm-up, rumore gaussiano sulle azioni (`action_noise_std`, per scegliere il rumore di esplorazione del PPO), suite, numero di episodi, semi, cosa registrare, video (per un elenco di situazioni, per tutte, oppure `one_per_family`: la prima situazione di ogni blocco di `situations.yaml`), criterio del report ("peggiore del riferimento"). Le opzioni da riga di comando di `evaluate.py` sovrascrivono suite, episodi, rumore sulle azioni, registrazione, video e processi.

**`training.yaml`** — un file con quattro sezioni:
- `ppo`: le specialiste della fase 2. Contiene:
  - la policy di partenza (`init_controller: alpha_walking`, da `policies.yaml`) e il rumore d'azione iniziale `action_std`, che non è nell'ONNX e si sceglie con `evaluate.py --action-noise`;
  - ambienti, seme, logger, frequenza dei checkpoint;
  - le opzioni di ricompensa del task hostile di microduck_rl (`feet`, `track`, `progress`);
  - le due fasi: `critic_warmup` (iterazioni e learning rate del solo critic) e `finetune` (iterazioni, learning rate, schedule, KL, entropia);
  - la scala di terreni (dimensione dei patch, righe, quota di piano);
  - le `families`: per ogni specialista, i terreni con le chiavi di `situations.yaml`, dove un valore `[facile, difficile]` cresce lungo le righe della scala.

  **Aggiungere una specialista non richiede codice:** basta una famiglia.
- `distill`: parametri DAgger comuni più le quattro forme di adapter (`const`, `lora1`, `smallnet`, `sae_guided`) con i loro iperparametri, e la policy allievo.
- `sae`: layer, dimensione del dictionary, tipo (TopK/BatchTopK), sparsità.
- `recognizer`: finestra (0,5 / 1 / 2 / 4 s), input (osservazioni o osservazioni + feature SAE), situazioni da classificare, grandezze continue da stimare, soglia di incertezza sotto cui il router non applica nulla.

---

## 4. `steerduck/`: dieci moduli

| Modulo | Cosa contiene |
|---|---|
| `config.py` | Caricamento dei YAML ed espansione di `situations.yaml` in situazioni, selezione per suite, seme di una situazione, percorsi canonici di `models/`, `data/`, `runs/`, `results/`, creazione della cartella di un run, manifest `run.json`, logging. Nessun altro file scrive percorsi a mano. |
| `scene.py` | Una voce di `situations.yaml` → un modello MuJoCo: terreno (compresi i due heightfield scritti per la tesi, piramide a facce piane e striscia piano → pendenza → piano), robot, carico, attrito di piedi **e** pavimento, motori bam. Tiene lo stato della situazione durante l'episodio e applica l'evento all'istante previsto. Espone le 61 osservazioni e lo stato misurato dal simulatore. |
| `rollout.py` | Il ciclo di simulazione: esegue gli episodi di **qualsiasi** Controller, in parallelo su più processi, e restituisce una riga per episodio con tutte le metriche (errore di velocità, potenza, caduta entro 2 s, metriche dopo l'evento). Con la registrazione attiva salva anche ogni passo in `data/`; con il video attivo salva un mp4 e l'immagine della partenza. Lo usano la valutazione, la raccolta dei dataset e i cicli DAgger. |
| `policy.py` | L'interfaccia Controller (`reset`, `act`, `activations`, `episode_diagnostics`), il Controller ONNX, la ricostruzione in PyTorch del grafo ONNX (MLP 61 → 512 → 256 → 128 → 14 con la normalizzazione degli input) con gli hook che leggono i tre hidden layer `h1`, `h2`, `h3`, e il registro: `resolve(nome)` (scarica la policy se serve) e `build_controller`, l'unico punto in cui i pezzi vengono assemblati. Per la fase 2 anche la ricostruzione dell'attore di rsl_rl dall'ONNX (`onnx_to_rsl_rl_actor`: pesi, normalizzazione congelata, rumore d'azione scelto) e il controllo che dia le stesse azioni dell'ONNX (`actor_onnx_difference`). |
| `steering.py` | I forward hook che leggono e modificano le activation; le quattro forme di adapter con **la stessa API** (`apply(h, α)`, layer di destinazione, limite di sicurezza): vettore costante, LoRA di rango 1, piccola rete, forma guidata dal SAE — con α = 0 ciascuna restituisce esattamente la policy originale. Più il router, che somma gli adapter pesati dal riconoscitore, li limita e non applica nulla quando il riconoscitore è incerto, nelle tre modalità `prob`, `oracle` e `switch`. |
| `distill.py` | Il ciclo DAgger, generico rispetto a chi impara: dataset iniziale dalla specialista, ottimizzazione dei soli parametri dell'allievo, rollout dell'allievo, etichettatura con la specialista, ripetizione fino al criterio di stop. Addestra le quattro forme di adapter e la policy allievo con memoria. |
| `interpret.py` | Il SAE (TopK/BatchTopK) e i probe lineari. Dà un nome agli elementi del dictionary confrontandoli con il ground truth, sceglie le feature rilevanti per ogni situazione, scompone un vettore costante negli elementi del dictionary, le estrae come input del riconoscitore. |
| `recognizer.py` | Le finestre temporali costruite dai passi registrati e la piccola rete con due uscite: probabilità delle situazioni e stima di pendenza e attrito. Mantiene la finestra durante l'esecuzione, così il router la usa direttamente. |
| `mjlab_tasks.py` | Il training su GPU con mjlab. Contiene:<br>• le **stesse** voci di `situations.yaml` come terreni mjlab: `LadderTerrainCfg` usa `scene.make_subterrain` della valutazione, con i valori `[facile, difficile]` interpolati lungo le righe;<br>• il task hostile di microduck_rl (ricompensa, modalità fine-tuning, promozione sulla scala) con la scala di una famiglia (`make_env_cfg`) e la sua configurazione PPO (`make_rl_cfg`);<br>• il fine-tuning di una specialista (`train_specialist`): attore ricostruito dall'ONNX e verificato, poi riscaldamento del solo critic con l'attore congelato e learning rate fisso, poi PPO completo, tutto in un solo run rsl_rl.<br>I task si costruiscono direttamente, senza registrarli in mjlab, e `external/microduck_rl` non si tocca. Più avanti: eventi di attrito e carico, wrapper con la policy frozen per le baseline RL (`activation` in stile CLAE e `action` residual). |
| `report.py` | Dagli `episodes.parquet` dei run: medie, minimi e massimi sui semi, vantaggio recuperato, degrado sul piano, costo di una situazione nuova; e tutte le figure della tesi con stile e nomi uniformi. |

---

## 5. `scripts/`: quattro comandi

Ogni script legge una config, crea la cartella del run, chiama la libreria, salva risultati e manifest. Nessuno contiene logica che serve anche altrove.

| Comando | Cosa fa |
|---|---|
| `evaluate.py <controller> [...]` | **Valutazione e raccolta dati, per tutti i metodi.** `--suite`, `--only`, `--episodes`, `--workers`, `--action-noise`, `--no-record`, `--no-activations`, `--no-video`, `--preview` (solo l'immagine della partenza, per controllare a occhio i terreni), `--name`. Dalla fase 3 `--alpha` (prova di steering). |
| `train.py ppo\|distill\|sae\|recognizer` | **Tutto ciò che si addestra.** `ppo --situation <s>` per le specialiste, `ppo --baseline clae\|residual`; `distill --form const\|lora1\|smallnet\|sae_guided --layer h1\|h2\|h3\|all --situation <s>`, `distill --student`; `sae`; `recognizer --window 2`. Esiste per ora `ppo --situation <famiglia>`, con `--num-envs`, `--warmup-iterations`, `--finetune-iterations`, `--device`. Alla fine stampa la voce `run:` da aggiungere a `policies.yaml` per valutare la specialista con `evaluate.py`. |
| `report.py` | Rigenera **tutte** le tabelle e le figure di `results/`. `--phase`, `--run` (altrimenti i run di `results/runs_used.yaml`); più avanti `--all`. |
| `convert.py` | La conversione ONNX → attore rsl_rl della fase 2 non ne ha bisogno: la fa `train.py`, verificandola sul runner, e la controlla `test_rsl_rl_actor_equivalence`. In fase 6, `--export system` produce il grafo ONNX unico con policy, riconoscitore, router e libreria. |

### I comandi, fase per fase

| Fase | Comandi |
|---|---|
| 0. Setup | `convert --import` · `evaluate alpha --suite card110` · `evaluate rough_walk_g --suite card110` · un `train ppo` breve per i tempi |
| 1. Benchmark | `evaluate --preview` · `evaluate alpha_walking [rough_walk_g]` · `report --phase 1` |
| 2. Specialiste | `evaluate alpha_walking --suite flat --action-noise <σ>` (scelta del rumore) · `train ppo --situation <famiglia>` per ogni famiglia · `evaluate specialist_<famiglia> --record` |
| 3. Libreria | `train sae` · `train distill --form … --layer …` · `evaluate alpha+<adapter> --alpha 0,…,1.5` |
| 4. Sistema | `train recognizer` · `evaluate system` e `evaluate system_oracle --suite fixed,transitions,combined` |
| 5. Confronti | `train ppo --baseline clae,residual` · `train distill --student` · `evaluate` con tutti i Controller, stessa suite e stessi semi |
| 6. Robot reale | `convert --export system` |
| 7. Scrittura | `report --all` |

---

## 6. `tests/test_sanity.py`

Le funzioni di test stanno in un unico file. Si eseguono con `uv run --project microduck_rl --with-editable . --with pytest python -m pytest tests`.

| Test | Cosa verifica | Perché |
|---|---|---|
| `test_policy_equivalence` | PyTorch e ONNX danno le stesse uscite (differenza < 10⁻⁵), normalizzazione inclusa. | Senza, ogni risultato successivo è inaffidabile. |
| `test_rsl_rl_actor_equivalence` | L'attore di rsl_rl ricostruito dall'ONNX di `alpha_walking` dà le stesse azioni dell'ONNX (< 10⁻⁵). | È il punto di partenza di ogni specialista: deve essere la policy valutata nella fase 1. `train.py` ripete il controllo sul runner vero. |
| `test_zero_adapter` | Con α = 0, ogni forma di adapter e il router incerto restituiscono esattamente le azioni della policy originale. | "Nessuna correzione" deve significare davvero la policy validata. |
| `test_conditions` | Dalla stessa voce di `situations.yaml`, simulatore CPU e mjlab producono lo stesso attrito e la stessa massa. La geometria è la stessa per costruzione, perché entrambi usano `scene.make_subterrain`. | È l'unico punto in cui una situazione viene tradotta due volte. |

---

## 7. Come si salvano i risultati

### Un run

```
runs/<AAAAMMGG-HHMM>_<comando>_<nome>/
├── config.yaml        la config completa usata davvero
├── run.json           commit git, stato "dirty", semi, hardware, versioni, durata
├── log.txt
├── episodes.parquet   una riga per episodio (se c'è una valutazione)
├── model_<it>.pt      i checkpoint (se è un training): train ppo ne salva ogni save_interval iterazioni,
├── <run_id>.onnx      più init_actor.pt (l'attore di partenza) e l'ONNX dell'ultimo checkpoint
├── events.out.*       le curve di training (tensorboard, oppure W&B)
└── curves.parquet     curve dei cicli DAgger (fase 3)
```

Esempi: `runs/20261110-1430_train_ppo_slope_down/`, `runs/20261201-0915_evaluate_alpha+const_h2_slope_down/`.

### Formati

| Dato | Formato | Dove |
|---|---|---|
| Risultati per episodio | Parquet | `runs/.../episodes.parquet` |
| Passi registrati (osservazioni, activation in float16, azioni, ground truth, scenario) | npz compresso, un blocco per (controller, situazione, seme), più un `fields.json` che descrive le colonne | `data/<run_id>/<controller>/<situazione>__seed<seme>.npz` |
| Modelli | `.pt` (state_dict) + `.json` di metadati (forma, layer, situazione, limite di sicurezza, run di origine); `.onnx` dove serve | nel run, poi in `models/` se promossi |
| Tabelle finali | CSV + LaTeX | `results/tables/` |
| Figure | PDF + PNG | `results/figures/` |

### `episodes.parquet`: una riga per episodio, colonne uguali per tutti i Controller

`run_id` · `controller` · `situation`, `group` · `seed`, `episode` · `action_noise_std` · `outcome`, `duration_s`, `fall_time_s` · `vel_error_m_s`, `vel_error_1s_m_s` · `path_speed_m_s`, `yaw_rate_rad_s`, `power_w`, `max_tilt_deg`, `distance_m` · `fell_after_event`, `vel_error_after_event_m_s`, `power_after_event_w` · `start_x`, `start_y`, `end_x`, `end_y`, `heading_deg`, `vin_v` · `torch_onnx_max_diff` · `alpha`, `recog_accuracy`, `recog_delay`.

Medie, minimi, massimi e vantaggio recuperato **non** si calcolano qui ma in `report.py`: cambiare un'aggregazione non deve costare una nuova simulazione.

### Sei regole

1. **Un run non si sovrascrive mai.** Si ripete un esperimento creando un run nuovo.
2. **`results/` contiene solo ciò che genera `report.py`.** Ogni numero della tesi si rigenera con un comando.
3. **I run usati nella tesi sono elencati in `results/runs_used.yaml`,** versionato, uno per cella delle tabelle. I run falliti restano in `runs/` ma non entrano nelle tabelle.
4. **Un modello entra in `models/` solo quando è promosso.** `policies.yaml` cita solo `models/`.
5. **Niente `.txt` o `.xlsx` come risultato.** Un Excel si esporta dal Parquet per consultarlo, non è la fonte.
6. **Ogni run registra il commit git** e segnala le modifiche non salvate.

---

## 8. Convenzioni

- **Situazioni:** minuscolo con underscore, identiche in `situations.yaml`, nelle cartelle di `data/` e nella colonna `situation`.
- **Controller con adapter:** `<base>+<forma>_<layer>_<situazione>`, per esempio `alpha+lora1_h2_slippery`.
- **Layer:** sempre `h1`, `h2`, `h3` (512, 256, 128 neuroni) oppure `all`.
- **Semi:** elenco fisso in `eval.yaml`, uguale per tutti i metodi del confronto. Il terreno e gli episodi di una situazione derivano da (nome della situazione, seme): ogni Controller riceve esattamente gli stessi episodi.
- **Notebook:** solo per guardare dati e risultati. Importano `steerduck` e leggono da `runs/` o `results/`; nessun `%%writefile`. Una funzione che serve in due notebook va spostata nella libreria.

---

## 9. Decisioni aperte

1. **Due simulatori.** La valutazione resta sullo stack CPU MuJoCo + bam delle fasi 0 e 1; il PPO gira su mjlab GPU. Entrambi leggono `situations.yaml` (via `scene.py` e `mjlab_tasks.py`) e `test_conditions` controlla che coincidano. Resta da verificare che una specialista addestrata su mjlab mantenga il proprio vantaggio sullo stack CPU: è il primo controllo della fase 2.
2. **Rollout di DAgger.** Usano `rollout.py` sullo stack CPU. Se fossero troppo lenti, si aggiunge a `rollout.py` un backend mjlab senza toccare `distill.py`.
3. **npz o Zarr.** Si parte con npz a blocchi. Si passa a Zarr solo se i dataset superano qualche GB.

---

## 10. Che cosa è cambiato rispetto alla prima versione

Stessa architettura, meno file: da 38 a circa 16 moduli, da 12 YAML a 4, da 9 script a 4, da 10 sottopacchetti a nessuno.

| Prima | Ora |
|---|---|
| `src/steerduck/` con 10 sottopacchetti | `steerduck/` piatto, 10 moduli |
| `paths.py` + `io.py` | `config.py` |
| `sim/terrains.py`, `scene.py`, `robot.py`, `conditions.py` | `scene.py` |
| `sim/rollout.py`, `metrics.py`, `recording.py`, `render.py` | `rollout.py` |
| `policies/controller.py`, `mlp.py`, `registry.py` | `policy.py` |
| `steering/hooks.py`, `adapters.py`, `router.py` | `steering.py` |
| `sae/model.py`, `interpret.py`, `analysis/probes.py` | `interpret.py` |
| `recognizer/dataset.py`, `model.py` | `recognizer.py` |
| `mjlab_tasks/` (3 file) + `baselines/frozen_env.py` | `mjlab_tasks.py` |
| `baselines/student.py` | dentro `distill.py` |
| `analysis/aggregate.py`, `plots.py` | `report.py` |
| `sim.yaml` + `eval/<fase>.yaml` | `eval.yaml` |
| `ppo/` + `distill/` + `sae.yaml` + `recognizer.yaml` (10 file) | `training.yaml`, quattro sezioni |
| Un file di config per ogni specialista e per ogni forma di adapter | Un argomento da riga di comando: `--situation`, `--form` |
| `inherit: _base.yaml` tra i file di config | Niente ereditarietà: i default stanno nel codice |
| `train_ppo.py`, `distill.py`, `train_sae.py`, `train_recognizer.py` | `train.py` con quattro sottocomandi |
| `preview_situations.py` | `evaluate.py --preview` |
| `import_policy.py`, `export_robot.py` | `convert.py` |
| `make_report.py` | `report.py` |
| Tre file di test | `tests/test_sanity.py` |
| `runs/<fase>/<data>_…/` | `runs/<data>_…/` (la fase si legge dalla config) |

Quello che **non** è stato semplificato, perché è ciò che tiene insieme il lavoro: l'interfaccia Controller unica, `situations.yaml` come unica definizione delle situazioni per entrambi i simulatori, i run immutabili con il commit git, e `results/` rigenerabile con un comando.
