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
│   ├── action_distance.py distanza tra le azioni di due Controller sugli stati di un run
│   ├── choose_checkpoint.py  dal run di training alla specialista scelta, in un comando
│   ├── upload_to_hf.py    carica su Hugging Face una specialista (ultimo checkpoint o uno scelto)
│   └── convert.py         ONNX ↔ PyTorch, ONNX di un checkpoint di training
│
├── tests/test_sanity.py   tre controlli, in un file solo
├── notebooks/             solo per guardare i dati: importano steerduck
│
├── models/   data/   runs/    non versionate
└── results/                   versionata: tabelle e figure finali
```

Circa **16 file Python** in tutto. I moduli e gli script si creano quando serve la fase corrispondente, non prima.

**Stato (29 ottobre 2026).**
- Fase 1, dalla migrazione di `notebooks/fase_1.ipynb`: `configs/situations.yaml`, `policies.yaml`, `eval.yaml`; `steerduck/config.py`, `scene.py`, `rollout.py`, `policy.py` (che contiene anche gli hook che *leggono* le activation), `report.py`; `scripts/evaluate.py`, `scripts/report.py`.
- Fase 2: `configs/training.yaml` (sezione `ppo`), `steerduck/mjlab_tasks.py`, `scripts/train.py ppo`, con le famiglie `slope_down` e `slope_up` (camminata solo in discesa o solo in salita, come nella valutazione). Il fine-tuning parte dal checkpoint di training di `alpha_walking` pubblicato da Pollen (`checkpoint` in `policies.yaml`). La distanza di una specialista dalla policy ufficiale si misura con `scripts/action_distance.py` (`policy.relabel` e `report.action_distance`); un checkpoint intermedio si valuta dopo `scripts/convert.py checkpoint` (`policy.checkpoint_to_onnx`); `scripts/choose_checkpoint.py` sceglie la specialista tra i checkpoint di un training in un comando (candidati da `report.checkpoint_choice` valutazioni con `rollout.evaluate_run`, regola in `report.choose_checkpoint`).
- Test: `tests/test_sanity.py` con `test_policy_equivalence` e `test_checkpoint_actor_equivalence`.
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
Chiavi di una situazione: `terrain` con `height_cm`, `amplitude_cm`, `num_waves`, `deg`, `direction` secondo il tipo (su `slope`, `down` è la piramide percorsa dalla cima, `up` la piramide rovesciata di mjlab, con la partenza sul fondo e quattro rampe in salita intorno), e per la striscia delle transizioni `flat_before_m`, `slope_m`, `flat_after_m`. Poi `friction` (piedi **e** pavimento), il carico con `mass_pct` (% della massa del robot) e `offset` (un nome di `payload_offsets_m`, definiti in cima al file), l'evento con `event` (`friction` | `payload`), `event_t_s` e `friction_after`, e infine `episode_s` se diverso da `eval.yaml`.
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

**`policies.yaml`** — il registro dei Controller: a ogni nome corrisponde un tipo (`onnx`, `mlp`, `adapter`, `system`, `switch`, `clae`, `residual`, `student`) e i file di `models/` da cui costruirlo (`path`, o `hf: [repo, file]` da scaricare). Una policy può avere anche `checkpoint: [repo, file]`, il checkpoint di training rsl_rl completo (attore, critic, normalizzazioni, std delle azioni, ottimizzatore) da cui riparte il PPO: per `alpha_walking` è `pollen-robotics/microduck-training`, `bundles/velocity2/441tzs6d_3750/model_3750.pt` (run W&B `441tzs6d`, iterazione 3750). Una policy appena addestrata si valuta prima di promuoverla con `run: <run_id>`, che punta all'ONNX esportato dal run. Gli script ricevono solo il nome (`--controller system`) e non conoscono mai un percorso.

**`eval.yaml`** — simulatore e protocollo degli episodi, comuni a ogni valutazione: passo fisico, decimazione, motori bam, soglia di caduta, dimensioni dei terreni, comando di velocità, heading hold, tensione della batteria, rumore iniziale, warm-up, rumore gaussiano sulle azioni (`action_noise_std`, per scegliere il rumore di esplorazione del PPO), suite, numero di episodi, semi, cosa registrare, video (per un elenco di situazioni, per tutte, oppure `one_per_family`: la prima situazione di ogni blocco di `situations.yaml` e di ogni sua `direction`, per esempio `face_3deg_downhill` e `face_3deg_uphill`; `video_znear_m`, il piano di clipping vicino della telecamera in metri, perché quello di MuJoCo cresce con la scena e sulla piramide di 40 m tagliava il robot), criterio del report ("peggiore del riferimento"), soglia di successo di un episodio (`min_progress_m_s`, avanzamento medio lungo l'heading di partenza; `min_progress_sensitivity_m_s`, le soglie della prova di sensibilità). Le soglie di successo `report.py` le legge dall'`eval.yaml` corrente, non dal run: valgono anche per i run vecchi. Le opzioni da riga di comando di `evaluate.py` sovrascrivono suite, episodi, rumore sulle azioni, registrazione, video e processi.

**`training.yaml`** — un file con quattro sezioni:
- `ppo`: le specialiste della fase 2. Contiene:
  - la policy di partenza (`init_controller: alpha_walking`, da `policies.yaml`): si caricano tutti i dati del suo `checkpoint`, compreso il rumore d'azione;
  - ambienti, seme, frequenza dei checkpoint;
  - il logging su Weights & Biases (blocco `wandb`): progetto, entity, gruppo, tag (si aggiunge la famiglia), modalità `online`/`offline`, caricamento dei checkpoint. Il run W&B prende il nome della cartella del run e salva i suoi file dentro di essa; il suo config contiene tutta la sezione `ppo`;
  - il caricamento su Hugging Face (blocco `huggingface`: attivo o no, namespace, privato): a fine training la specialista va nel repo `<namespace>/<famiglia>-<AAAAMMGG-HHMM>`, con lo stesso timestamp della cartella del run e del run W&B;
  - le opzioni di ricompensa del task hostile di microduck_rl (`feet`, `track`, `progress`);
  - il PPO in `finetune` (iterazioni, learning rate, schedule, KL, entropia), su attore e critic insieme dalla prima iterazione;
  - il comando (blocco `command`), lo stesso della valutazione al posto di quello casuale di microduck_rl: solo velocità in avanti (`lin_vel_x_m_s`, intervallo ri-sorteggiato ogni 3-8 s), heading tenuto fisso nella direzione della famiglia (`heading_stiffness`, `max_yaw_rate_rad_s`), heading di partenza ± `heading_jitter_deg`; niente robot fermi né rotazioni sul posto;
  - la scala di terreni (dimensione dei patch, righe, quota di piano, `start_margin_m`). Il patch è abbastanza grande (20 m) perché un episodio di 20 s resti sulla stessa faccia della piramide;
  - la regola per scegliere i checkpoint da valutare dopo il training (blocco `checkpoint_choice`, letto da `scripts/choose_checkpoint.py` dal `training.yaml` corrente): finestra centrata con cui si mediano le curve (`window_iterations`), quanto la quota di episodi finiti in caduta può superare quella del checkpoint migliore (`max_fall_pct_above_best`, in punti percentuali), un passo registrato ogni quanti per la distanza dalla policy di partenza (`state_stride`);
  - le `families`: per ogni specialista, i terreni con le chiavi di `situations.yaml`, dove un valore `[facile, difficile]` cresce lungo le righe della scala. Su `face_slope` la chiave `direction` (`downhill` | `uphill`, uguale per tutti i terreni della famiglia) decide partenza e heading: `downhill` parte appena oltre la piattaforma in cima e cammina verso +x, `uphill` parte vicino al bordo +x e cammina verso la cima. Esistono `slope_down` e `slope_up`.

  **Aggiungere una specialista non richiede codice:** basta una famiglia.
- `distill`: parametri DAgger comuni più le quattro forme di adapter (`const`, `lora1`, `smallnet`, `sae_guided`) con i loro iperparametri, e la policy allievo.
- `sae`: layer, dimensione del dictionary, tipo (TopK/BatchTopK), sparsità.
- `recognizer`: finestra (0,5 / 1 / 2 / 4 s), input (osservazioni o osservazioni + feature SAE), situazioni da classificare, grandezze continue da stimare, soglia di incertezza sotto cui il router non applica nulla.

---

## 4. `steerduck/`: dieci moduli

| Modulo | Cosa contiene |
|---|---|
| `config.py` | Caricamento dei YAML ed espansione di `situations.yaml` in situazioni, selezione per suite e per nomi o pattern (`matches`: `face_*_downhill`), seme di una situazione, percorsi canonici di `models/`, `data/`, `runs/`, `results/`, creazione della cartella di un run, manifest `run.json`, logging. Nessun altro file scrive percorsi a mano. |
| `scene.py` | Una voce di `situations.yaml` → un modello MuJoCo: terreno (la piramide di mjlab, anche rovesciata, e i due heightfield scritti per la tesi, piramide a facce piane e striscia piano → pendenza → piano), robot, carico, attrito di piedi **e** pavimento, motori bam. Tiene lo stato della situazione durante l'episodio e applica l'evento all'istante previsto. Espone le 61 osservazioni e lo stato misurato dal simulatore. |
| `rollout.py` | Il ciclo di simulazione: esegue gli episodi di **qualsiasi** Controller, in parallelo su più processi, e restituisce una riga per episodio con tutte le metriche (errore di velocità, potenza, caduta entro 2 s, metriche dopo l'evento). Con la registrazione attiva salva anche ogni passo in `data/`; con il video attivo salva un mp4 e l'immagine della partenza. Lo usano la valutazione, la raccolta dei dataset e i cicli DAgger. `evaluate_run` è la valutazione completa che scrive un run (config, manifest, `episodes.parquet`, dati per passo), usata da `evaluate.py` e da `choose_checkpoint.py`. |
| `policy.py` | L'interfaccia Controller (`reset`, `act`, `activations`, `episode_diagnostics`), il Controller ONNX, la ricostruzione in PyTorch del grafo ONNX (MLP 61 → 512 → 256 → 128 → 14 con la normalizzazione degli input) con gli hook che leggono i tre hidden layer `h1`, `h2`, `h3`, e il registro: `resolve(nome)` (scarica la policy se serve) e `build_controller`, l'unico punto in cui i pezzi vengono assemblati. Per la fase 2 anche `resolve_checkpoint(nome)` (scarica il checkpoint di training di una policy) e il controllo che l'attore di un checkpoint dia le stesse azioni dell'ONNX (`actor_onnx_difference`). `training_checkpoints` elenca i `model_<it>.pt` di un run di training e `check_checkpoint_conversion` verifica che l'ultimo, convertito, dia le azioni dell'ONNX del run. `relabel(spec, obs)` dà le azioni di un Controller su osservazioni già registrate, senza simulare: è l'etichettatura del DAgger e la base della distanza tra due policy (solo per Controller senza memoria). `checkpoint_to_onnx` scrive l'attore di un checkpoint rsl_rl nel grafo ONNX esportato dallo stesso training (pesi, media e `std + NORMALIZER_EPS` della normalizzazione), senza ricostruire l'ambiente mjlab; `onnx_difference` confronta due ONNX su osservazioni casuali. |
| `steering.py` | I forward hook che leggono e modificano le activation; le quattro forme di adapter con **la stessa API** (`apply(h, α)`, layer di destinazione, limite di sicurezza): vettore costante, LoRA di rango 1, piccola rete, forma guidata dal SAE — con α = 0 ciascuna restituisce esattamente la policy originale. Più il router, che somma gli adapter pesati dal riconoscitore, li limita e non applica nulla quando il riconoscitore è incerto, nelle tre modalità `prob`, `oracle` e `switch`. |
| `distill.py` | Il ciclo DAgger, generico rispetto a chi impara: dataset iniziale dalla specialista, ottimizzazione dei soli parametri dell'allievo, rollout dell'allievo, etichettatura con la specialista, ripetizione fino al criterio di stop. Addestra le quattro forme di adapter e la policy allievo con memoria. |
| `interpret.py` | Il SAE (TopK/BatchTopK) e i probe lineari. Dà un nome agli elementi del dictionary confrontandoli con il ground truth, sceglie le feature rilevanti per ogni situazione, scompone un vettore costante negli elementi del dictionary, le estrae come input del riconoscitore. |
| `recognizer.py` | Le finestre temporali costruite dai passi registrati e la piccola rete con due uscite: probabilità delle situazioni e stima di pendenza e attrito. Mantiene la finestra durante l'esecuzione, così il router la usa direttamente. |
| `mjlab_tasks.py` | Il training su GPU con mjlab. Contiene:<br>• le **stesse** voci di `situations.yaml` come terreni mjlab: `LadderTerrainCfg` usa `scene.make_subterrain` della valutazione, con i valori `[facile, difficile]` interpolati lungo le righe;<br>• il task hostile di microduck_rl (ricompensa, modalità fine-tuning, promozione sulla scala) con la scala di una famiglia (`make_env_cfg`) e la sua configurazione PPO (`make_rl_cfg`);<br>• la camminata direzionale della valutazione: `face_start` sposta l'origine di spawn di una `face_slope` sulla faccia +x (in cima o al bordo, secondo `direction`), `family_heading` dà l'heading della famiglia e `set_straight_walking` sostituisce il comando casuale con velocità in avanti e heading fisso;<br>• il fine-tuning di una specialista (`train_specialist`): carica dal checkpoint di `alpha_walking` attore, critic, normalizzazioni, std delle azioni e ottimizzatore (non l'iterazione: il run conta da 0), verifica l'attore contro l'ONNX, imposta il learning rate di `finetune` e fa PPO;<br>• il caricamento della specialista su Hugging Face (`upload_to_huggingface`): ONNX, ultimo checkpoint, `config.yaml`, `run.json` e una scheda del modello; oppure, dato un run `convert` o `choose`, il checkpoint che ha convertito, nel repo `...-it<it>` (lo usa `upload_to_hf.py`).<br>I task si costruiscono direttamente, senza registrarli in mjlab, e `external/microduck_rl` non si tocca. Più avanti: eventi di attrito e carico, wrapper con la policy frozen per le baseline RL (`activation` in stile CLAE e `action` residual). |
| `report.py` | Dagli `episodes.parquet` dei run: l'avanzamento di ogni episodio lungo l'heading di partenza (`progress_m`, da `start_*`, `end_*`, `heading_deg`) e il suo esito (`success`, oppure il motivo del fallimento: `fell`, `in_place` se avanza poco e anche la velocità del corpo è bassa, `off_course` se cammina ma non nella direzione richiesta); per situazione, percentuale di successi con intervallo di Wilson (metrica principale), conteggio dei motivi, successi con le soglie della prova di sensibilità; la distanza di un Controller da quelli di un run sugli stati che hanno visitato (`action_distance`: errore quadratico medio tra le azioni registrate e quelle del Controller sulle stesse osservazioni, per controller sorgente e situazione, su tutti i passi, senza quelli entro `fall_horizon_s` da una caduta, e per motore; è l'errore da cui parte la distillazione; `recorded_steps` legge i passi registrati, con nomi o pattern come `face_*_downhill`); la scelta dei checkpoint da valutare dopo un training (`checkpoint_choice`: per ogni `model_<it>.pt` le curve di `training_curves` mediate intorno all'iterazione — reward, quota di cadute, livelli della scala, std delle azioni — e la distanza dalla policy di partenza sugli stati di un run di valutazione; sul plateau, scala completa e cadute vicine al migliore; candidati: il più vicino alla partenza sul plateau, quello a metà del plateau, l'ultimo); la scelta tra i candidati valutati (`choose_checkpoint`: cadute e successi su tutte le situazioni con gli intervalli di Wilson; equivalenti i candidati i cui intervalli si sovrappongono a quelli del candidato con meno cadute; scelto l'equivalente più vicino alla partenza); medie, minimi e massimi sui semi, vantaggio recuperato, degrado sul piano, costo di una situazione nuova; e tutte le figure della tesi con stile e nomi uniformi. |

---

## 5. `scripts/`: sette comandi

Ogni script legge una config, crea la cartella del run, chiama la libreria, salva risultati e manifest. Nessuno contiene logica che serve anche altrove.

| Comando | Cosa fa |
|---|---|
| `evaluate.py <controller> [...]` | **Valutazione e raccolta dati, per tutti i metodi.** `--suite`, `--only`, `--episodes`, `--workers`, `--action-noise`, `--no-record`, `--no-activations`, `--no-video`, `--preview` (solo l'immagine della partenza, per controllare a occhio i terreni), `--name`. Dalla fase 3 `--alpha` (prova di steering). |
| `train.py ppo\|distill\|sae\|recognizer` | **Tutto ciò che si addestra.** `ppo --situation <s>` per le specialiste, `ppo --baseline clae\|residual`; `distill --form const\|lora1\|smallnet\|sae_guided --layer h1\|h2\|h3\|all --situation <s>`, `distill --student`; `sae`; `recognizer --window 2`. Esiste per ora `ppo --situation <famiglia>`, con `--num-envs`, `--finetune-iterations`, `--device`; `--situation all` addestra tutte le famiglie di `training.yaml` una dopo l'altra, ognuna in un processo e in un run separato (una famiglia che fallisce non ferma le altre). Alla fine carica la specialista su Hugging Face (se attivo in `training.yaml`, e scrive il repo in `run.json`) e stampa la voce `run:` da aggiungere a `policies.yaml` per valutarla con `evaluate.py`. |
| `report.py` | `--run <run> [<run> ...]` (obbligatorio): la tabella dei run indicati e il criterio "peggiore del riferimento", in `results/tables/<run>.csv` (più run uniti da `+`), così la tabella di un run non sovrascrive quella di un altro. |
| `action_distance.py` | `--run <run>` (un run di valutazione con i dati per passo) e `--controller <nome>` (da `policies.yaml`, la policy che rivaluta le osservazioni), più `--source` (quali controller del run, predefiniti tutti) e `--only` (quali situazioni). Non simula: dà al Controller le osservazioni registrate in `data/<run>/` e confronta le sue azioni con quelle registrate. Legge il `config.yaml` del run (controller, situazioni, rumore sulle azioni: un run con `action_noise_std` > 0 è rifiutato) e il suo `fields.json`, non gli YAML correnti. Scrive `results/tables/action_distance__<run>__<controller>.csv` e stampa la media per gruppo. Controllo: un Controller contro i propri dati dà 0. |
| `choose_checkpoint.py` | `--run <run di training> --only <situazioni della famiglia, nomi o pattern>`, più `--episodes` e `--workers`. Tutta la scelta della specialista in un comando: (1) valuta l'ultimo checkpoint con i dati per passo (senza activation né video); (2) propone i candidati (`report.checkpoint_choice`: per ogni checkpoint le curve tensorboard del training e la distanza dalla policy di partenza, `init_controller` del run, su quegli stati; regola di `ppo.checkpoint_choice`); (3) valuta gli altri candidati sugli stessi episodi; (4) misura la distanza di ognuno dalla partenza sui propri stati; (5) applica la regola (`report.choose_checkpoint`). Scrive `runs/<AAAAMMGG-HHMM>_choose_<run>/` con `checkpoints.csv`, l'ONNX di ogni candidato (`it<it>.onnx`), quello scelto come `<run_id>.onnx` (quindi `{type: onnx, run: <run_id>}` in `policies.yaml`), `choice.csv`, `config.yaml`, `run.json`, `log.txt`, e stampa la voce per `policies.yaml`. Le valutazioni sono run `evaluate` a sé. Con `--shortlist` si ferma al passo 2; con anche `--states <run di valutazione> --source <controller>` il passo 2 usa gli stati di una valutazione già fatta e non simula nulla (circa 30 s). Le curve del training propongono soltanto: la specialista si sceglie dopo la valutazione (Notes.txt). Controllo: `model_0.pt` è a distanza circa 0. |
| `upload_to_hf.py` | `--run <run>`: carica una specialista su Hugging Face come fa `train.py` a fine training (`mjlab_tasks.upload_to_huggingface`), con le impostazioni `huggingface` salvate nel run di training (namespace `null` = l'account con cui si è fatto il login). Con un run di training carica l'ultimo checkpoint nel repo `<namespace>/<famiglia>-<AAAAMMGG-HHMM>`; con un run `convert` o `choose` carica il checkpoint che quel run ha convertito (`model_<it>.pt` e il suo ONNX) nel repo `<namespace>/<famiglia>-<AAAAMMGG-HHMM>-it<it>`, con i file di quel run in una cartella con il suo nome. Stampa la voce `hf:` per `policies.yaml`. Non scrive un run e non modifica i run che legge. |
| `convert.py` | `checkpoint --run <run di training> --iteration <it>`: l'ONNX di `model_<it>.pt`, per valutare un checkpoint intermedio come ogni altra policy. Scrive `runs/<AAAAMMGG-HHMM>_convert_<run>_it<it>/` con `<run_id>.onnx`, `config.yaml`, `run.json`, `log.txt`, e stampa la voce `run:` per `policies.yaml`. Prima controlla il metodo: l'ultimo checkpoint convertito deve dare le stesse azioni dell'ONNX del run (differenza < 10⁻⁶). Il PPO parte dal checkpoint di training di `alpha_walking`, il cui attore è verificato contro l'ONNX da `train.py` e da `test_checkpoint_actor_equivalence`. In fase 6, `--export system` produce il grafo ONNX unico con policy, riconoscitore, router e libreria. |

### I comandi, fase per fase

| Fase | Comandi |
|---|---|
| 0. Setup | `convert --import` · `evaluate alpha --suite card110` · `evaluate rough_walk_g --suite card110` · un `train ppo` breve per i tempi |
| 1. Benchmark | `evaluate --preview` · `evaluate alpha_walking [rough_walk_g]` · `report --run <run>` |
| 2. Specialiste | `evaluate alpha_walking --suite flat --action-noise <σ>` (scelta del rumore) · `train ppo --situation <famiglia>` per ogni famiglia · `evaluate specialist_<famiglia> --record` · `choose_checkpoint --run <training> --only <situazioni>` (oppure a mano: `choose_checkpoint --shortlist`, `convert checkpoint` ed `evaluate` dei candidati, `action_distance --run <run> --controller alpha_walking`) · `upload_to_hf --run <run choose>` |
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
| `test_checkpoint_actor_equivalence` | L'attore del checkpoint di training di `alpha_walking` dà le stesse azioni dell'ONNX (< 10⁻⁵). | È il punto di partenza di ogni specialista: deve essere la policy valutata nella fase 1. `train.py` ripete il controllo sul runner vero. |
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
├── model_<it>.pt      i checkpoint (se è un training): train ppo ne salva ogni save_interval iterazioni
├── <run_id>.onnx      l'ONNX dell'ultimo checkpoint
├── events.out.*       le curve di training (tensorboard), copiate anche su W&B
├── wandb/             i file locali del run W&B (da sincronizzare se offline)
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

Stessa architettura, meno file: da 38 a circa 16 moduli, da 12 YAML a 4, da 9 script a 4 (poi 7, con `action_distance.py`, `choose_checkpoint.py` e `upload_to_hf.py` nella fase 2), da 10 sottopacchetti a nessuno.

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
