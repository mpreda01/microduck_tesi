# Vettori di steering distillati da policy specialiste per la camminata del Microduck

*Guida all'implementazione e alla sperimentazione. Documento operativo, 1 ottobre 2026. Candidato: Matteo Preda.*

## Il problema in breve

Il Microduck è un piccolo robot bipede open source. A farlo camminare è una **policy**: una rete neurale che, 50 volte al secondo, legge le misure interne del robot (angoli dei motori, inclinazione del corpo) e decide la posizione di ciascun motore delle gambe. La policy distribuita con il robot è stata addestrata in simulazione solo su **pavimento piano** (sezione 2).

Fuori dal laboratorio, però, il robot incontra condizioni diverse dal pavimento piano, in due modi:
- **condizioni che restano le stesse**, per tutto il tragitto ma diverse da quelle viste nel training: per esempio una strada sempre in salita, o un pavimento sempre scivoloso;
- **condizioni che cambiano mentre il robot cammina**: si passa dal piano a una discesa, poi su un pavimento scivoloso, poi il robot raccoglie un oggetto e diventa più pesante.

In entrambi i casi la policy ufficiale cammina peggio o cade.

## Perché il problema conta, anche se esistono robot come Spot

Robot commerciali come Spot di Boston Dynamics camminano bene su terreni molto diversi. Perché allora occuparsi della camminata di un piccolo robot open source? Per tre motivi.

**I controller dei robot commerciali sono chiusi.** Anche il controller di camminata di Spot è una rete neurale addestrata in simulazione per tentativi ed errori (*reinforcement learning*). I suoi parametri, i dati e i modelli usati per addestrarla non sono pubblici, e anche il software con cui si leggono gli stati del robot è proprietario ([Mind the Phase, appendice C](https://arxiv.org/pdf/2609.06958)). Per la ricerca Boston Dynamics offre un *RL Researcher Kit*, che permette di comandare direttamente i singoli motori e quindi di eseguire sul robot una policy **propria**, addestrata in simulazione con NVIDIA Isaac Lab ([NVIDIA](https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/), [Boston Dynamics](https://bostondynamics.com/industry/academia-education/)). Il lavoro che ha presentato il kit lo descrive come la prima dimostrazione pubblica, su Spot, di una policy addestrata in questo modo con codice di training aperto ([ICRA 2025](https://arxiv.org/abs/2504.17857)). In pratica, chi vuole un comportamento diverso da quello di fabbrica non modifica il controller esistente: ne addestra e ne collauda uno nuovo da zero.

**La robustezza dei robot commerciali costa molto.** Richiede grandi quantità di simulazione, addestramento su condizioni molto variate e lunghe campagne di prove sul robot reale: risorse che un laboratorio universitario o una comunità open source non hanno.

**Nel mondo aperto si parte da una policy collaudata ma ristretta.** Per far camminare il Microduck su terreni diversi oggi ci sono due strade:
- addestrare e collaudare una nuova policy, con il rischio di peggiorare ciò che funzionava. La guida del repository ufficiale avverte che ogni nuovo addestramento richiede in genere più tentativi per correggere il modo in cui si premia il robot, e che le differenze tra simulatore e robot reale hanno richiesto settimane di debug ([AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md));
- scegliere a mano tra le policy pubblicate dalla comunità. Una variante addestrata su terreno irregolare cade meno della policy ufficiale su terreno accidentato, ma consuma il 17% di energia in più ([scheda della policy](https://huggingface.co/RemiFabre/microduck-rough-walk-g)). Bisogna quindi scegliere in anticipo tra una camminata efficiente e una robusta.

## L'idea

La policy ufficiale **resta com'è**: i suoi parametri non vengono modificati (in gergo è *frozen*). Le si affiancano tre componenti piccoli.

1. **Una correzione per ogni situazione.** Mentre calcola il movimento, la policy produce dei risultati intermedi, chiamati *activation*. Una correzione è un vettore di numeri che si somma a questi risultati intermedi e cambia il movimento finale, senza toccare la rete (*vettore di steering*, sezione 3.1). Per ogni situazione (salita, discesa, gradini bassi, pavimento scivoloso, carico) si addestra in simulazione una **policy specialista** che sa affrontarla bene. Poi si cerca la correzione che fa comportare la policy ufficiale come la specialista (sezione 3.2). La specialista serve solo per questo e non va mai sul robot.
2. **Un riconoscitore.** Una piccola rete che, dagli ultimi secondi di misure del robot, stima in quale situazione si trova: per esempio "80% discesa, 20% piano" (sezione 3.4).
3. **Un router.** Una regola che a ogni istante somma le correzioni delle situazioni riconosciute, ciascuna pesata con la probabilità stimata dal riconoscitore (sezione 3.4).

Il sistema funziona **in anello chiuso**: la correzione applicata cambia la camminata, la camminata cambia le misure del robot, le misure cambiano la correzione successiva. Quando il robot passa dal piano alla discesa la correzione per la discesa cresce gradualmente; quando torna in piano svanisce, e la policy cammina come in origine. Lo stesso meccanismo vale quando la condizione non cambia: su una strada sempre in salita il riconoscitore identifica la salita nei primi passi e il router mantiene applicata la correzione per tutto il tragitto. Il caso con condizioni che cambiano è il più impegnativo, perché richiede di accorgersi in fretta del cambio.

**Che tipo di correzione è.** Il vettore di steering non aggiunge pezzi alla rete né ne modifica i parametri: è un insieme di numeri che, mentre la policy calcola, viene sommato ai suoi risultati intermedi. Ha però lo stesso scopo di LoRA, la tecnica più usata per adattare i grandi modelli linguistici: adattare una rete già addestrata con una modifica piccola, che si può attivare, togliere o scambiare senza riaddestrare la rete. La differenza sta nel punto in cui si interviene. LoRA aggiunge alla rete due piccole matrici di parametri, che correggono i suoi pesi; lo steering lascia i pesi intatti e agisce sui risultati intermedi. Nel machine learning le modifiche di questo genere si chiamano in generale *adapter*; la sezione 3.3 confronta le diverse forme possibili.

## Cosa porta la tesi

1. **L'adattamento di una policy collaudata alla condizione in cui si trova il robot,** senza riaddestrarla: sia una condizione che resta la stessa per tutto il tragitto, sia una che cambia durante la camminata. Oggi si può solo riaddestrare la policy, oppure scegliere in anticipo e a mano quale policy usare: la prima strada è costosa e rischiosa, la seconda non segue il terreno quando cambia.
2. **Un confronto con le altre strade possibili,** sugli stessi terreni, sia con condizioni fisse sia con condizioni che cambiano. Le alternative sono tre: tenere sul robot una policy specialista per ogni situazione e passare dall'una all'altra; addestrare per tentativi ed errori un modulo che corregge la policy a ogni istante; addestrare da zero una nuova policy capace di gestire tutte le situazioni. Si misurano tre cose: quanto bene il robot cammina quando il terreno cambia, se peggiora sul piano, e quanto costa aggiungere una situazione nuova (sezione 5).
3. **Quale forma conviene dare alle correzioni.** La più semplice è un vettore fisso, che si somma sempre uguale; altre forme cambiano da un istante all'altro e sono più potenti (sezione 3.3). Il vettore fisso ha un vantaggio: si può leggere, cioè scomporre in concetti come "passo più corto" o "corpo più arretrato" (sezione 3.6). Si misura quanto si perde in prestazioni scegliendo la forma leggibile, e se una forma guidata dal SAE può essere insieme flessibile e leggibile (sezione 3.6).

**Un limite da dichiarare.** Il metodo richiede l'accesso ai parametri e ai calcoli interni della policy. Si applica quindi a qualunque policy pubblica, come quelle del Microduck o della raccolta MuJoCo Playground, ma non ai controller chiusi come quello di fabbrica di Spot.

## Come usare questo documento

| Sezione | A cosa serve | Quando leggerla |
|---|---|---|
| 1. Glossario | Definizioni, con esempi, di tutti i termini tecnici usati da qui in avanti | Prima di proseguire, e poi come riferimento |
| 2. Punto di partenza | Cosa sa fare la policy ufficiale, cosa vede, cosa non ha mai visto | Prima di tutto |
| 3. Il metodo | Steering, distillazione, adapter, riconoscitore e router, verifica, ruolo del SAE nel guidare e leggere lo steering | Per capire cosa si costruisce |
| 4. Guida all'implementazione | Codice di partenza, simulazione, terreni, condizioni dinamiche, raccolta dei dati, training | Durante lo sviluppo |
| 5. Piano sperimentale | Metriche, fasi settimana per settimana, criteri per proseguire | Per organizzare il lavoro |
| 6. Rassegna della letteratura | Lavori vicini, con sede, link, codice e differenze | Come riferimento e per la scrittura |
| 7. Possibili sviluppi | Estensioni oltre la tesi: riconoscimento con la vista, VLA, un secondo robot in simulazione | A fine lavoro, o per pianificare un seguito |

Tutte le affermazioni sul codice del Microduck sono state verificate sul repository [microduck_rl](https://github.com/pollen-robotics/microduck_rl), branch `develop`, commit `cfe1c2a` del 29 settembre 2026, su mjlab 1.3.0 e sul pacchetto dei motori `better-actuator-models`.

---

## 1. Glossario

| Termine | Significato |
|---|---|
| **Policy** | Rete neurale che controlla il robot: a ogni istante riceve ciò che il robot misura e decide l'azione successiva. |
| **Frozen** | Detto di una policy i cui pesi non vengono più modificati. |
| **Layer** | Uno degli stadi di calcolo di una rete. La policy del Microduck ha tre hidden layer, da 512, 256 e 128 neuroni. |
| **Activation** | Il vettore di numeri che un layer produce per un dato input. Per il primo hidden layer del Microduck è un vettore di 512 numeri. |
| **Feature** | Una direzione nello spazio delle activation a cui si può associare un significato. Esempio: una direzione lungo cui l'activation cresce quando il robot allunga il passo. |
| **Vettore di steering** | Un vettore con tanti numeri quanti sono i neuroni di un layer, che durante l'esecuzione si somma all'activation di quel layer, moltiplicato per un'intensità, per cambiare il comportamento della policy. Esempio: un vettore che, sommato all'activation del secondo hidden layer (256 numeri), fa allungare il passo. |
| **Steering** | L'operazione di sommare uno o più vettori di steering alle activation di una policy frozen, senza modificarne i pesi. |
| **Probe lineare** | Regressione lineare che predice una grandezza scelta in anticipo (per esempio la velocità del robot) a partire dalle activation. Si addestra con il valore vero della grandezza, che in simulazione è noto. È supervisionato: trova solo ciò che si cerca. |
| **SAE** (Sparse Autoencoder) | Piccola rete che, senza etichette, impara a scrivere ogni activation come somma di pochi vettori scelti da un elenco appreso, detto *dictionary*. Ogni vettore del dictionary tende a corrispondere a un concetto usato dalla rete, il cui significato si scopre dopo. A differenza del probe lineare, non serve sapere in anticipo cosa cercare. |
| **Dictionary** | L'elenco di vettori appreso da un SAE. Contiene di solito più vettori di quanti neuroni ha il layer, perché una rete può rappresentare più concetti che neuroni mescolandoli negli stessi neuroni (*superposition*). |
| **Anello chiuso** (closed-loop) | Modo di controllo in cui la decisione viene ricalcolata continuamente in base a ciò che il sistema misura. Esempio: un termostato regola il riscaldamento secondo la temperatura, a differenza di una stufa lasciata a potenza fissa (*anello aperto*). |
| **Policy specialista** | Policy addestrata per una sola situazione, per esempio la discesa. Serve solo in simulazione, come esempio di comportamento corretto da imitare. |
| **Distillazione** | Trasferire il comportamento di una rete in un'altra facendole imitare le sue azioni. Qui si trasferisce il comportamento di una specialista nella policy ufficiale, ottimizzando solo un vettore di steering. |
| **DAgger** | Procedura per la distillazione: si fa agire la rete che impara, si raccolgono gli stati che visita, si chiede alla rete esperta quale azione avrebbe scelto e si riaddestra. Evita che chi impara si trovi in stati mai visti durante il training. |
| **Libreria di vettori** | L'insieme dei vettori di steering, uno per situazione, ciascuno con il proprio limite di sicurezza. |
| **Riconoscitore** | Piccola rete che, dagli ultimi secondi di misure del robot, stima in quale situazione si trova (per esempio 80% discesa, 20% piano) e, per le condizioni continue, il loro valore (per esempio la pendenza). |
| **Router** | Regola che, a ogni passo, somma alle activation i vettori della libreria pesati con le probabilità del riconoscitore. |
| **Controllore in stile CLAE** | Piccola rete con memoria, addestrata con reinforcement learning, che a ogni passo decide direttamente come modificare poche feature di un SAE. Nella tesi è un termine di confronto. |
| **Action residual** | Correzione appresa che si somma alle azioni della policy frozen, invece che alle sue activation. |
| **LoRA** (Low-Rank Adaptation) | Fine-tuning che modifica i pesi della policy aggiungendo un aggiornamento descritto da pochissimi parametri. |
| **Adapter** | Modifica piccola e separabile che adatta una rete frozen a un compito o a una situazione, senza riaddestrarla: si può attivare, togliere o scambiare. Può agire sui risultati intermedi (un vettore di steering sommato alle activation, senza aggiungere né modificare parametri della rete) oppure sui pesi (LoRA, che aggiunge due piccole matrici che correggono una matrice di pesi). |
| **Bias** | In un layer che calcola "pesi × input + bias", il vettore fisso che si somma sempre all'uscita. Sommare un vettore costante all'activation equivale a cambiare il bias. |
| **PPO** (Proximal Policy Optimization) | L'algoritmo di reinforcement learning usato da microduck_rl per addestrare la camminata. |
| **Rollout** | Un episodio in cui una policy già addestrata controlla il robot simulato, senza essere addestrata: a ogni passo il simulatore le dà le osservazioni, lei calcola l'azione, il simulatore la applica. Eseguire rollout serve a raccogliere dati e a valutare le prestazioni. |
| **ONNX** | Formato standard per salvare una rete neurale già addestrata ed eseguirla su altri dispositivi. Le policy del Microduck sono distribuite in questo formato. |
| **mjlab** | Libreria che esegue il simulatore fisico MuJoCo sulla GPU, con migliaia di copie del robot in parallelo. È la base di microduck_rl. |
| **Heightfield** | Un modo di descrivere un terreno come griglia di altezze, come una mappa topografica. Si usa per pendenze e superfici irregolari. |
| **Corpo mocap** | Un oggetto del simulatore che non risponde alla fisica ma si sposta esattamente dove indicato dal programma. Serve, per esempio, a far vibrare una piattaforma. |
| **Forward hook** | Funzione di PyTorch che si aggancia a un layer e viene chiamata ogni volta che il layer produce la sua activation. Permette di leggere l'activation o di modificarla prima che passi al layer successivo. |

---

## 2. Punto di partenza: cosa sa fare la policy ufficiale

**Com'è fatta.** È un MLP con tre hidden layer da 512, 256 e 128 neuroni, attivazione ELU e circa 198 mila parametri, calcolati dalla configurazione. Decide 50 volte al secondo le posizioni dei 14 servomotori (5 per gamba, 4 per collo e testa) a partire da 61 numeri: 48 misure interne (velocità angolare e inclinazione del corpo, posizioni e velocità dei motori, ultima azione) e 13 comandi (velocità richiesta e posa della testa). È distribuita come file ONNX, `alpha_walking.onnx`.

La policy ufficiale del Microduck (`alpha_walking`) è addestrata a **seguire comandi di velocità camminando su pavimento piano**. Riceve tre tipi di comando, cioè velocità in avanti, velocità laterale e velocità di rotazione, più la posa della testa, e deve muovere le gambe in modo che il robot vada alla velocità richiesta senza cadere. Il training è avvenuto solo su pavimento piano (task `Mjlab-Velocity-Flat-MicroDuck`), con alcune variazioni casuali per renderla robusta:

La tabella indica, per ogni variazione, l'intervallo usato nel training e la situazione reale che vuole riprodurre. Lo scopo di queste variazioni è far sì che la policy non dipenda da valori esatti che sul robot reale non si conoscono o cambiano da un giorno all'altro.

| Variazione nel training | Intervallo | Scenario reale che riflette |
|---|---|---|
| Attrito dei piedi | da 0,7 a 1,3 volte il valore nominale | Pavimenti diversi (piastrelle, parquet, linoleum) e piedini più o meno usurati o puliti. |
| Massa del corpo | ±5% | Piccole differenze tra un esemplare e l'altro del robot, dovute a montaggio, cavi e viti, e piccoli accessori fissati sul corpo. |
| Spinte al corpo | variazioni di velocità fino a ±0,3 m/s ogni 3-6 secondi | Urti leggeri, una mano che tocca il robot, un piede che inciampa su una piccola irregolarità. |
| Tensione della batteria | da 6,5 a 8,2 V, scelta a inizio episodio e poi fissa, più un calo momentaneo quando i motori fanno molto sforzo | Il livello di carica della batteria: a batteria carica i motori sono più forti e più rapidi, a batteria scarica più deboli. Il calo momentaneo riproduce quello che succede a qualunque batteria quando le si chiede molta corrente tutta insieme, per esempio durante un passo brusco. |
| Ritardo dei comandi ai motori | da 3 a 6 passi di controllo, cioè da 60 a 120 ms | Il tempo che passa tra il momento in cui la policy decide un'azione e il momento in cui il motore la esegue. Non riguarda segnali radio: il ritardo nasce all'interno del robot, dal tempo di calcolo sul computer di bordo, dalla comunicazione sul bus seriale che collega il computer ai 14 servomotori Dynamixel e dal controllo interno di ciascun servomotore. Varia da robot a robot e con il carico di lavoro del computer, quindi la policy deve funzionare con ritardi diversi. |

**Cosa sa fare bene:** camminare su pavimento piano alla velocità richiesta, anche con piccole spinte e piccole differenze di attrito e di massa.

**Cosa non ha mai visto:** pendenze, gradini, pavimenti molto scivolosi, carichi importanti. In queste situazioni ci si aspetta che cammini peggio o cada. Su terreno irregolare è già stato misurato dalla comunità: secondo la [scheda della policy `rough-walk-g`](https://huggingface.co/RemiFabre/microduck-rough-walk-g) su Hugging Face, la policy ufficiale cade 32 volte su 110 prove su terreno accidentato, contro le 12 della variante addestrata su terreno irregolare, che però consuma il 17% di potenza in più. Il confronto va riprodotto nella fase 0 (sezione 5).

**Cosa vede, e cosa no.** La policy è **cieca**, come si verifica nella configurazione delle osservazioni:
- tra i 61 input non ci sono telecamere né informazioni sul terreno davanti al robot. mjlab aggiungerebbe di default una scansione delle altezze del terreno, ma microduck_rl la rimuove esplicitamente perché il robot non ha un sensore montato sul corpo che la fornisca;
- non riceve nemmeno la propria velocità lineare, che durante il training è disponibile solo al critic;
- non ha memoria: vede solo l'istante corrente, senza la storia dei passi precedenti.

**Come reagisce a un ostacolo.** Non potendolo vedere, la policy se ne accorge solo quando lo tocca, e solo indirettamente: un piede che urta o il corpo che viene frenato cambiano posizioni e velocità dei motori, inclinazione e velocità angolare. Per lei è una perturbazione simile alle spinte viste nel training, e reagisce allo stesso modo, con passi di recupero. Ci si aspetta che:
- su ostacoli molto bassi, attorno a 1 cm, a volte passi e a volte inciampi, a seconda di dove cade il passo. Secondo un commento nel codice il robot solleva i piedi solo di 1-2 cm;
- su ostacoli più alti il piede urti e il robot inciampi, con buona probabilità di cadere;
- davanti a un muro continui a spingere o a camminare sul posto, perché il comando di velocità continua a chiedere di avanzare, finché non cade o finché il comando non cambia.

Sono previsioni basate su come è addestrata, non misure: si verificano in simulazione con i terreni con ostacoli di mjlab (per esempio `HfDiscreteObstaclesTerrainCfg`, sezione 4.6) oppure aggiungendo un blocco alla scena di `infer_policy.py`.

**Chi gestisce gli ostacoli.** Non la policy di camminata, ma chi le invia i comandi di velocità: l'utente con il controller, oppure un modulo di navigazione che usa altri sensori. La policy esegue "vai a questa velocità", non decide dove andare.

**Cosa implica per il resto del documento.**
- Quando si esegue la policy ufficiale su pendenze, gradini o pavimenti scivolosi non ci si aspetta che si adatti. Lo si fa per **osservarla**: vedere quanto peggiora, che è la misura del problema, e registrare come cambiano le sue activation, che servono al riconoscitore, ai probe e al SAE.
- Lo steering nella tesi ha un solo obiettivo: far fare alla policy ufficiale ciò che **non ha mai imparato**, come scendere bene una rampa. Quel comportamento non è dentro la policy, quindi non basta cercare una direzione tra le sue activation: si addestra una specialista che lo sa fare e se ne trasferisce il comportamento in un vettore (sezione 3.2).
- Gli **ostacoli da evitare non sono una situazione adatta** al sistema della tesi: il riconoscitore usa le stesse misure interne della policy, quindi se ne accorgerebbe solo dopo l'urto, troppo tardi per cambiare andatura. Diverso è il caso di **gradini bassi e terreno irregolare continuo**: dopo i primi passi il riconoscitore può capire che il terreno è sconnesso e applicare un adapter che, per esempio, alza di più i piedi nei passi successivi. Per questo il benchmark include gradini bassi e terreno irregolare, ma non ostacoli isolati.

---

## 3. Il metodo

Questa sezione descrive cosa si costruisce, nell'ordine in cui i pezzi si usano: cos'è lo steering, come si ottiene un vettore per una situazione, che tipo di adapter è, come riconoscitore e router lo applicano, come si verifica che funzioni e a cosa serve il SAE.

### 3.1 Cos'è lo steering, con un esempio

La policy del Microduck calcola l'azione in quattro stadi: dai 61 numeri in ingresso produce l'activation del primo hidden layer (512 numeri), poi quella del secondo (256 numeri), poi quella del terzo (128 numeri), e infine le 14 posizioni dei motori.

Prendiamo, come esempio, il secondo hidden layer. In un certo istante la sua activation è un vettore di 256 numeri, che chiamiamo **h**. I layer successivi trasformano **h** nell'azione.

Fare steering significa, in quell'istante, sostituire **h** con

> **h** + α · **v**

e lasciare che i layer successivi calcolino l'azione a partire da questo nuovo vettore. Qui:
- **v** è il **vettore di steering**, anch'esso di 256 numeri, scelto una volta per tutte;
- α è l'**intensità**, un singolo numero: con α = 0 non cambia nulla, con α più grande l'effetto cresce.

Se **v** è scelto bene, i layer successivi ricevono uno stato interno modificato che li porta a produrre le azioni giuste per la situazione: per esempio, in discesa, le azioni che sceglierebbe una policy capace di scendere bene. I pesi della policy restano identici.

**Su quale layer si fa lo steering?** Non serve farlo su ogni layer. Basta sommare il vettore all'uscita di **un** layer: la modifica si propaga da sola a tutti i layer successivi, e quindi all'azione. Quale layer scegliere è una decisione di progetto, da prendere con gli esperimenti, perché ogni scelta ha un significato diverso:
- **primo hidden layer (512 numeri):** vicino all'input. La correzione cambia il modo in cui la policy "interpreta" le misure del robot, prima che vengano elaborate;
- **secondo hidden layer (256 numeri):** a metà della rete, una via di mezzo;
- **terzo hidden layer (128 numeri):** vicino all'output. La correzione agisce quasi direttamente sul movimento; è il vettore più piccolo, ma ha meno strati di rete a disposizione per trasformarsi in azioni coerenti.

Si possono anche sommare vettori a più layer insieme, con più parametri da imparare. La tesi confronta le tre scelte a layer singolo e quella su tutti e tre (fase 3 del piano, sezione 5). Negli esempi di questo documento si usa il secondo hidden layer solo per avere numeri concreti: con un altro layer cambia la lunghezza del vettore (512 o 128 numeri invece di 256), non il procedimento.

Tutta la difficoltà sta nel **trovare v**. La sezione 3.2 spiega come: si ottiene un vettore per ogni **situazione** (discesa, gradini bassi, pavimento scivoloso, ...) facendo imitare alla policy ufficiale una policy specialista di quella situazione.

### 3.2 Distillazione da una policy specialista

**Il problema.** Scendere bene una rampa non fa parte di ciò che la policy ufficiale sa fare (sezione 2): non l'ha mai imparato. Il comportamento corretto va quindi preso da un'altra rete che lo conosce, e trasferito nella policy ufficiale senza toccarne i pesi.

**L'idea in una frase.** Si addestra una policy che sa scendere bene, la *specialista*, e poi si cerca il vettore che, sommato alle activation della policy ufficiale, la fa agire come la specialista quando il robot è in discesa.

**Perché non usare direttamente la specialista?** Si potrebbe sostituire la policy ufficiale con la specialista ogni volta che il robot è in discesa, ed è infatti uno dei termini di confronto (fase 5 del piano). I lavori più vicini a questa tesi spiegano perché conviene invece lasciare intatta la policy di base e aggiungerle una correzione.
- **Rifare il training o il fine-tuning rischia di far dimenticare ciò che la policy sapeva fare.** CLAE ([arXiv, giugno 2026](https://arxiv.org/abs/2606.11489)) osserva che modificare i pesi per ottenere un nuovo comportamento rischia di peggiorare le prestazioni di base della policy (*catastrophic forgetting*, cioè l'oblio di ciò che la rete aveva imparato) e obbliga a collaudarla di nuovo. Lo stesso gruppo, nel lavoro precedente LAE ([arXiv, settembre 2025](https://arxiv.org/abs/2509.20623)), definisce costoso correggere i rari fallimenti di una policy rifacendo il training o il fine-tuning, con il rischio di perdere capacità già acquisite.
- **Mantenere una versione completa per ogni comportamento costa.** Sempre secondo CLAE, tenere varianti separate della policy, una per ogni comportamento desiderato, aumenta i costi di distribuzione e di collaudo. Il problema cresce con la diffusione di modelli generalisti di grandi dimensioni.
- **I dati lo confermano.** Nello stesso lavoro, sul compito di far volare droni in formazione, il fine-tuning della policy raggiunge un errore di formazione di 0,199 ma riduce gli episodi senza collisioni dal 99,5% al 93,5%. La modifica delle activation lasciando intatti i pesi arriva a 0,086 con il 98,5% di episodi senza collisioni (errore di formazione: scostamento medio dalle distanze desiderate tra i droni, più basso è meglio).

A questi argomenti, tratti dalla letteratura, si aggiungono due motivi specifici del nostro caso:
- **Il passaggio da una situazione all'altra è graduale.** Sostituire una rete con un'altra è un cambio netto; una correzione si somma con un peso che il router fa crescere e decrescere, quindi le transizioni tra piano e discesa restano morbide;
- **Le correzioni si combinano.** In una discesa con un carico si sommano due correzioni; con le specialiste servirebbe una terza rete addestrata apposta per la combinazione.

Va detto con onestà che sul Microduck l'argomento della memoria pesa poco: una policy completa ha circa 198 mila parametri, quindi anche tenerne una per situazione è economico. Gli argomenti che contano sono il collaudo della policy, la gradualità delle transizioni e la possibilità di combinare le correzioni. Un vettore è una correzione piccola, con tanti numeri quanti sono i neuroni del layer scelto (da 128 a 512), che si somma con intensità graduale alla policy già collaudata.

**Passo 1: addestrare la specialista.** Si prende la policy ufficiale e si continua il suo training con PPO, con microduck_rl, su terreni in discesa (*fine-tuning*). Si parte dalla policy ufficiale invece che da zero per un motivo preciso: la specialista resta simile all'originale e cambia solo ciò che serve per scendere bene, quindi la differenza da colmare con un vettore è piccola. Prima di proseguire si verifica che la specialista scenda davvero meglio della policy ufficiale, per esempio con molte meno cadute. Se non è così, non c'è un comportamento migliore da trasferire e la situazione si scarta.

**Passo 2: raccogliere gli stati di discesa.** Si esegue la specialista nel simulatore su terreni in discesa, senza più addestrarla, e a ogni passo di controllo si salvano due cose:
- l'osservazione ricevuta (61 numeri);
- l'azione scelta dalla specialista (14 numeri, le posizioni target dei motori).

Il risultato è un dataset di coppie "in questo stato, la specialista fa questo", per esempio 100.000 coppie. Il numero esatto va verificato, ma il vettore da imparare ha al massimo 512 numeri, quindi servono molti meno dati che per addestrare una rete intera.

**Passo 3: cercare il vettore.** Il vettore **v_discesa** parte da zero. Per ogni coppia del dataset:
1. si dà l'osservazione alla policy ufficiale;
2. all'uscita del layer scelto, nell'esempio il secondo hidden layer, si somma **v_discesa** all'activation;
3. la policy, con i suoi pesi invariati, calcola l'azione;
4. si confronta questa azione con quella della specialista.

L'errore è la differenza al quadrato tra le due azioni, mediata sui 14 motori e su tutte le coppie. Un esempio con due soli motori: se la specialista sceglie [0,30; −0,10] e la policy ufficiale con il vettore sceglie [0,25; −0,05], l'errore è (0,05² + 0,05²) / 2 = 0,0025.

Si modifica **v_discesa** con la discesa del gradiente per ridurre questo errore. È lo stesso procedimento con cui si addestra una rete, con una differenza: gli unici parametri che cambiano sono i numeri del vettore. Tutti i pesi della policy ufficiale restano bloccati e servono solo a calcolare come l'errore dipende dal vettore. In PyTorch:

```python
# esempio con lo steering sul secondo hidden layer (256 neuroni)
v = torch.zeros(256, requires_grad=True)       # il vettore da imparare, parte da zero
for p in policy.parameters():
    p.requires_grad_(False)                     # i pesi della policy ufficiale restano fissi
ottimizzatore = torch.optim.Adam([v], lr=1e-3)  # si ottimizza solo v

for osservazioni, azioni_specialista in dataset:            # blocchi di coppie del passo 2
    h1 = policy.layer1(osservazioni)
    h2 = policy.layer2(h1) + v                  # lo steering: v sommato al secondo hidden layer
    h3 = policy.layer3(h2)
    azioni = policy.uscita(h3)
    errore = ((azioni - azioni_specialista) ** 2).mean()
    ottimizzatore.zero_grad(); errore.backward(); ottimizzatore.step()
```

I nomi dei layer sono indicativi: dipendono da come si ricostruisce la policy in PyTorch (sezione 4.5). Il training dura minuti su GPU, da misurare.

**Passo 4: correggere con DAgger.** Il passo 3 ha un difetto nascosto, che si capisce con un'analogia. Chi impara a guidare guardando soltanto un istruttore esperto vede sempre l'auto al centro della corsia, perché l'istruttore non sbaglia. Non vede mai cosa fare quando l'auto è un po' spostata verso il bordo. Alla prima piccola deviazione si trova in una situazione mai vista, non sa come correggerla e la deviazione cresce.

Con la policy succede lo stesso.
1. **Al passo 2 i dati vengono tutti dalla specialista.** Contengono solo gli stati in cui la specialista si trova quando scende bene: per esempio il corpo sempre inclinato in avanti di pochi gradi, mai di più.
2. **Al passo 3 il vettore impara a imitare la specialista solo in quegli stati.** Su di essi va bene, ma l'imitazione non è mai perfetta.
3. **Quando la policy con il vettore cammina da sola, i piccoli errori si accumulano.** Supponiamo che faccia passi appena più lunghi della specialista. Dopo qualche passo il corpo si sbilancia in avanti più di quanto la specialista abbia mai fatto, per esempio di 10 gradi invece di 5.
4. **In quegli stati il vettore non ha esempi da cui aver imparato.** Il dataset non contiene nessuno stato con il corpo inclinato di 10 gradi, perché la specialista non ci è mai arrivata. Il vettore produce azioni poco adatte, il corpo si sbilancia ancora di più e il robot può cadere.

Il problema quindi non è che il vettore sia sbagliato, ma che è stato addestrato sugli stati della specialista e poi viene usato sugli stati in cui si trova da solo.

**La soluzione: DAgger** (*Dataset Aggregation*, aggregazione del dataset). L'idea è raccogliere esempi proprio negli stati in cui la policy con il vettore finisce da sola, e chiedere alla specialista cosa avrebbe fatto lì. Si ripete un ciclo:
1. **Si fa camminare la policy ufficiale con il vettore attuale,** non la specialista, sulle discese del simulatore, e si salvano le osservazioni che incontra. Tra queste ci sono anche gli stati sbilanciati, per esempio con il corpo inclinato di 10 gradi.
2. **Per ciascuna di queste osservazioni si chiede alla specialista quale azione avrebbe scelto.** Non serve una nuova simulazione: basta dare l'osservazione in ingresso alla rete della specialista e leggerne l'uscita. Si ottiene così, per esempio, "con il corpo inclinato di 10 gradi la specialista avrebbe fatto un passo più corto per riprendere l'equilibrio".
3. **Si aggiungono queste nuove coppie a quelle già raccolte** e si ripete il passo 3 su tutto il dataset.

A ogni ciclo il dataset copre meglio gli stati in cui la policy con il vettore finisce davvero, e il vettore impara non solo a imitare la specialista quando tutto va bene, ma anche a **riportare il robot verso il comportamento della specialista quando se ne sta allontanando**. Il ciclo si interrompe quando le prestazioni nella situazione (cadute, errore di velocità) smettono di migliorare da un ciclo al successivo.

**Passo 5: misurare quanto funziona.** Si eseguono tre policy sulle stesse discese: la policy ufficiale da sola, la policy ufficiale con il vettore e la specialista. La misura chiave è il **vantaggio recuperato**: quanta parte del miglioramento della specialista il vettore riesce a ottenere. Esempio con le cadute su 100 prove:
- policy ufficiale: 30 cadute;
- specialista: 5 cadute;
- policy ufficiale con il vettore: 15 cadute;
- vantaggio recuperato = (30 − 15) / (30 − 5) = 0,6.

Il vettore recupera il 60% del vantaggio della specialista: 0 significa nessun miglioramento, 1 significa che il robot va come con la specialista. Si misura anche che il vettore non peggiori la camminata quando *non* serve, applicandolo per errore sul piano.

**Passo 6: fissare l'intensità sicura.** Il vettore è stato imparato per essere applicato "per intero". Il router (sezione 3.4) lo applicherà moltiplicato per un peso tra 0 e 1, la probabilità che il robot sia in discesa. Con la prova di steering della sezione 3.5 si verifica che pesi intermedi diano comportamenti intermedi e ragionevoli, e si misura fino a che intensità il vettore resta sicuro.


### 3.3 Il vettore come adapter: tre varianti

Si chiama *adapter* una modifica piccola e separabile che adatta una rete frozen senza riaddestrarla. Gli adapter si distinguono per il punto in cui intervengono: LoRA, il più noto, aggiunge alla rete due piccole matrici che correggono i pesi di un layer; il vettore di steering non aggiunge nulla alla rete e non ne tocca i pesi, ma si somma alle activation durante l'esecuzione. Accostare lo steering agli adapter conviene per due motivi.

*Primo motivo: chiarisce che cosa è, matematicamente.* Ogni layer della policy calcola "pesi × input + bias", dove il *bias* è un vettore fisso che si somma sempre. Sommare un vettore costante **v** all'activation produce lo stesso risultato che si otterrebbe cambiando quel bias, anche se il bias della rete resta intatto: il vettore si tiene a parte e si può togliere in qualunque momento. Se lo si somma prima della ELU del secondo layer, il bias b₂ diventa b₂ + **v**. La distillazione di un vettore costante è quindi un fine-tuning dei soli bias di un layer, una tecnica nota in NLP con il nome di BitFit (sezione 6.5).

*Secondo motivo: suggerisce varianti più potenti da confrontare.* Il vettore costante ha un limite. Il router gli assegna un peso α secondo la situazione, ma dentro la stessa situazione la correzione resta identica in ogni istante. Mentre scende, però, la specialista non fa sempre la stessa cosa: quando il piede è a terra frena il corpo, quando il piede è in aria accorcia il passo. Una correzione sempre identica non può riprodurre entrambi i comportamenti. Le due varianti seguenti producono invece una correzione che cambia da un istante all'altro, a seconda dello stato del robot.

**Variante 2: LoRA di rango 1.** Per capirla bisogna partire da come calcola un layer. Il layer riceve un input **x**, cioè l'activation del layer precedente, e calcola

> uscita = **W** · **x** + **b**

dove **W** è la matrice dei pesi e **b** il bias. L'input **x** cambia a ogni istante, perché dipende dalle misure del robot in quell'istante.

LoRA di rango 1 non tocca **W**: le affianca una piccola correzione dei pesi, costruita con due vettori da imparare, **u** e **w**. Il layer calcola allora

> uscita = **W** · **x** + **b** + **u** · (**w** · **x**)

L'ultimo termine è la correzione. Il fattore (**w** · **x**) è un singolo numero: si moltiplica ciascun numero dell'input **x** per il numero corrispondente di **w** e si sommano i risultati. Poiché dipende da **x**, questo numero è diverso a ogni istante. Nessuno lo sceglie: è la conseguenza automatica del fatto che LoRA corregge i **pesi**, e i pesi agiscono sempre moltiplicando l'input.

**La differenza con α.** α e (**w** · **x**) moltiplicano entrambi la correzione, ma rispondono a due domande diverse:
- **α**, deciso dal router, dice **in quale situazione** si è. In discesa vale circa 1 e resta tale per tutta la discesa;
- **(w · x)**, calcolato dal layer, cambia **all'interno della situazione**, istante per istante, seguendo lo stato del robot.

Un esempio in discesa, con α = 1 per tutto il tempo. Se durante la distillazione **w** impara a riconoscere il piede in aria, (**w** · **x**) vale circa 1 quando il piede è in aria e circa 0 quando è a terra. La correzione **u**, per esempio "accorcia il passo", agisce allora solo nella metà del passo in cui serve. Con il vettore costante questo non è possibile: α resta 1 per tutta la discesa e la correzione è sempre la stessa.

**Variante 3: piccola rete.** Una rete minuscola, con uno o due layer, riceve l'activation e restituisce direttamente il vettore da sommare, ricalcolato a ogni istante. Può quindi produrre correzioni di tipo diverso nei diversi momenti del passo: una che frena il corpo quando il piede è a terra, un'altra che accorcia il passo quando è in aria. Un'idea simile è usata da CLAE (sezione 6.1), che però addestra la rete con reinforcement learning invece che per distillazione.

| Variante | Correzione sommata all'activation | Dentro la stessa situazione, la correzione... | Numeri da imparare (secondo hidden layer) |
|---|---|---|---|
| 1. Vettore costante | α · **v** | è sempre la stessa | 256 |
| 2. LoRA di rango 1 | α · (**w** · **x**) · **u** | è sempre dello stesso tipo (**u**), ma con una forza (**w** · **x**) che cambia a ogni istante con l'input del layer | 256 + 512 = 768 |
| 3. Piccola rete | α · g(**h**) | può cambiare tipo e forza a ogni istante | Qualche migliaio |

**Come si confrontano.** Le tre varianti si addestrano allo stesso modo: stessa specialista, stessi dati, stesso ciclo DAgger della sezione 3.2. Cambia solo cosa si impara: il vettore **v**, i due vettori **u** e **w**, oppure i pesi della piccola rete. Poi vengono applicate dallo stesso riconoscitore e dallo stesso router. Si misura quanto ciascuna si avvicina alla specialista. Ci si aspetta che le varianti più flessibili imitino meglio; il vettore costante ha però un vantaggio di interpretazione, spiegato nella sezione 3.6. Nella stessa sezione, dopo aver introdotto il SAE, si descrive una quarta forma di correzione, guidata dal SAE, che cerca di unire flessibilità e leggibilità.

### 3.4 Riconoscitore e router

Il riconoscitore e il router decidono, a ogni passo di controllo, quali vettori della libreria applicare e con quale peso.

**Il riconoscitore.** È una piccola rete che riceve gli ultimi secondi di osservazioni del robot, eventualmente insieme ad alcune feature del SAE, e restituisce la probabilità di ciascuna situazione. Per le condizioni continue, come pendenza e attrito, restituisce anche una stima del valore. Si addestra in modo supervisionato, perché in simulazione la situazione vera è nota a ogni istante (sezione 4.11).

**Sulla finestra di 2 secondi del riconoscitore.** È un compromesso. Una finestra più lunga riconosce la situazione con più sicurezza; una più corta reagisce prima ai cambi di terreno. La lunghezza va scelta sui dati, confrontando 0,5, 1, 2 e 4 secondi per accuratezza e tempo di riconoscimento.

**Sul router.** Non si addestra: è una formula. Il vettore applicato è la somma dei vettori della libreria, ciascuno moltiplicato per la probabilità che il riconoscitore assegna alla sua situazione:

> v_applicato = p_salita · v_salita + p_discesa · v_discesa + p_gradini · v_gradini + …

Per le condizioni continue il peso è proporzionale al valore stimato: per esempio, più la pendenza stimata è ripida, più pesa v_discesa. Ogni vettore viene limitato al proprio limite di sicurezza (sezione 3.5). Se il riconoscitore è incerto su tutte le situazioni, non si applica alcun vettore e la policy cammina come in origine.

### 3.5 Come si verifica che un adapter funzioni

Un vettore distillato si considera valido solo dopo una **prova di steering** in simulazione, nella situazione per cui è stato costruito.
1. Lo si somma all'activation moltiplicato per un'intensità crescente: 0 (policy ufficiale da sola), 0,25, 0,5, 1 (il vettore così come è stato distillato) e 1,5, per vedere cosa succede oltre.
2. Per ogni intensità si misurano le prestazioni nella situazione (cadute, errore tra velocità richiesta ed effettiva, potenza dei motori) e il comportamento sul piano, dove il vettore non dovrebbe servire.
3. Il vettore è valido se le prestazioni nella situazione migliorano in modo regolare al crescere dell'intensità e se, applicato per errore sul piano, non peggiora la camminata in modo significativo.
4. L'intensità oltre cui le cadute aumentano è il **limite di sicurezza** del vettore: il router non la supera mai, né in simulazione né sul robot reale.

### 3.6 Il ruolo del SAE: guidare e leggere lo steering

Il SAE può servire in due momenti: **prima** della distillazione, per decidere dove e come intervenire, e **dopo**, per capire cosa fa una correzione già trovata. È lo stesso strumento in entrambi i casi; prima si spiega come funziona, poi i due usi.

**Come funziona il SAE, con un esempio.** Si registrano le activation del layer scelto mentre la policy cammina in molte situazioni (sezione 4.8) e su di esse si addestra il SAE. Il SAE impara due cose:
- un **dictionary**, cioè un elenco di vettori della stessa lunghezza dell'activation, per esempio 1.024. Ogni vettore dell'elenco si chiama *elemento* del dictionary;
- un modo per scrivere **ogni** activation come somma di **pochi** elementi del dictionary, ciascuno con un peso. Per esempio, in un certo istante:

  > activation = 0,8 × (elemento 12) + 0,3 × (elemento 57)

  e tutti gli altri elementi hanno peso zero.

Dopo il training si guarda, per ogni elemento, **in quali istanti ha un peso alto**, e lo si confronta con ciò che il simulatore registra in quegli istanti. Se l'elemento 12 ha peso alto esattamente quando il piede sinistro è a terra, e solo allora, l'elemento 12 rappresenta il concetto "piede sinistro a terra". In questo modo ogni elemento riceve un nome.

Il motivo per cui serve è che un singolo neurone non ha un significato preciso: ciascuno partecipa a molti concetti insieme. Gli elementi del dictionary, invece, tendono a corrispondere a un concetto ciascuno.

#### Uso 1, prima: guidare lo steering

È il modo in cui il SAE è usato da CLAE (sezione 6.1), e diventa la quarta forma di correzione da confrontare con le tre della sezione 3.3. Si procede in due passi.

**Passo A: scegliere gli elementi rilevanti per ciascuna situazione.** Per ogni elemento del dictionary si confronta il suo peso medio in discesa con quello in piano, usando le activation registrate. Gli elementi che cambiano di più tra le due situazioni sono quelli legati alla discesa. Se ne tengono pochi, per esempio i primi 8. È lo stesso criterio di CLAE, che sceglie gli elementi misurando quanto il loro peso è correlato al comportamento che interessa. Il numero di elementi è un compromesso da scegliere con gli esperimenti: pochi rendono la correzione semplice e leggibile, troppi la riportano verso la libertà della modifica libera.

**Passo B: modificare solo quegli elementi.** A ogni istante:
1. il SAE scrive l'activation come somma di elementi del dictionary;
2. per ciascuno degli elementi scelti al passo A, il peso viene **moltiplicato** per un fattore e gli viene **sommato** un valore, per esempio "peso di *passo corto* × 1,5 + 0,2". Gli altri elementi restano invariati;
3. dai pesi così modificati si ricostruisce l'activation, che prosegue nella policy.

I fattori e i valori da sommare, due numeri per ciascun elemento scelto (16 numeri con 8 elementi), si imparano con la stessa distillazione della sezione 3.2: stessa specialista, stessi dati, stesso ciclo DAgger. Il router li applica con il peso α della situazione: con α = 0 il fattore torna a 1 e il valore sommato a 0, e la policy cammina come in origine.

**Perché è interessante.** Questa forma unisce due vantaggi che nella sezione 3.3 sembravano in conflitto:
- **è leggibile per costruzione:** si sa sempre quali concetti si stanno modificando, perché gli elementi scelti hanno un nome;
- **cambia con lo stato del robot:** la parte moltiplicativa agisce in proporzione a quanto l'elemento è già attivo. "Peso di *passo corto* × 1,5" ha effetto solo negli istanti in cui *passo corto* è attivo, come succede con LoRA, ma con un significato chiaro.

Limitare la correzione a pochi elementi del dictionary sembra anche renderla più sicura. CLAE confronta due controllori che modificano a ogni passo le activation di una policy per droni, per farli volare in formazione a quadrato. L'errore di formazione misura quanto le distanze tra i droni si scostano da quelle desiderate: 0 è la formazione perfetta, più basso è meglio.

| Configurazione | Errore di formazione | Episodi senza collisioni |
|---|---|---|
| Policy senza intervento | 0,280 | 99,5% |
| Activation sostituita dalla sua ricostruzione con il SAE, senza alcuna modifica | 0,284 | 99,1% |
| Controllore libero di modificare tutti i 30 numeri dell'activation | 0,797 | 66,6% |
| Controllore che modifica solo pochi elementi del dictionary | 0,086 | 98,5% |

La seconda riga mostra un controllo da fare anche nella tesi. Il SAE non ricostruisce l'activation in modo perfetto, quindi bisogna verificare che la policy, con l'activation sostituita dalla sua ricostruzione e nessuna modifica, cammini ancora come prima. Se non è così, il SAE va migliorato prima di usarlo per guidare lo steering.

**Un rischio specifico del nostro caso.** La policy del Microduck è piccola: layer da 512, 256 e 128 neuroni. In una rete piccola i concetti potrebbero essere meno mescolati tra i neuroni, e il SAE potrebbe dare un vantaggio minore che in reti più grandi. È una delle cose che il confronto misura.

#### Uso 2, dopo: leggere una correzione già trovata

Se la correzione è stata trovata senza SAE, per esempio il vettore costante libero della sezione 3.3, la si può comunque scomporre negli elementi del dictionary:

> **v_discesa** = 0,9 × ("passo corto") + 0,6 × ("corpo arretrato") + piccoli contributi di altri elementi

Questo dice, in parole comprensibili, che cosa la policy ufficiale ha imparato dalla specialista. Confrontando i vettori di situazioni diverse si vede anche cosa le distingue: discesa e pavimento scivoloso potrebbero condividere "passo corto" ma differire nella posizione del corpo.

La lettura funziona direttamente solo per il vettore costante, perché lì la correzione è un unico vettore. Per LoRA si legge il vettore **u**, cioè il tipo di correzione, ma non la regola con cui la sua forza (**w** · **x**) cambia nel tempo. Per la piccola rete non c'è un vettore unico da leggere. La forma guidata dal SAE dell'uso 1 non ha questo problema, perché è leggibile fin dall'inizio.

#### Uso 3, opzionale: aiutare il riconoscitore

I pesi degli elementi del dictionary nell'istante corrente, per esempio quello di "corpo inclinato in avanti", sono informazioni già elaborate sullo stato del robot e possono essere input utili per il riconoscitore.

---


## 4. Guida all'implementazione

Questa sezione segue l'ordine in cui il lavoro si svolge:
- **4.1-4.2, strumenti:** il codice da cui si parte e le librerie utili;
- **4.3-4.5, preparazione:** come è organizzata la simulazione, come si installa tutto, come si ottiene la policy ufficiale in PyTorch;
- **4.6-4.9, scenari e dati:** come si costruiscono terreni e condizioni dinamiche, cosa si registra e con quale programma;
- **4.10-4.11, steering e training:** come si applica un vettore nel codice e cosa si addestra, con quali dati.

Nomi di classi, funzioni e attributi sono stati verificati su microduck_rl (commit `cfe1c2a`), su mjlab 1.3.0 e sul pacchetto `better-actuator-models` (importato come `bam`), che simula i motori. Dove un dettaglio non è stato verificato lo si dice esplicitamente.

**Idea d'insieme.** Non si registrano video. I dati sono numeri, generati automaticamente in simulazione, senza annotazioni manuali. Si esegue la policy ufficiale, senza addestrarla, su migliaia di copie del robot simulate in parallelo, ciascuna su un pezzo di terreno e in condizioni diverse. A ogni passo di controllo si salvano tre cose:
- ciò che la rete calcola internamente (le activation);
- ciò che succede fisicamente, letto dal simulatore (il ground truth);
- le condizioni dello scenario in quel momento.

### 4.1 Codice di partenza

Non serve partire da zero. Quasi tutta l'infrastruttura esiste già; il codice nuovo è circoscritto.

| Componente | Codice di partenza | Cosa scriviamo noi |
|---|---|---|
| Simulatore, ambienti di camminata, domain randomization, terreni irregolari e pendenze | [microduck_rl](https://github.com/pollen-robotics/microduck_rl): basato su [mjlab](https://github.com/mujocolab/mjlab) 1.3.0 e su rsl_rl per PPO. Include un generatore di terreni con gradini, selciato e pendenze, e un terreno personalizzato con rampa in discesa (`slope_terrain.py`). | Nulla per l'ambiente di base. |
| Policy da studiare | `alpha_walking.onnx` (policy ufficiale) e varianti della comunità su Hugging Face, per esempio [`rough-walk-g`](https://huggingface.co/RemiFabre/microduck-rough-walk-g). | Convertire i pesi da ONNX a PyTorch. L'architettura è nota (MLP 61 → 512 → 256 → 128 → 14), quindi basta copiare le matrici di pesi e verificare che le uscite coincidano. |
| Esecuzione e valutazione di una policy ONNX | `scripts/infer_policy.py` di microduck_rl. | Una versione con forward hook e registrazione delle activation. |
| Training di nuove policy | `train` di microduck_rl (PPO di rsl_rl) ed export ONNX con `scripts/export.py`. | Configurazioni per la famiglia di policy (semi, larghezze, memoria). |
| Condizioni dinamiche | Parzialmente presenti: attrito, massa, spinte, terreno. | Cambio di attrito e di carico a metà episodio, piattaforma che vibra. |
| SAE | [overcomplete](https://github.com/KempnerInstitute/overcomplete), che accetta un qualunque tensore di activation. | Addestramento e analisi delle feature. |
| Specialiste, libreria, riconoscitore e router | Il training delle specialiste usa il comando `train` di microduck_rl. Distillazione, riconoscitore e router sono codice PyTorch standard. | Il ciclo di distillazione con DAgger, il riconoscitore, il router e la loro valutazione. |
| Controllore in stile CLAE (baseline) | Il metodo di CLAE è descritto in dettaglio nel paper. Il codice di LAE, il lavoro precedente degli stessi autori, è indicato come disponibile sulla [pagina del progetto](https://lae-robotics.github.io/). | Un wrapper dell'ambiente in cui l'azione è il vettore delle intensità e la policy frozen produce i comandi ai motori; training con il PPO di rsl_rl. |
| Esecuzione sul robot reale | Runtime del [Microduck](https://github.com/pollen-robotics/microduck), che esegue policy ONNX. | Esportare in ONNX un unico grafo con policy, riconoscitore, router e libreria. Va verificato che il calcolatore di bordo regga il controllore a 50 Hz. |

### 4.2 Framework utili

| Strumento | A cosa serve | PRO | CONTRO |
|---|---|---|---|
| [overcomplete](https://github.com/KempnerInstitute/overcomplete) (Kempner Institute, Harvard) | Libreria PyTorch per SAE con varianti TopK, BatchTopK, JumpReLU e classica, più notebook di esempio. | Accetta direttamente una matrice di activation, senza assumere un transformer; si installa con pip. | Documentazione orientata ai modelli di visione; le analisi automatiche delle feature sono pensate per le immagini. |
| SAELens, Sparsify (EleutherAI), sparse_autoencoder (OpenAI) | Librerie mature per SAE sugli LLM, citate anche dalla documentazione di overcomplete. | Molto usate e curate. | Pensate per agganciarsi a transformer e testo: per un MLP da 198 mila parametri sono sovradimensionate. |
| SAE scritto a mano in PyTorch | Due layer lineari e una funzione di sparsità, circa 50 righe. | Controllo completo; nessuna dipendenza; utile anche per capire il metodo. | Bisogna gestire da sé i problemi noti, come gli elementi del dictionary che smettono di attivarsi. |
| Forward hook di PyTorch | Leggere e modificare le activation durante l'esecuzione. | Già incluso in PyTorch; bastano poche righe (sezione 4.10). | Nessuno rilevante per un MLP così piccolo. |
| mjlab + rsl_rl | Simulazione e PPO per le specialiste e le baseline. | Già lo stack di microduck_rl; nessuna integrazione da fare. | mjlab è recente e cambia spesso: va fissata la versione. |

### 4.3 Come è organizzata la simulazione

microduck_rl si appoggia su **mjlab**, una libreria che esegue MuJoCo sulla GPU (tramite MuJoCo Warp). Cinque concetti bastano per orientarsi.

| Concetto | Cosa è | Dove si configura |
|---|---|---|
| **Ambiente** | Una copia indipendente del robot nel simulatore. mjlab ne simula migliaia insieme sulla GPU, ciascuna su una porzione diversa del terreno. | `cfg.scene.num_envs` |
| **Episodio** | Una camminata di una copia, dall'avvio alla caduta o allo scadere del tempo. Quando finisce, quella copia riparte da capo (*reset*). | Nella configurazione dei *termination* |
| **Terreno** | Il pavimento: piano oppure generato a pezzi di tipo diverso (gradini, pendenze, sassi). | `cfg.scene.terrain` |
| **Evento** | Una funzione che modifica la simulazione in un momento preciso: all'avvio (`"startup"`), a ogni reset (`"reset"`), a intervalli di tempo (`"interval"`) o a ogni passo (`"step"`). È così che si cambiano attrito, masse, spinte e, per noi, le condizioni durante la camminata. | `cfg.events` |
| **Comando** | La velocità che si chiede al robot (avanti, di lato, rotazione), ricampionata ogni tanto durante l'episodio. | Configurazione del comando di velocità |

Tutto parte da una configurazione Python: per la camminata del Microduck è la funzione `make_microduck_velocity_env_cfg` nel file `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`. I task registrati che ci interessano sono:
- `Mjlab-Velocity-Flat-MicroDuck`: pavimento piano, il task da cui deriva la policy ufficiale;
- `Mjlab-Velocity-Rough-MicroDuck`: terreno generato a pezzi, la base per il nostro benchmark.

### 4.4 Installazione e primo avvio

1. Clonare microduck_rl e installare le dipendenze con `uv`, il gestore di pacchetti usato dal repository. Le versioni sono già fissate nel file `uv.lock` (mjlab 1.3.0, torch 2.9.1, Python 3.12). Le istruzioni esatte sono nel README del repository.
2. Elencare i task disponibili con lo script `list_envs` di mjlab.
3. Guardare i terreni con lo script `visualize_terrain` di mjlab, che li mostra senza robot. Serve a controllare a occhio un terreno appena configurato.
4. Guardare la policy che cammina:
   - con `scripts/infer_policy.py` di microduck_rl, che esegue un file ONNX in MuJoCo con il viewer grafico. Simula un solo robot ed è lento: serve per guardare e fare video, non per raccogliere dati;
   - con il comando `play` di mjlab, che esegue un checkpoint su più ambienti.

Le opzioni da riga di comando di ciascuno script si leggono con `--help`.

### 4.5 Ottenere la policy in PyTorch

Per leggere e modificare le activation serve la policy come modulo PyTorch: un file ONNX non permette di agganciare i forward hook. Ci sono due strade, e conviene percorrerle entrambe.

**Strada A: convertire la policy ufficiale.** L'architettura è nota: un MLP con layer 61 → 512 → 256 → 128 → 14 e attivazione ELU. Basta ricrearlo in PyTorch e copiarci dentro i pesi del file ONNX.
```python
import onnx, onnxruntime as ort, numpy as np, torch
from onnx import numpy_helper

modello = onnx.load("alpha_walking.onnx")
pesi = {t.name: numpy_helper.to_array(t) for t in modello.graph.initializer}
for nome, valore in pesi.items():
    print(nome, valore.shape)   # 1) si guarda quali tensori ci sono e con che forma
```
Dalla stampa si riconoscono le quattro matrici di pesi, con forme 512×61, 256×512, 128×256 e 14×128 o le trasposte, più i bias. Va controllato anche se il file contiene una **normalizzazione degli input**: rsl_rl normalizza le osservazioni sottraendo una media e dividendo per una deviazione standard, e l'export ONNX può includerla come primi passi del grafo. Se c'è, va ricopiata nel modello PyTorch.

L'ultimo passo è la verifica: si danno gli stessi 10.000 input casuali al modello ONNX (con `onnxruntime`) e al modello PyTorch, e si controlla che la differenza massima tra le uscite sia trascurabile, per esempio sotto 10⁻⁵. Senza questa verifica ogni risultato successivo è inaffidabile.

**Strada B: addestrare una policy propria.** Con il comando `train` di microduck_rl sul task `Mjlab-Velocity-Flat-MicroDuck` si ottiene direttamente un checkpoint PyTorch di rsl_rl. Secondo le schede pubblicate dalla comunità servono circa 4.000 iterazioni, circa un'ora su una RTX 5090. Questa strada serve comunque per la famiglia di policy della fase 4, e permette di controllare la domain randomization.

### 4.6 Creare terreni diversi

**Come funziona il generatore.** mjlab costruisce il terreno come una **griglia di pezzi quadrati** (*patch*). Nel Microduck ogni pezzo misura 8 × 8 m e la griglia ha 10 righe e 20 colonne. Le copie del robot vengono distribuite sui pezzi.
- Le **colonne** corrispondono ai *tipi* di terreno: la configurazione elenca i tipi (*sub_terrains*) e, per ciascuno, la frazione di colonne che occupa (`proportion`).
- Le **righe** corrispondono alla *difficoltà*: lo stesso tipo diventa più difficile scendendo di riga, per esempio gradini più alti o pendenze più ripide, entro gli intervalli indicati.

La configurazione attuale del Microduck, `MICRODUCK_ROUGH_TERRAINS_CFG`, è un buon esempio:
```python
MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.25, step_height_range=(0.0, 0.015), ...),
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.30, grid_width=0.45, grid_height_range=(0.0, 0.010), ...),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20, slope_range=(0.03, 0.10), ...),
    },
)
```
Si legge così: un quarto delle colonne è piano, un quarto ha gradini alti fino a 1,5 cm, il 30% ha un selciato irregolare con dislivelli fino a 1 cm, il 20% ha pendenze. `slope_range` è espresso come dislivello diviso distanza orizzontale: da 0,03 a 0,10, cioè da circa 1,7° a 5,7°.

**Tipi di terreno disponibili in mjlab 1.3.0.** Il prefisso dice come è costruito il terreno: `Box` indica un insieme di parallelepipedi, `Hf` (*heightfield*) una superficie continua definita da una griglia di altezze.

| Tipo | Cosa produce | Uso per noi |
|---|---|---|
| `BoxFlatTerrainCfg` | Pavimento piano. | Riferimento. |
| `HfPyramidSlopedTerrainCfg` | Una piramide con piattaforma piana in cima: il robot parte in alto e cammina in discesa, in salita o di traverso. | Salita e discesa. |
| `HfWaveTerrainCfg` | Ondulazioni regolari. | Avvallamenti morbidi. |
| `HfRandomUniformTerrainCfg` | Superficie con piccoli dislivelli casuali. | Pavimenti irregolari, esterni. |
| `HfPerlinNoiseTerrainCfg` | Superficie irregolare con variazioni graduali. | Terreno naturale. |
| `BoxRandomGridTerrainCfg` | Selciato di blocchi ad altezze diverse. | Pavimento sconnesso. |
| `BoxPyramidStairsTerrainCfg`, `BoxRandomStairsTerrainCfg`, `BoxOpenStairsTerrainCfg` | Gradini in varie disposizioni. | Gradini bassi, soglie. |
| `BoxTiltedGridTerrainCfg` | Blocchi inclinati. | Superfici sconnesse inclinate. |
| `BoxSteppingStonesTerrainCfg`, `BoxNarrowBeamsTerrainCfg`, `BoxNestedRingsTerrainCfg`, `BoxRandomSpreadTerrainCfg`, `HfDiscreteObstaclesTerrainCfg` | Pietre distanziate, travi strette, anelli, blocchi sparsi, ostacoli. | Probabilmente troppo difficili per un robot che alza i piedi di 1-2 cm. |

**Configurazione proposta per il benchmark.** Si parte da quella esistente e la si modifica così:
- pendenze fino a 0,18 (circa 10°), quasi il doppio del massimo visto nel training, per misurare cosa succede oltre;
- un tipo `HfWaveTerrainCfg` e un tipo `HfRandomUniformTerrainCfg` con ampiezze di 1-2 cm, compatibili con l'altezza a cui il robot solleva i piedi;
- per le discese lunghe, il terreno personalizzato già scritto in `src/mjlab_microduck/tasks/slope_terrain.py`: una zona piana seguita da una rampa in discesa con angolo da 2° a 20° secondo la difficoltà. È nato per un altro task, ma la classe si può riusare.

**Due avvertenze dal codice esistente.**
- Il commento in `microduck_velocity_env_cfg.py` spiega perché è stato tolto `BoxInvertedPyramidStairsTerrainCfg`: con le piramidi rovesciate il robot viene posizionato sotto il pavimento e cade nel vuoto. Lo stesso rischio vale per qualunque terreno "a fossa".
- I terreni fatti di blocchi possono creare instabilità numeriche negli spigoli. Il repository le attenua ammorbidendo i contatti con la funzione `_soften_terrain_contacts`, da mantenere.

**Come si assegna il terreno.** Nella configurazione dell'ambiente:
```python
cfg.scene.terrain.terrain_type = "generator"
cfg.scene.terrain.terrain_generator = IL_NOSTRO_TERRAIN_CFG
cfg.scene.terrain.terrain_generator.curriculum = False   # tutte le difficoltà subito, non progressivamente
```

**Come sapere su quale terreno si trova ogni copia.** L'oggetto terreno della scena tiene, per ogni copia, la riga (attributo `terrain_levels`, cioè la difficoltà) e la colonna (attributo `terrain_types`, cioè il tipo). Si salvano insieme agli altri dati. Per sapere la pendenza *esattamente sotto i piedi* in un istante si usa il sensore di altezza del terreno già presente nella configurazione (`TerrainHeightSensorCfg`, un raggio per piede).

### 4.7 Rendere le condizioni dinamiche

Ogni condizione si realizza con un **evento** di mjlab. Per un cambiamento controllato si scrive una funzione che, a ogni passo, decide il valore in base al tempo trascorso nell'episodio, e la si registra con `mode="step"`.

**Pavimento che diventa scivoloso.** Oggi l'attrito dei piedi si sceglie all'avvio con l'evento `foot_friction`, che usa la funzione `dr.geom_friction` di mjlab con intervallo da 0,7 a 1,3. Per cambiarlo durante la camminata ci sono due modi.
1. *Programma temporale.* Un evento a ogni passo che, dopo un istante scelto a caso nell'episodio, porta l'attrito dei piedi a un valore basso, per esempio 0,4, sotto il minimo del training. Si può riusare `dr.geom_friction` sugli stessi geom dei piedi, applicandola solo alle copie che hanno raggiunto l'istante del cambio.
2. *Zone del pavimento.* Si costruisce un tipo di terreno personalizzato con fasce a basso attrito, partendo dal modello di `slope_terrain.py`, che mostra come generare la geometria di un terreno. **Attenzione:** in MuJoCo, quando due superfici si toccano, l'attrito usato è il *massimo* dei due. Abbassare solo l'attrito del pavimento non basta se quello dei piedi resta alto: bisogna abbassare anche quello dei piedi oppure usare il campo `priority` dei geom, che permette di imporre l'attrito della superficie con priorità maggiore. La documentazione di MuJoCo spiega entrambi i meccanismi.

Il modo 1 è più semplice e va fatto per primo.

**Carico.** L'evento `randomize_mass_inertia` usa la funzione `dr.pseudo_inertia` all'avvio, con ±5%. Per un carico che compare durante la camminata (un oggetto raccolto) si usa un evento a ogni passo con la funzione `dr.body_mass` sul corpo del robot, aumentando la massa da un istante in poi.

**Spinte.** L'evento `push_robot` esiste già: imposta di colpo la velocità del corpo a intervalli casuali, oggi entro ±0,3 m/s ogni 3-6 secondi. Per il benchmark basta allargare `velocity_range`.

**Vibrazioni.** È l'unica condizione che richiede di aggiungere un oggetto alla scena. Si aggiunge una piattaforma sotto il robot come corpo *mocap*, cioè un corpo che non risponde alla fisica ma si muove esattamente dove gli si dice. A ogni passo un evento ne aggiorna l'altezza con un'oscillazione sinusoidale, di ampiezza e frequenza scelte per episodio; mjlab permette di scrivere la posizione di un corpo mocap con il metodo `write_mocap_pose`. Il robot che cammina sopra sente le vibrazioni attraverso i contatti. Va verificato come aggiungere un corpo alla scena di mjlab; il repository modifica già la scena con una funzione `spec_fn` (si veda `_soften_terrain_contacts`), che è il punto di partenza naturale. Se richiedesse troppo lavoro, si tralascia: le altre condizioni bastano.

**Comandi di velocità.** I comandi si ricampionano durante l'episodio. Per i dati conviene coprire bene velocità in avanti, laterali e di rotazione, così da avere anche cambi di andatura.

### 4.8 Cosa registrare a ogni passo

| Gruppo | Contenuto | Dove si legge | Numeri per passo |
|---|---|---|---|
| Input della policy | Le 61 osservazioni (misure interne e comandi) | Osservazioni dell'ambiente | 61 |
| Activation | I tre hidden layer | Forward hook sul modello PyTorch | 512 + 256 + 128 = 896 |
| Azione | Le 14 posizioni target dei motori | Uscita della policy | 14 |
| Moto del corpo | Velocità lineare e angolare nel riferimento del robot; inclinazione rispetto alla gravità | Proprietà `root_link_lin_vel_b`, `root_link_ang_vel_b`, `projected_gravity_b` dei dati del robot | 9 |
| Piedi | Contatto di ciascun piede, tempo da cui è a terra o in aria, altezza da terra | Sensore di contatto dei piedi (`found`, `current_contact_time`, `current_air_time`) e sensore di altezza del terreno | circa 8 |
| Motori | Coppie applicate | Proprietà `joint_torques` o `actuator_force` | 14 |
| Scenario | Tipo e difficoltà del terreno, attrito attuale, massa aggiunta, ampiezza della vibrazione, spinta in corso | Terreno della scena e variabili impostate dai nostri eventi | circa 8 |
| Esito | Se la copia cade entro i 2 secondi successivi | Calcolato dopo, rileggendo i dati | 1 |

Due grandezze utili non sono misurate direttamente e si calcolano dai dati salvati:
- la **fase del passo**, dal tempo di contatto e di volo di ciascun piede;
- la **potenza dei motori**, come coppia per velocità dei giunti.

### 4.9 Il programma di raccolta

Conviene partire dallo script `play` di mjlab (`mjlab/scripts/play.py`), che costruisce già l'ambiente con `load_env_cfg`, lo avvolge con `RslRlVecEnvWrapper` e carica una policy. Lo si copia e lo si modifica in tre punti: il terreno e gli eventi (sezioni 4.6 e 4.7), gli hook sulla policy, il salvataggio. Lo schema:

```python
import torch
import mjlab_microduck                                  # registra i task del Microduck
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

cfg = load_env_cfg("Mjlab-Velocity-Rough-MicroDuck")
cfg.scene.num_envs = 1024                              # copie in parallelo
cfg.scene.terrain.terrain_generator = BENCHMARK_TERRAIN_CFG
cfg.events["cambio_attrito"] = EVENTO_ATTRITO          # sezione 4.7
cfg.events["carico"] = EVENTO_CARICO
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))

policy = carica_policy_pytorch()                       # sezione 4.5
activation = {}
for nome, layer in [("h1", policy[1]), ("h2", policy[3]), ("h3", policy[5])]:
    layer.register_forward_hook(lambda m, i, o, n=nome: activation.__setitem__(n, o.detach()))

obs = env.get_observations()
for t in range(n_passi):
    with torch.no_grad():
        azione = policy(obs["policy"])                 # gli hook riempiono `activation`
    obs, _, done, _ = env.step(azione)
    salva_passo(obs, activation, azione, ground_truth(env), scenario(env), done)
```

Gli indici dei layer (`policy[1]`, `policy[3]`, `policy[5]`) presuppongono un `nn.Sequential` che alterna layer lineari e ELU, con l'hook sull'uscita della ELU. La chiave `"policy"` delle osservazioni e la forma esatta del valore restituito da `env.step` vanno verificate nella classe `RslRlVecEnvWrapper` di mjlab 1.3.0.

**Dove salvare.** Un file per blocco di passi, in formato NumPy (`.npz`) o Zarr. Le activation si salvano in mezza precisione (float16) per dimezzare lo spazio. Ogni riga porta l'indice della copia e il tempo dall'inizio dell'episodio, così che le sequenze si possano ricostruire.

**Quanti dati.** Un passo occupa circa 4 KB in float32, quasi tutti dovuti alle 896 activation. Una raccolta di 1.024 copie per 20 secondi a 50 passi al secondo produce circa un milione di passi, cioè circa 4 GB, circa 2 GB in float16. Un milione di passi è un punto di partenza ragionevole per un SAE sul layer da 512 neuroni. Si verifica che basti controllando che la qualità della ricostruzione smetta di migliorare aggiungendo dati.

### 4.10 Applicare uno steering nel codice

In PyTorch lo steering richiede poche righe. Si usa un **forward hook**, una funzione che PyTorch chiama ogni volta che un layer produce la sua activation e che può restituire un'activation modificata:

```python
v = vettore_di_steering      # lungo quanto il layer scelto (512, 256 o 128 numeri), distillato da una specialista (sezione 3.2)
alpha = 0.5                  # intensità; nel sistema completo la decide il router

def hook(modulo, input, output):
    return output + alpha * v   # i layer successivi ricevono l'activation modificata

handle = secondo_hidden_layer.register_forward_hook(hook)
# ... da qui ogni chiamata alla policy usa l'activation modificata ...
handle.remove()              # per tornare alla policy originale
```

### 4.11 Cosa si addestra, e con quali dati

La policy di camminata ufficiale **non** si addestra: resta frozen. Si addestrano i componenti seguenti.

| Cosa | Input | Output | Come si addestra | Costo |
|---|---|---|---|---|
| Policy specialiste, una per situazione | Le stesse 61 osservazioni della policy ufficiale | Le 14 posizioni dei motori | Fine-tuning della policy ufficiale con PPO nella sola situazione, oppure training da zero con microduck_rl | Da zero: circa un'ora su RTX 5090 su piano e circa 5 ore e mezza su terreno irregolare, secondo le schede della comunità. Il fine-tuning dovrebbe costare meno: da misurare |
| Adapter della libreria, uno per situazione | Nessuno: sono parametri da ottimizzare | Tre varianti: vettore costante (da 128 a 512 numeri, secondo il layer), LoRA di rango 1 (per il secondo hidden layer circa 768 numeri), piccola rete | Distillazione supervisionata dalla specialista, con il ciclo DAgger (sezione 3.2); stessa procedura per le tre varianti | Minuti su GPU per adapter, da misurare |
| Riconoscitore | Gli ultimi 2 secondi di osservazioni (100 passi di controllo), eventualmente con alcune feature del SAE | La probabilità di ciascuna situazione; per le condizioni continue (pendenza, attrito) una stima del valore | Classificazione e regressione supervisionate sui dati registrati, perché in simulazione la situazione vera è nota a ogni istante | Minuti su GPU, da misurare |
| Probe lineari e SAE | Activation di un layer | Una grandezza fisica (probe) o la ricostruzione dell'activation (SAE) | Sui dati registrati | Secondi (probe); minuti (SAE), da misurare |
| Baseline di confronto | Variano | Variano | PPO nel simulatore: controllore in stile CLAE, action residual, policy allievo con memoria. LoRA è invece una delle tre varianti di adapter, distillata come le altre (sezione 3.3) | Da misurare nella fase 5 |

**Con quali dati.** I dati registrati servono a quasi tutto, perché i componenti principali si addestrano in modo supervisionato.

| Componente | Dati usati |
|---|---|
| Probe lineari e SAE | Activation della policy ufficiale con il ground truth, in tutte le situazioni. |
| Riconoscitore | Osservazioni degli ultimi secondi con l'etichetta della situazione vera (tipo e difficoltà del terreno, attrito, carico), letta dal simulatore. |
| Vettori della libreria | Osservazioni raccolte dalle specialiste nelle rispettive situazioni, più quelle raccolte a ogni ciclo DAgger dalla policy ufficiale con il vettore. Per ogni osservazione si salva anche l'azione che la specialista avrebbe scelto. |

Il reinforcement learning serve in tre casi:
- per addestrare o fare il fine-tuning delle specialiste;
- per le baseline addestrate con PPO (controllore in stile CLAE, action residual, policy allievo);
- per il controllore in stile CLAE, che deve vedere l'effetto delle proprie decisioni sui passi successivi. Per questa baseline si scrive un wrapper dell'ambiente in cui l'azione è il vettore delle intensità e la policy frozen produce i comandi ai motori.

**Sul robot reale.** Durante le prove della fase 6 si registra la stessa tabella, salvo il ground truth che non è disponibile. Al suo posto si usano le misure del robot e i video. Serve a verificare due cose: se il riconoscitore riconosce le situazioni reali e se le feature si comportano come in simulazione. Per farlo il grafo ONNX eseguito a bordo deve restituire, oltre all'azione, le activation e le probabilità del riconoscitore: è una modifica da fare all'export.

---

## 5. Piano sperimentale

Il piano copre circa 26 settimane, da ottobre 2026 a fine marzo 2027, ed è allineato con l'arrivo del Microduck a dicembre. Ogni fase ha un prodotto concreto e un criterio per decidere se proseguire. Le durate sono stime da rivedere dopo la fase 0, quando si avranno i tempi reali sulle nostre GPU.

### Metriche usate in tutte le fasi

Ogni metrica si riporta con media, minimo e massimo sui semi casuali.

| Metrica | Cosa misura | Unità | Verso migliore |
|---|---|---|---|
| Cadute | Numero di episodi terminati con una caduta, su 100 prove. | Cadute ogni 100 prove | Più basso |
| Errore di velocità | Differenza media tra velocità richiesta ed effettiva. | m/s | Più basso |
| Potenza media dei motori | Costo energetico della camminata, come coppia per velocità dei giunti sommata sui motori. | W | Più basso |
| Vantaggio recuperato | Quota del miglioramento della specialista sulla policy ufficiale che l'adapter recupera (sezione 3.2, passo 5). | 0 = nessun miglioramento, 1 = come la specialista | Più alto |
| Accuratezza del riconoscitore | Frazione di istanti in cui la situazione stimata è quella vera. | Frazione tra 0 e 1 | Più alto |
| Tempo di riconoscimento | Tempo tra un cambio di situazione e la sua identificazione. | s | Più basso |
| Degrado sul piano | Differenza di prestazioni sul piano tra il sistema completo e la policy ufficiale da sola. | Stessa unità della metrica di prestazione | Più vicino a zero |
| Costo di una situazione nuova | Ore di GPU per aggiungere una situazione: specialista più adapter per il nostro sistema, nuovo training per la policy allievo. | Ore di GPU | Più basso |

### Fase 0: setup e riproduzione (settimane 1-2)

**Obiettivo:** avere lo stack funzionante e verificare che il problema esista davvero nelle nostre mani.
1. Installare microduck_rl (sezione 4.4) su un nodo con RTX 3090.
2. Eseguire `alpha_walking.onnx` e `rough-walk-g` nel simulatore su terreno piano e irregolare. Riprodurre il confronto riportato nella [scheda della policy `rough-walk-g`](https://huggingface.co/RemiFabre/microduck-rough-walk-g): cadute su 110 prove e potenza media dei motori.
3. Convertire i pesi di `alpha_walking` da ONNX a PyTorch e verificare che, sugli stessi input, le uscite delle due versioni coincidano (sezione 4.5).
4. Fare un breve fine-tuning di `alpha_walking` su una sola situazione, la discesa, e misurarne il tempo sulla RTX 3090 e sulla RTX 5090 di Moro43. È la stima del costo di una specialista.

**Prodotto:** script riproducibili e una tabella con cadute, potenza e tempi.
**Criterio per proseguire:** il divario tra le due policy sul terreno irregolare è riprodotto.

### Fase 1: benchmark di situazioni (settimane 3-5)

**Obiettivo:** le situazioni del benchmark, sia fisse sia con cambi *durante* la camminata. Come costruirle è spiegato nelle sezioni 4.6 e 4.7.

Situazioni:
- piano (riferimento);
- salita e discesa, con il generatore di pendenze e il terreno con rampa di `slope_terrain.py`, anche oltre il massimo visto nel training;
- gradini bassi e terreno irregolare;
- pavimento scivoloso, con attrito sotto il minimo del training;
- carico aggiunto oltre il ±5% del training;
- vibrazioni, opzionale se richiede troppo lavoro.

Due tipi di episodi:
- **a condizione fissa:** il robot resta per tutto l'episodio nella stessa situazione, per esempio una salita continua o un pavimento sempre scivoloso;
- **con transizioni:** la situazione cambia durante l'episodio, per esempio piano, poi discesa, poi piano; oppure il pavimento diventa scivoloso o compare un carico.

Metriche registrate per ogni episodio: cadute, errore tra velocità richiesta ed effettiva, potenza media dei motori.

**Prodotto:** suite di valutazione e prima tabella di risultati per `alpha_walking` e `rough-walk-g`.
**Criterio per proseguire:** almeno tre situazioni in cui la policy ufficiale peggiora in modo misurabile. Solo queste entrano nella libreria.

### Fase 2: policy specialiste (settimane 5-7)

**Obiettivo:** una specialista per ogni situazione selezionata.
1. Fine-tuning di `alpha_walking` con PPO in ciascuna situazione. Se il fine-tuning non basta, training da zero con microduck_rl.
2. Valutare ogni specialista nella propria situazione e sul piano.
3. Registrare i dati (sezioni 4.8 e 4.9): rollout della policy ufficiale in tutte le situazioni per probe, SAE e riconoscitore; rollout di ciascuna specialista nella propria situazione per la distillazione.

**Prodotto:** specialiste e dataset.
**Criterio per proseguire:** ogni specialista è nettamente migliore della policy ufficiale nella propria situazione. Se per una situazione non lo è, la situazione viene tolta dalla libreria: non c'è un comportamento migliore da distillare.

### Fase 3: libreria di correzioni (settimane 7-11)

**Obiettivo:** una correzione verificata per ogni situazione, nelle quattro forme da confrontare.
1. **Addestrare il SAE** sulle activation registrate (sezione 3.6) e verificare che la policy, con l'activation sostituita dalla sua ricostruzione e senza alcuna modifica, cammini come la policy originale. Dare un nome agli elementi del dictionary confrontandoli con le grandezze del simulatore.
2. **Distillare il vettore costante** di ogni specialista (sezione 3.2), su ciascuno dei tre hidden layer e su tutti e tre insieme, con il ciclo DAgger. Si sceglie la configurazione che imita meglio (sezione 3.1).
3. **Distillare dalle stesse specialiste, con la stessa procedura, le altre forme:** LoRA di rango 1 e piccola rete (sezione 3.3), e la forma guidata dal SAE, che modifica solo pochi elementi del dictionary scelti per ciascuna situazione (sezione 3.6, uso 1).
4. **Misurare quanto ciascuna forma avvicina la policy ufficiale alla specialista:** differenza tra le azioni e prestazioni nella situazione (cadute, errore di velocità, potenza).
5. **Prova di steering** per ogni correzione e limite di sicurezza (sezione 3.5).
6. **Leggere le correzioni** con il SAE (sezione 3.6, uso 2): quali concetti cambiano in ciascuna situazione.

**Prodotto:** la libreria nelle quattro forme, con limiti di sicurezza e lettura in termini di concetti.
**Criterio per proseguire:** per almeno due situazioni almeno una forma recupera una parte sostanziale del vantaggio della specialista sulla policy ufficiale, per esempio metà della riduzione di cadute. Se nessuna ci riesce, la tesi si concentra sul confronto della fase 5, che resta pubblicabile.


### Fase 4: riconoscitore, router e sistema completo (settimane 10-14)

1. Addestrare il riconoscitore sui dati registrati; confrontare finestre da 0,5 a 4 secondi e input diversi (solo osservazioni, osservazioni più feature del SAE).
2. Misurare accuratezza e **tempo di riconoscimento**, cioè quanti secondi passano tra un cambio di situazione e il momento in cui il riconoscitore lo segnala.
3. Valutare il sistema completo (riconoscitore, router e libreria) sugli episodi a condizione fissa, su quelli con transizioni e sulle situazioni combinate, per esempio discesa con un carico, dove il router somma due vettori.
4. Valutare lo stesso sistema con un **riconoscitore perfetto**, che riceve la situazione vera dal simulatore. La differenza tra le due versioni separa gli errori dovuti al riconoscimento da quelli dovuti ai vettori.

**Prodotto:** risultati del sistema completo.

### Fase 5: confronti (settimane 13-17)

Sulla stessa suite e con lo stesso budget di simulazione:
1. **Policy ufficiale da sola.**
2. **Sostituzione tra specialiste:** si esegue la specialista della situazione corrente, sia con il riconoscitore sia con la situazione vera. È il riferimento più forte, ma richiede di tenere e validare una rete completa per situazione.
3. **Controllore in stile CLAE:** un controllore addestrato con PPO che decide a ogni passo le modifiche di poche feature del SAE.
4. **Action residual** addestrato con PPO sulle stesse situazioni. LoRA non compare qui perché è già confrontato nella fase 3, come variante di adapter distillata dalle stesse specialiste.
5. **Policy allievo con memoria:** una nuova policy addestrata a imitare tutte le specialiste, con in ingresso gli ultimi secondi di osservazioni, secondo lo schema maestro-allievo classico della locomozione.

Per ogni metodo si misurano:
- prestazioni nelle situazioni difficili;
- prestazioni sul piano, dove la policy validata non deve peggiorare;
- **costo di aggiungere una situazione nuova:** per la libreria, una specialista più un vettore; per l'allievo, un nuovo training completo.

Se il tempo lo permette, si ripete il sistema su altre policy della comunità (stesso formato di input e output) e su un robot diverso di [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).

**Prodotto:** il confronto completo, contributo principale del paper insieme alla libreria.

### Fase 6: esperimenti sul Microduck reale (da dicembre, in parallelo alle fasi 3-5)

1. Installare il runtime ed eseguire `alpha_walking` sul robot.
2. Esportare in un unico grafo ONNX la policy, il riconoscitore, il router e la libreria; verificare che il calcolatore di bordo regga i 50 Hz.
3. Prove con e senza sistema: rampa in legno in salita e in discesa, passaggio da pavimento liscio a tappeto, camminata con e senza un peso fissato al corpo. Robot assicurato a un supporto; vettori sempre entro i limiti di sicurezza.
4. Registrare video, cadute e, istante per istante, la situazione stimata dal riconoscitore e il vettore applicato.

**Prodotto:** video e tabella qualitativa.

### Fase 7: scrittura (settimane 20-26)

Prima la stesura del paper, poi la tesi. Il paper presenta come contributo principale l'adattamento in anello chiuso alla situazione in cui si trova il robot, fissa o variabile durante la camminata (riconoscitore, router e libreria di adapter distillati), seguito dal confronto della fase 5 e dal confronto tra tipi di adapter; gli esperimenti reali fanno da supporto. Target realistici: CoRL o ICRA; ICLR o NeurIPS se il confronto produce indicazioni generali e nette.

### Riepilogo

| Fase | Settimane | Prodotto | Decisione |
|---|---|---|---|
| 0. Setup e riproduzione | 1-2 | Stack funzionante, tempi reali | Il problema è riprodotto? |
| 1. Benchmark di situazioni | 3-5 | Suite di situazioni e metriche | Almeno tre situazioni critiche? |
| 2. Specialiste | 5-7 | Una specialista per situazione, dataset | Ogni specialista è migliore della policy ufficiale? |
| 3. Libreria di correzioni | 7-11 | Correzioni distillate nelle quattro forme, con limiti di sicurezza | Almeno due situazioni con metà del vantaggio recuperato? |
| 4. Riconoscitore e router | 10-14 | Sistema completo | Funziona a condizione fissa e sulle transizioni? |
| 5. Confronti | 13-17 | Confronto completo | Contributo principale pronto |
| 6. Robot reale | da dicembre | Video e misure qualitative | Rafforzamento del paper |
| 7. Scrittura | 20-26 | Paper e tesi | Sottomissione |

**Calcolo previsto.** Fasi 0-1: una RTX 3090. Fase 2: specialiste in parallelo sui 4 nodi di Faretra e su Moro43. Fasi 3-4: una RTX 3090. Fase 5: i 4 nodi di Faretra in parallelo, uno per metodo o per seme. Le ore esatte si fissano dopo la fase 0.

---

## 6. Rassegna della letteratura

Nella ricerca svolta non è emerso nessun lavoro che applichi SAE o steering a una policy di **locomozione** con le gambe. I lavori esistenti si dividono in cinque gruppi:
1. steering delle activation di policy robotiche addestrate con reinforcement learning, il gruppo più vicino a noi;
2. SAE e steering sulle VLA, cioè le policy visione-linguaggio-azione per la manipolazione;
3. probe e analisi delle rappresentazioni interne di policy di locomozione;
4. adattamento di policy di locomozione frozen senza passare dalle activation;
5. adapter e loro composizione negli LLM, gli antecedenti diretti della distillazione in vettori.

Per ogni lavoro: sede di pubblicazione, link al paper e al codice, una spiegazione semplice, perché ci è utile e in cosa si differenzia dalla tesi. "Codice non trovato" significa che il paper e le pagine consultate non indicano un repository pubblico.

### 6.1 Steering delle activation di policy robotiche

**CLAE: Steering Multirobot Behavior via Closed-Loop Affine Activation Editing**
- **Sede:** preprint arXiv, giugno 2026 (University of Southern California e NVIDIA). [Paper](https://arxiv.org/abs/2606.11489) · [Pagina del progetto](https://clae-robotics.github.io/) · Codice non indicato nel paper.
- **In parole semplici:** una squadra di droni sa già volare verso un obiettivo evitando ostacoli. Senza riaddestrare la loro policy, gli autori insegnano ai droni nuovi comportamenti: seguire un profilo di velocità, restare in formazione, evitare le telecamere. Addestrano un SAE sulle activation della policy, scelgono poche feature legate al comportamento voluto e addestrano con PPO un piccolo controllore che, a ogni passo, moltiplica e sposta il valore di quelle feature entro limiti fissati. Funziona anche su droni reali Crazyflie, con tutto il calcolo a bordo.
- **Perché ci è utile:** è il termine di confronto più vicino: lo stesso obiettivo (modificare una policy frozen tramite le activation, istante per istante) ottenuto con un controllore addestrato con reinforcement learning invece che con riconoscitore e libreria di vettori. Fornisce anche indicazioni pratiche: per esempio, le modifiche vanno prodotte con una distribuzione limitata, perché le alternative non limitate rendono il training instabile.
- **Cosa fa di diverso:** navigazione di droni, non locomozione; obiettivo di aggiungere comportamenti nuovi, non di adattarsi a condizioni fisiche che cambiano; nessun confronto con LoRA né con la sostituzione tra policy specializzate.

**LAE: Latent Activation Editing for Safer Multirobot Navigation**
- **Sede:** preprint arXiv, settembre 2025 (stesso gruppo di CLAE). [Paper](https://arxiv.org/abs/2509.20623) · [Pagina del progetto con video e codice](https://lae-robotics.github.io/).
- **In parole semplici:** un piccolo classificatore legge le activation della policy dei droni e riconosce gli istanti in cui una collisione è vicina. In quei momenti un secondo modulo sostituisce l'activation con una versione "più preoccupata del rischio", e la policy reagisce prima. Le collisioni calano molto, senza riaddestrare nulla.
- **Perché ci è utile:** ha la stessa struttura della nostra architettura, "riconoscere la situazione e poi modificare le activation", ma per un solo caso. Il codice, se disponibile come indicato, mostra come agganciarsi a una policy RL frozen.
- **Cosa fa di diverso:** una sola situazione (collisione imminente) e una modifica non leggibile, ottenuta con un modello che predice activation future; nessuna libreria di vettori distillati da specialiste; nessuna locomozione.

### 6.2 SAE e steering sulle VLA

**Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models**
- **Sede:** CoRL 2026 (Stanford). [Paper](https://arxiv.org/abs/2603.19183) · [Codice](https://github.com/swannaiden/drvla), annunciato per il rilascio il 1° ottobre 2026 (da verificare).
- **In parole semplici:** gli autori addestrano SAE sulle activation di due VLA e trovano feature che corrispondono a movimenti elementari (alzare il braccio, chiudere la pinza), all'avanzamento del compito e a concetti del linguaggio. Amplificando una feature durante l'esecuzione il robot compie l'azione corrispondente, in simulazione (LIBERO) e su un robot reale.
- **Perché ci è utile:** mostra la procedura completa per addestrare un SAE su una policy robotica e validare le feature con lo steering. Il codice sarà un riferimento diretto per la fase 2.
- **Cosa fa di diverso:** modelli da miliardi di parametri; steering con intensità fissata a mano; manipolazione.

**Mechanistic Interpretability for Steering Vision-Language-Action Models**
- **Sede:** CoRL 2025 (UC Berkeley). [Paper](https://arxiv.org/abs/2509.00328) · [Codice](https://github.com/vla-mech-interp/mechanistic-steering-vlas).
- **In parole semplici:** dentro una VLA gli autori trovano neuroni legati a parole come "lento", "su", "attento". Amplificandoli il robot si muove più lentamente, più in alto o con più cautela, in tempo reale e senza alcun training aggiuntivo.
- **Perché ci è utile:** è il primo lavoro pubblicato in una conferenza principale a mostrare che lo steering di policy robotiche funziona anche su robot reali; il codice è disponibile.
- **Cosa fa di diverso:** sfrutta il vocabolario del modello linguistico dentro la VLA, che una policy di locomozione non ha; steering fisso.

**Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies**
- **Sede:** preprint arXiv, maggio 2026. [Paper](https://arxiv.org/html/2605.17204) · Codice non trovato.
- **In parole semplici:** gli autori legano le feature di un SAE agli eventi dei rollout (presa, rilascio, fallimento) e provano lo steering su due VLA. Su OpenVLA singole feature cambiano il comportamento; su π0.5 lo stesso metodo ha effetti minimi su una parte del modello, mentre un'altra parte smette di funzionare con quasi qualunque modifica.
- **Perché ci è utile:** è la prova più chiara che lo steering può essere fragile e dipendere molto dal modello. Motiva i nostri test di sicurezza e di intensità massima prima di usare un vettore sul robot.
- **Cosa fa di diverso:** manipolazione; nessun controllore in anello chiuso.

**Observing and Controlling Features in Vision-Language-Action Models**
- **Sede:** preprint arXiv, marzo 2026 (Stanford). [Paper](https://arxiv.org/abs/2603.05487) · Codice non verificato.
- **In parole semplici:** si addestra un probe lineare per una grandezza (per esempio la velocità del braccio) e, a ogni passo, si calcola la modifica più piccola dell'activation che porta il probe al valore desiderato.
- **Perché ci è utile:** è un metodo di steering che cambia istante per istante senza addestrare un controllore e senza SAE. È una baseline economica e forte, già usata da CLAE.
- **Cosa fa di diverso:** la modifica è calcolata con una formula, non appresa dai risultati; manipolazione.

### 6.3 Probe e analisi di policy di locomozione

**SleepWalking: Privileged Representation Shaping for End-to-End Blind Locomotion**
- **Sede:** preprint arXiv, agosto 2026. [Paper](https://arxiv.org/abs/2608.30883) · Codice non trovato.
- **In parole semplici:** è soprattutto un metodo di training per far ricordare a una policy cieca le informazioni utili. Come analisi, gli autori congelano la policy e addestrano probe lineari su più layer, trovando che la velocità del corpo è sempre più leggibile man mano che si sale nella rete (R² da 0,76 a 0,88, dove R² misura quanta variabilità della grandezza vera il probe riproduce; 1 è il massimo).
- **Perché ci è utile:** mostra come impostare i probe su una policy di locomozione (layer, dati di training e di test separati) e quali grandezze aspettarsi.
- **Cosa fa di diverso:** solo lettura delle activation, nessun intervento; policy umanoide e quadrupede con memoria.

**PRISM: Polynomial Representations for Interaction-Structured Motor Control**
- **Sede:** preprint arXiv, luglio 2026. [Paper](https://arxiv.org/abs/2607.23473) · Codice non trovato.
- **In parole semplici:** propone un'architettura che moltiplica tra loro i segnali in ingresso, perché grandezze come la potenza o lo scivolamento sono prodotti di segnali. Con probe lineari verifica che queste grandezze diventano più leggibili nelle activation.
- **Perché ci è utile:** è un esempio di probe su grandezze meccaniche (scivolamento, potenza) che ci interessano per i pavimenti scivolosi e i carichi.
- **Cosa fa di diverso:** propone una nuova architettura, non analizza né modifica una policy esistente.

**Mind the Phase: Effective Rank and Representation Health in Legged Locomotion**
- **Sede:** preprint arXiv, settembre 2026. [Paper](https://arxiv.org/html/2609.06958) · [Pagina del progetto](https://felipe-tommaselli.github.io/MindThePhase/).
- **In parole semplici:** misura quante "direzioni indipendenti" usa la policy per decidere l'azione, separando i momenti in cui il piede è a terra da quelli in cui è in aria. Trova differenze tra architetture che le misure medie nascondono; negli MLP semplici la differenza è assente.
- **Perché ci è utile:** suggerisce di analizzare le activation separando le fasi del passo, e conferma che gli MLP semplici come quello del Microduck hanno una struttura interna più povera.
- **Cosa fa di diverso:** misura globale della rete, nessuna feature leggibile, nessun intervento.

**Neural dynamics of robust legged robots**
- **Sede:** Frontiers in Robotics and AI, 2024. [Paper](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1324404/full) · Codice non trovato.
- **In parole semplici:** studia policy ricorrenti per quadrupedi e, spegnendo gruppi di neuroni, individua quelli indispensabili per camminare in modo robusto.
- **Perché ci è utile:** è il primo esempio di intervento causale dentro una policy di locomozione.
- **Cosa fa di diverso:** lavora su singoli neuroni, nessuno steering, nessun adattamento.

**Interpreting Neural Policies with Disentangled Tree Representations**
- **Sede:** preprint arXiv, 2022, rivisto nel 2023 (MIT); sede di pubblicazione non verificata. [Paper](https://arxiv.org/abs/2210.06650) · Codice non trovato.
- **In parole semplici:** estrae alberi di decisione che spiegano quando si attivano i neuroni di piccole policy di controllo, e trova neuroni allineati con le fasi della camminata.
- **Perché ci è utile:** mostra che anche policy piccole possono avere neuroni leggibili, utile come termine di confronto per il SAE.
- **Cosa fa di diverso:** analisi descrittiva su ambienti semplificati, nessun intervento.

### 6.4 Adattamento di policy di locomozione frozen senza steering delle activation

**SLowRL: Safe Low-Rank Adaptation Reinforcement Learning for Locomotion**
- **Sede:** preprint arXiv, marzo 2026. [Paper](https://arxiv.org/abs/2603.17092) · Codice non trovato.
- **In parole semplici:** per correggere le differenze tra simulazione e robot reale, aggiunge ai pesi di una policy per il quadrupede Unitree Go2 un aggiornamento LoRA e lo addestra direttamente sul robot. Una seconda policy di recupero interviene quando il robot rischia di cadere. Basta un aggiornamento di rango 1.
- **Perché ci è utile:** è la baseline "sui pesi" del nostro confronto, ed è già dimostrata sulla locomozione reale.
- **Cosa fa di diverso:** modifica i pesi una volta per tutte; non si adatta a condizioni che cambiano durante la camminata.

**SteerGenPO: Steering Generative Reinforcement Learning into Stable Robotic Controller**
- **Sede:** preprint arXiv, giugno 2026. [Paper](https://arxiv.org/abs/2606.16572) · Codice non trovato.
- **In parole semplici:** una policy generativa per la camminata di un umanoide Unitree G1 produce l'azione a partire da un vettore casuale. Gli autori addestrano con PPO un modulo che sceglie quel vettore in base allo stato del robot, rendendo la camminata più stabile.
- **Perché ci è utile:** conferma che lo steering appreso funziona sulla locomozione umanoide.
- **Cosa fa di diverso:** richiede una policy generativa, mentre quella del Microduck è un MLP deterministico; non studia condizioni che cambiano.

### 6.5 Adapter e loro composizione in NLP

Il vettore distillato (sezione 3.2) è un caso di adapter, e l'idea di combinare più adapter scelti a seconda del compito esiste già per gli LLM. Questi lavori non riguardano la robotica, ma vanno citati e chiariscono cosa è nuovo nella tesi.

**BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models**
- **Sede:** ACL 2022. [Paper](https://arxiv.org/abs/2106.10199).
- **In parole semplici:** per adattare un modello linguistico a un compito, modifica solo i bias dei layer, lasciando fissi tutti gli altri pesi, e ottiene risultati vicini al fine-tuning completo.
- **Perché ci è utile:** il nostro vettore costante è matematicamente un BitFit applicato a un solo layer. Va citato come antecedente diretto.
- **Cosa fa di diverso:** un solo adattamento per compito, addestrato sui dati del compito e non distillato da una specialista; nessun riconoscitore; testo, non controllo.

**LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition**
- **Sede:** COLM 2024. [Paper](https://arxiv.org/abs/2307.13269) · [Codice](https://github.com/sail-sg/lorahub).
- **In parole semplici:** si addestrano molti LoRA, ciascuno per un compito diverso. Per un compito nuovo, con pochi esempi, si trova automaticamente la combinazione pesata di LoRA che funziona meglio.
- **Perché ci è utile:** è l'equivalente NLP della nostra "libreria più router": più adapter specializzati combinati con pesi.
- **Cosa fa di diverso:** i pesi della combinazione si scelgono una volta per compito, a partire da esempi. Nel nostro caso cambiano a ogni passo di controllo, in base a un riconoscitore della situazione fisica. Gli adapter sono distillati da policy specialiste invece che addestrati sui dati del compito.

**Cosa è nuovo nella tesi, alla luce di questi lavori.** Non l'idea di adapter, né quella di combinarne più d'uno. È nuovo applicarle al controllo di un robot che cammina, dove:
- la combinazione cambia 50 volte al secondo secondo le condizioni fisiche riconosciute;
- gli adapter sono distillati da policy specialiste addestrate in simulazione;
- il risultato si verifica su un robot reale.

È nuovo anche il confronto tra tipi di adapter (bias, LoRA, piccola rete) per una policy di locomozione, con la misura di quanto costa la leggibilità.

---

## 7. Possibili sviluppi

Le direzioni seguenti vanno oltre il perimetro della tesi. Sono elencate perché il metodo, se funziona, si estende naturalmente a ciascuna, e possono diventare un secondo paper o una tesi successiva.

**Riconoscere la situazione prima di arrivarci, con informazioni visive.** La policy ufficiale e il riconoscitore usano solo misure interne del robot: si accorgono di una discesa o di un gradino solo dopo i primi passi, e di un ostacolo solo dopo l'urto (sezione 2). Un riconoscitore che riceve anche un'immagine o una mappa delle altezze del terreno davanti al robot potrebbe applicare la correzione giusta **prima** del cambio. La policy resterebbe cieca e invariata: la vista servirebbe solo a scegliere la correzione. Il simulatore lo permette già: mjlab fornisce di default una scansione delle altezze del terreno, che microduck_rl rimuove dagli input della policy (sezione 2) ma che si può dare al riconoscitore. Sul robot reale serve un sensore adatto, da verificare sulla configurazione del Microduck.

**Applicare lo stesso schema a una VLA.** Le VLA sono policy di manipolazione che ricevono immagini e un'istruzione in linguaggio naturale. Anche loro sono addestrate su un insieme ristretto di situazioni e falliscono quando la scena cambia. Lo schema della tesi si trasferisce direttamente: una specialista per situazione (per esempio una VLA affinata su oggetti scivolosi o su una nuova telecamera), un vettore distillato per ciascuna, un riconoscitore che sceglie quale applicare. Le differenze sono due. Le VLA hanno da centinaia di milioni a miliardi di parametri, quindi activation molto più grandi: conviene partire da modelli piccoli, come SmolVLA, già usata da Matteo nel progetto d'esame su LeRobot, o TurboVLA da 0,2 miliardi di parametri. Inoltre lo steering di VLA è un'area molto attiva (sezione 6.2), quindi il contributo andrebbe posizionato con attenzione.

**Un secondo robot, solo in simulazione.** Per mostrare che il metodo non dipende dal Microduck, lo si può ripetere su un robot con le gambe diverso, per esempio un quadrupede o un umanoide tra quelli disponibili in [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (l'elenco esatto va verificato nel repository). Non serve il robot fisico: bastano una policy di camminata pubblica e il simulatore. Questo esperimento è già previsto come estensione opzionale della fase 5; se i risultati sul Microduck sono buoni, diventa il modo più economico per rafforzare il paper.

**Steering su più layer e correzioni che dipendono dallo stato.** La tesi parte da vettori fissi su un solo layer e confronta tre forme di correzione (sezione 3.3). Uno sviluppo naturale è combinare più layer e forme diverse, e studiare come si compongono quando più situazioni sono presenti insieme, per esempio discesa e carico.

**Situazioni scelte da un comando esterno.** Il riconoscitore stima la situazione dalle misure del robot. In alcuni casi però chi controlla il robot la conosce già, per esempio "stai per entrare in una zona bagnata". Il router può ricevere la situazione direttamente da un comando, anche in linguaggio naturale, senza aspettare il riconoscitore. Diventa così un modo semplice per dare istruzioni di alto livello a una policy di camminata che non le prevede.
