# Tesi magistrale in Embodied / Physical AI: aggiornamento delle direzioni e nuova proposta

*Documento di lavoro, 29 settembre 2026. Candidato: Matteo Preda.*

Il documento è organizzato in cinque parti:

1. i vincoli che ogni proposta deve rispettare;
2. le due proposte di tesi individuate il 15 settembre 2026;
3. i lavori recenti che le hanno rese poco originali;
4. le novità appena accettate a NeurIPS 2026, con modelli e hardware usati;
5. una nuova proposta di tesi.

I termini tecnici di dominio (per esempio *layer*, *feature*, *policy*, *training*) restano in inglese, come nella letteratura.

---

## Glossario

I termini seguenti sono usati in tutto il documento sempre con questo significato.

### Robotica

| Termine | Significato |
|---|---|
| **Policy** | Rete neurale che controlla un robot. A ogni istante riceve ciò che il robot misura e decide l'azione successiva, per esempio la posizione a cui portare ciascun motore. |
| **Locomozione** | La capacità di un robot di spostarsi usando le gambe. Richiede di coordinare molti motori per avanzare nella direzione voluta senza cadere, adattandosi a pavimenti diversi (liscio, ruvido, in pendenza) e a spinte improvvise. |
| **Policy di locomozione** | Policy che fa camminare un robot con le gambe. Decide decine di volte al secondo la posizione di ogni motore delle gambe. Di solito è "cieca": non usa telecamere, ma solo misure interne del robot (angoli e velocità dei motori, inclinazione del corpo) e il comando ricevuto, per esempio "cammina in avanti a 0,3 m/s". |
| **Fase del passo** | Camminare è un ciclo che si ripete: ogni piede alterna un periodo in cui è appoggiato a terra e sostiene il peso, e un periodo in cui è sollevato e si sposta in avanti. La fase del passo indica in quale punto di questo ciclo si trova il robot in un dato istante, per esempio "piede sinistro appoggiato, piede destro a metà del movimento in aria". |
| **Rollout** | Un episodio completo in cui la policy controlla il robot, in simulazione o nel mondo reale. |
| **Reward** | Il punteggio che, durante il training con reinforcement learning, dice alla policy quanto sta facendo bene. Per la locomozione è una formula scritta a mano, per esempio "premia la velocità corretta e penalizza le cadute". |
| **Domain randomization** | Tecnica di training in simulazione: i parametri fisici (attrito del pavimento, masse, forza dei motori) cambiano casualmente a ogni episodio, così la policy impara a funzionare in condizioni diverse e resiste meglio alle differenze con il mondo reale. |
| **Frozen** | Detto di una policy i cui pesi non vengono più modificati. Si può cambiare il suo comportamento solo dall'esterno, per esempio con lo steering. |
| **Anello chiuso** (closed-loop) | Modo di controllo in cui la decisione viene ricalcolata continuamente a partire da ciò che il sistema sta misurando, e l'effetto della decisione cambia a sua volta le misure successive. Esempio: un termostato alza o abbassa il riscaldamento secondo la temperatura che misura, a differenza di una stufa lasciata a potenza fissa (anello aperto). Per lo steering significa che l'intensità dell'intervento cambia a ogni passo di controllo secondo le condizioni, invece di essere fissata all'inizio. |
| **Action residual** | Correzione appresa che si somma alle azioni prodotte da una policy frozen, senza toccarne le activation né i pesi. |
| **LoRA** (Low-Rank Adaptation) | Tecnica di fine-tuning che modifica i pesi di una rete sommando un aggiornamento descritto da pochissimi parametri ("di rango basso"). Il caso estremo, rango 1, corrisponde a un solo vettore di correzione per ogni matrice di pesi modificata. |
| **Sim-to-real** | Trasferimento di una policy addestrata in simulazione al robot fisico. Il *sim-to-real gap* è il peggioramento che si osserva quando il robot reale si comporta diversamente dal robot simulato. |
| **Sim-to-sim** | Esperimento di controllo che imita il sim-to-real restando in simulazione: la policy viene addestrata con certi parametri fisici e provata con parametri diversi. Poiché i parametri veri sono noti, si può misurare esattamente quale differenza causa il problema. |
| **VLA** (Vision-Language-Action) | Policy che riceve immagini e un'istruzione in linguaggio naturale e produce azioni. Tipicamente riusa un modello visione-linguaggio preaddestrato. |
| **Policy generativa** | Policy che produce l'azione trasformando passo dopo passo un vettore casuale di partenza, come fanno i modelli diffusion e flow matching per le immagini. Cambiando quel vettore di partenza si cambia l'azione prodotta, senza toccare i pesi. La policy del Microduck non è di questo tipo: è un MLP che calcola l'azione direttamente dalle osservazioni. |
| **Critic** | Modello che valuta cosa sta facendo il robot: stima quanto il compito è avanzato, oppure giudica se un'azione o una traiettoria è buona. Può essere usato come reward. |
| **Verifier** | Critic usato per scegliere, tra più azioni candidate proposte da una policy, quella da eseguire. |
| **Monitor** (o *failure detector*) | Sistema che osserva il robot mentre lavora e segnala in tempo reale un problema, per esempio una caduta imminente o un oggetto che sta scivolando dalla pinza. Non controlla il robot: dà l'allarme, e chi lo usa decide se fermarlo, chiedere aiuto a un operatore o far partire una manovra di recupero. |

### Interpretabilità delle reti neurali

| Termine | Significato |
|---|---|
| **Layer** | Uno degli stadi di calcolo in cui è organizzata una rete neurale. L'input attraversa i layer uno dopo l'altro fino all'output. |
| **Activation** | Il vettore di numeri che un layer produce per un certo input. È lo "stato interno" della rete in quel punto. |
| **Feature** | Una direzione nello spazio delle activation a cui si può associare un significato. Per esempio, se l'activation cresce lungo una certa direzione ogni volta che il piede sinistro tocca terra, quella direzione è una feature "contatto del piede sinistro". |
| **Superposition** | Fenomeno per cui una rete rappresenta più concetti di quanti neuroni ha. I concetti vengono mescolati negli stessi neuroni, quindi leggere un neurone alla volta non basta per capire cosa rappresenta la rete. |
| **Probe lineare** | Una regressione lineare (o una classificazione lineare) che prende in input l'activation di un layer e prova a predire una grandezza scelta in anticipo, per esempio la velocità del robot. Si addestra su molti istanti di cui si conosce il valore vero della grandezza; in simulazione il valore vero si legge dal simulatore. Se sugli istanti mai visti la predizione è accurata, la grandezza è contenuta nell'activation in una forma facile da leggere. È un metodo *supervisionato*: bisogna decidere prima cosa cercare e disporre del valore vero, e risponde solo alla domanda posta ("la velocità è qui?"). Non rivela grandezze a cui non si è pensato e non dice se la rete usa davvero quella informazione. Esempio numerico nel riquadro sotto la tabella. |
| **Dictionary** | Elenco di direzioni imparato da un SAE (voce successiva) per un certo layer. Ogni direzione, detta *elemento* del dictionary, è un vettore con tanti numeri quanti sono i neuroni del layer e rappresenta un possibile componente dello stato interno della rete, per esempio "piede sinistro appoggiato" oppure "robot veloce". L'idea è che ogni activation si possa descrivere come somma di pochi elementi del dictionary, ciascuno moltiplicato per un peso che indica quanto quel componente è presente in quell'istante. Il dictionary contiene di solito più elementi di quanti neuroni ha il layer, perché per la superposition la rete rappresenta più concetti che neuroni. Il significato degli elementi non è noto quando vengono imparati: si scopre dopo, osservando in quali situazioni ciascun elemento ha peso alto. Esempio numerico nel riquadro sotto la tabella. |
| **SAE** (Sparse Autoencoder) | Piccola rete neurale che si applica a un layer di un modello già addestrato, senza modificarlo, e che impara da sola un dictionary per quel layer. Per ogni activation calcola un peso per ciascun elemento del dictionary, con il vincolo che quasi tutti i pesi siano zero ("sparse"), e poi ricostruisce l'activation sommando gli elementi con peso non nullo. Il training non usa etichette: l'unico obiettivo è che le ricostruzioni siano fedeli usando ogni volta pochi elementi. Così facendo, gli elementi tendono a corrispondere ai componenti che la rete combina davvero nel suo stato interno. Rispetto al probe lineare la logica è rovesciata. Il probe riceve una grandezza scelta da noi e ne verifica la presenza. L'SAE non riceve alcuna grandezza, propone da solo i componenti, e il loro significato va poi verificato, per esempio confrontandoli con il ground truth del simulatore. Esempio numerico nel riquadro sotto la tabella. |
| **Steering** | Modifica delle activation durante l'esecuzione, per esempio sommando una feature moltiplicata per un coefficiente, per cambiare il comportamento del modello senza rifarne il training. |
| **Ground truth fisico** | Valore vero di una grandezza fisica (velocità del corpo, piedi appoggiati, attrito del pavimento), letto direttamente dal simulatore a ogni istante. |


### Riquadro: probe lineare, dictionary e SAE con un esempio

Per semplicità immaginiamo che un layer della policy di locomozione abbia solo 4 neuroni. In un certo istante la sua activation è

> **a** = [0,9; 0,2; 0,7; 0,0]

cioè il primo neurone vale 0,9, il secondo 0,2, e così via. Guardare i singoli neuroni non basta a capire cosa la rete "sa" in quell'istante. Per via della superposition, lo stesso neurone può accendersi sia quando il piede sinistro è appoggiato sia quando il robot va veloce. Probe lineare e SAE sono due modi diversi di dare un significato a questi numeri.

**Il probe lineare parte da una domanda precisa.** Supponiamo di voler sapere se il layer contiene la velocità del robot.
1. Si fanno camminare molti robot in simulazione e si registrano, per migliaia di istanti, l'activation del layer e la velocità vera letta dal simulatore.
2. Si cercano quattro pesi, uno per neurone, tali che la somma pesata dei neuroni riproduca la velocità vera. Per esempio: velocità ≈ 0,5 × neurone 1 − 0,2 × neurone 2 + 0,3 × neurone 3 + 0 × neurone 4. È una normale regressione lineare.
3. Si provano i pesi su istanti mai visti. Se la velocità predetta è vicina a quella vera, la velocità "è leggibile" in quel layer.

Il probe risponde sì o no a una domanda scelta da noi. Ha però due limiti. Trova solo ciò che abbiamo deciso di cercare, e ogni grandezza cercata richiede il suo valore vero per ogni istante. Inoltre dice che l'informazione è presente, non che la rete la usa davvero per decidere come muoversi.

**L'SAE parte senza domande.** Non riceve nessun valore vero: vede solo le activation.
1. Durante il training impara un **dictionary**, cioè un elenco di direzioni lunghe quanto l'activation. Nell'esempio sono vettori di 4 numeri. Il dictionary ne contiene più di 4, per esempio 8, chiamati **e1, e2, …, e8**. Sono più dei neuroni proprio perché, per la superposition, la rete rappresenta più concetti che neuroni.
2. Impara anche a riscrivere ogni activation come somma di **pochi** elementi del dictionary, ciascuno con il suo peso, e con peso zero per tutti gli altri. È questo il significato di "sparse". Nell'esempio, se e3 = [0,6; 0,2; 0,4; 0,0] ed e6 = [0,6; 0,0; 0,6; 0,0], allora

   > **a** = 1,0 × e3 + 0,5 × e6

   Infatti 1,0 × e3 = [0,6; 0,2; 0,4; 0,0], 0,5 × e6 = [0,3; 0,0; 0,3; 0,0], e la somma dà esattamente [0,9; 0,2; 0,7; 0,0]. Gli altri sei elementi hanno peso zero.
3. L'unico obiettivo del training è che queste ricostruzioni siano fedeli usando ogni volta pochi elementi. Nessuno dice all'SAE cosa debbano significare gli elementi.
4. Il significato si scopre *dopo*, guardando in quali istanti ogni elemento ha peso alto. Se e3 si accende esattamente quando il piede sinistro è appoggiato, e3 è la feature "piede sinistro appoggiato". Se il peso di e6 cresce con la velocità del robot, e6 è la feature "velocità". Può anche emergere un elemento che nessuno avrebbe pensato di cercare, per esempio uno che si accende sempre qualche decimo di secondo prima di una caduta.

**Le differenze in sintesi.**

| | Probe lineare | SAE |
|---|---|---|
| Serve il valore vero della grandezza? | Sì, per ogni istante di training. | No, usa solo le activation. |
| Domanda a cui risponde | "La grandezza X è leggibile in questo layer?" | "Di quali componenti è fatto lo stato interno della rete?" |
| Cosa produce | Un insieme di pesi per ogni grandezza cercata. | Un dictionary e, per ogni istante, quali elementi sono accesi e con che peso. |
| Può trovare qualcosa di inatteso? | No. | Sì. |
| Rischio principale | Trovare una grandezza che è presente ma che la rete non usa. | Produrre elementi a cui non corrisponde nessun significato chiaro. |


---

## 1. Premessa e vincoli

Ogni proposta deve rispettare tre vincoli.

**Vincolo 1: basso costo computazionale.** Il gruppo dispone delle GPU seguenti.

| Server | GPU | Memoria per GPU |
|---|---|---|
| Faretra | 4× NVIDIA RTX 3090 | 24 GB |
| Moro43 | 1× NVIDIA RTX 5090 | 32 GB |
| Deeplearn2 | 1× RTX 3090 + 1× Titan XP | 24 GB + 12 GB |
| Moro232 | 1× RTX 3090 | 24 GB |
| (nodo dedicato) | 1× RTX A6000 | 48 GB |

Non ci sono GPU da datacenter (A100, H100) né training su più nodi. Il riferimento di scala è un modello come **TurboVLA** ([arXiv 2607.27205](https://arxiv.org/abs/2607.27205)): 0,2 miliardi di parametri, meno di 1 GB di memoria GPU, 32 azioni al secondo su una RTX 4090, 97,7% di successo su LIBERO. Ogni esperimento della tesi deve stare su una singola GPU da 24 GB, al massimo sui 4 nodi di Faretra in parallelo.

**Vincolo 2: alto impatto, con obiettivo di pubblicazione.** La tesi deve portare a un articolo per una conferenza di fascia AAAI, IJCAI, ICLR, NeurIPS o per una venue ACL-like, oppure per una venue di robotica (CoRL, ICRA). Serve una domanda di ricerca che interessi una comunità ampia, non solo un'applicazione.

**Vincolo 3: esperimenti reali qualitativi sul Microduck.** Il gruppo dispone di un Reachy Mini e riceverà a dicembre 2026 un **Microduck**, bipede open-source di Pollen Robotics e Hugging Face. Il robot pesa circa 800 g, è alto circa 25 cm e ha 14 servomotori Dynamixel XL330: 5 per ogni gamba e 4 per collo e testa. Il software ufficiale è rilasciato con licenza Apache-2.0:

- il repository [microduck](https://github.com/pollen-robotics/microduck) contiene il runtime che esegue sul robot le policy già addestrate (camminata, rialzata, raccolta da terra e altre), esportate in formato ONNX, un formato standard per reti neurali;
- il repository [microduck_rl](https://github.com/pollen-robotics/microduck_rl) contiene gli ambienti di training con PPO (Proximal Policy Optimization, l'algoritmo di reinforcement learning più usato per la locomozione) sul simulatore MuJoCo Warp, con domain randomization;
- esiste un [simulatore ufficiale nel browser](https://huggingface.co/spaces/pollen-robotics/microduck-simulator).

Il lavoro principale deve essere pubblicabile anche solo in simulazione. Il robot reale serve per esperimenti qualitativi di conferma, non come prerequisito.

---

## 2. Le due proposte del 15 settembre 2026

**Proposta A: shortcut learning linguistico nelle VLA.** Due lavori recenti, AFP (CoRL 2026) e LIT (settembre 2026), mostrano che le policy robotiche imparano *scorciatoie visive*: si affidano a dettagli della scena che nei dati di training sono correlati al compito ma non ne sono la causa. La proposta era verificare se lo stesso accade sul lato del linguaggio. Una VLA capisce davvero l'istruzione ("prendi la tazza rossa a sinistra del piatto"), oppure associa parole ricorrenti a traiettorie fisse?

Il piano prevedeva tre passi:
1. perturbazioni controllate dell'istruzione (parafrasi, sinonimi, riordino sintattico, negazioni);
2. misura di quanto la policy usa davvero le parole rilevanti;
3. una tecnica di mitigazione.

**Proposta B: un critic per VLA nello stile di RL²-VLA.** RL²-VLA ([arXiv 2607.26991](https://arxiv.org/abs/2607.26991), [codice](https://github.com/rl2-vla/RL2-VLA)) migliora una VLA già addestrata senza modificarla. Il sistema ha tre componenti:
- una piccola policy, addestrata con reinforcement learning sulle rappresentazioni interne della VLA, che corregge le azioni della VLA durante l'esecuzione;
- un monitor, che attiva la correzione solo quando prevede che la VLA stia per fallire;
- un verifier, che sceglie la migliore tra le azioni candidate.

Gli autori stessi indicano che il monitor va migliorato, in particolare nella tempestività dell'allarme, e che il verifier è assunto affidabile senza essere costruito né studiato. La proposta era sviluppare proprio questi pezzi mancanti, cioè un critic che stimi l'avanzamento del compito e segnali il fallimento, sfruttando le competenze del gruppo sui modelli linguistici.

---

## 3. Perché entrambe le proposte vanno abbandonate

### 3.1 Proposta A: diagnosi, benchmark e mitigazione già pubblicati

**LangGap** ([arXiv 2603.00592](https://arxiv.org/abs/2603.00592), febbraio 2026) fa esattamente l'esperimento centrale della proposta A:
- varia il significato dell'istruzione lungo quattro dimensioni tenendo fissa la disposizione degli oggetti sul tavolo;
- mostra che le VLA allo stato dell'arte ignorano in gran parte l'istruzione, con analisi dettagliata su π0.5;
- costruisce un benchmark che obbliga a capire il linguaggio;
- prova una mitigazione con aumento mirato dei dati. Con training su un solo compito il successo passa dallo 0% al 90%; con più compiti insieme si ferma al 28%.

Copre quindi tutti e tre i passi del piano.

A NeurIPS 2026 è stato inoltre accettato un lavoro sull'auditing delle istruzioni delle VLA (sezione 4), che genera automaticamente istruzioni capaci di far fallire π0 e OpenVLA.

A noi resterebbero solo varianti minori di esperimenti già fatti.

### 3.2 Proposta B: ogni contributo che avremmo potuto dare è già stato dato

La proposta B poteva portare tre contributi. Per ciascuno, un lavoro recente lo ha già realizzato, con risorse che non possiamo eguagliare oppure con un risultato che ne riduce l'interesse.

**Contributo 1: un critic basato su un modello visione-linguaggio che stima l'avanzamento del compito.** Lo ha realizzato **SOLE-R1** (accettato a NeurIPS 2026, dettagli nella sezione 4). È un modello da 8 miliardi di parametri, addestrato su 4 milioni di esempi di ragionamento, che batte GPT-5, Gemini-3-Pro e tutti i reward model robotici esistenti. Un nostro critic dovrebbe confrontarsi con questo modello, e non abbiamo né i dati né il calcolo per fare meglio. Il modello è però pubblico e utilizzabile così com'è.

**Contributo 2: un monitor migliore, che risolva il limite di tempestività dichiarato da RL²-VLA.** Lo ha realizzato **Hide-and-Seek** (accettato a NeurIPS 2026, sezione 4). Il suo risultato indebolisce anche l'idea di usare un modello visione-linguaggio come monitor. Un piccolo modello che legge le rappresentazioni interne della VLA supera un monitor basato su Qwen3-VL-8B di 13,1 punti di accuratezza bilanciata ed è oltre 2.000 volte più veloce (0,001 secondi per passo contro 2,343). Per il monitoraggio, le rappresentazioni interne della policy sono una fonte migliore e molto più economica delle immagini giudicate da un modello esterno.

**Contributo 3: studiare la robustezza del critic al modo in cui è scritta l'istruzione.** Era l'angolo che avrebbe distinto un contributo del nostro gruppo, perché sfrutta le competenze NLP. Lo ha realizzato **ROBORMBENCH** ([arXiv 2609.05401](https://arxiv.org/abs/2609.05401), settembre 2026, venue non indicata). Il lavoro riguarda la proposta B, non la A: non studia le policy ma i *reward model*, cioè proprio i critic.
- **Benchmark:** 2.390 traiettorie di robot reali, di 14 tipi diversi, con l'avanzamento vero del compito, e 21.673 parafrasi dell'obiettivo verificate come equivalenti.
- **Risultato principale:** i critic basati su modelli visione-linguaggio danno reward diversi, a volte opposti, alla stessa traiettoria quando l'obiettivo è riformulato. Il problema persiste anche con modelli più grandi o con ragionamento esplicito.
- **Limiti dichiarati:** solo istruzioni in inglese e solo il reward di fine episodio.

A questi si aggiunge **SAIL** (Scaling In-Context Imitation Learning; Sakana AI e Università di Tokyo, IROS 2026, [arXiv 2603.08269](https://arxiv.org/abs/2603.08269)). Il problema che affronta è far eseguire a un robot un compito nuovo senza fare training, partendo solo da poche dimostrazioni riuscite dello stesso compito. Il sistema usa due modelli visione-linguaggio con ruoli diversi, e il secondo è proprio un critic.

1. **Il modello che propone.** Riceve le dimostrazioni riuscite e scrive una traiettoria completa per il robot, cioè la sequenza di posizioni da raggiungere.
2. **La prova in simulazione.** La traiettoria viene eseguita nel simulatore e se ne registra il video.
3. **Il modello che valuta (il critic).** Guarda il video e indica in quale punto il robot ha smesso di avanzare verso l'obiettivo, per esempio "la pinza si chiude troppo presto e l'oggetto non viene preso".
4. **La correzione.** Il modello che propone usa questo giudizio per scrivere una nuova traiettoria. Una ricerca ad albero (Monte Carlo Tree Search, che esplora più varianti e approfondisce quelle più promettenti) ripete i passi 2-4 su molte traiettorie candidate.
5. **L'esecuzione reale.** Solo la traiettoria migliore viene eseguita sul robot fisico.

Su sei compiti di manipolazione in simulazione, passare da 1 a 45 traiettorie candidate fa salire la frequenza con cui si trova una traiettoria riuscita dal 25% al 73%. Perché SAIL riguarda la proposta B. In RL²-VLA la piccola policy di correzione produce, a ogni passo, più azioni alternative, ma il robot può eseguirne una sola: serve quindi un modello che le giudichi e scelga quella da eseguire, cioè il verifier (sezione 2). RL²-VLA usa un verifier già esistente e assume che funzioni bene, senza studiarlo; la proposta B voleva costruirne uno migliore basato su un modello visione-linguaggio. In SAIL il secondo modello svolge lo stesso compito: giudica le traiettorie candidate e determina quale verrà eseguita sul robot. Ed è già un modello visione-linguaggio, valutato sperimentalmente: il salto dal 25% al 73% misura proprio quanto rende giudicare più candidati. Un verifier di questo tipo quindi esiste già, e costruirne un altro non sarebbe un contributo nuovo.

Resterebbero solo domande di dettaglio, per esempio se la fragilità alle parafrasi peggiora davvero una policy quando il critic viene usato come reward per il training. Sono valide, ma della taglia di un articolo da workshop.

---

## 4. News: lavori rilevanti accettati a NeurIPS 2026

NeurIPS 2026 si terrà in tre sedi: Sydney (6-12 dicembre), Atlanta e Parigi (9-13 dicembre). La [lista degli eventi](https://neurips.cc/Downloads/2026) conta 9.127 voci. Quelle rilevanti per questa tesi sono le seguenti.

**SOLE-R1: un critic che basta da solo ad addestrare un robot.** (MIT e RAI Institute, [paper](https://arxiv.org/abs/2603.28730), [pesi](https://huggingface.co/Philip-MIT/SOLE-R1-8B), [sito](https://sole-r1.github.io/))

Il modello guarda il video del robot insieme all'obiettivo scritto in linguaggio naturale. A ogni istante descrive cosa è cambiato e cosa resta da fare, poi restituisce un numero che misura l'avanzamento del compito. Questo numero viene usato come unico reward: il robot parte da movimenti casuali e impara compiti di manipolazione mai visti, senza dimostrazioni né reward scritte a mano. Raggiunge almeno il 50% di successo su 24 dei 40 compiti provati; GPT-5 e Gemini ci riescono rispettivamente su 7 e 5. Funziona su quattro simulatori e su un robot reale.

- **Modello:** fine-tuning di Qwen3-VL-8B-Instruct (8 miliardi di parametri), prima supervisionato e poi con reinforcement learning (GRPO).
- **Dati:** 1,2 milioni di esempi di ragionamento ricavati da 41 mila video di robot, più dati generali di ragionamento spaziale e temporale, per un totale di 10 milioni di immagini e fotogrammi e 4 milioni di esempi di ragionamento.
- **Hardware di training:** riportato in un'appendice che non è stato possibile consultare; in ogni caso fuori portata per noi, vista la quantità di dati.
- **Uso da parte nostra:** i pesi sono pubblici con licenza MIT. Un modello da 8 miliardi di parametri in mezza precisione occupa circa 16 GB, quindi l'inferenza dovrebbe stare su una RTX 3090 o sulla A6000. È una nostra stima, da verificare con video lunghi. Il modello è pensato per la manipolazione con bracci robotici, non per la locomozione.

**Hide-and-Seek: accorgersi in tempo che il robot sta sbagliando.** (University of Wisconsin-Madison e Georgia Tech, [paper](https://arxiv.org/abs/2605.30834))

Un monitor legge le rappresentazioni interne di una VLA e dà l'allarme appena prima che l'errore diventi irreparabile, per esempio quando un oggetto inizia a scivolare dalla pinza. Per il training non servono annotazioni istante per istante: basta sapere se ogni episodio è riuscito o fallito.

- **Modello:** il monitor è una piccola rete ricorrente (LSTM con 256 unità). Le VLA monitorate sono OpenVLA (7 miliardi di parametri), π0 e π0.5 (circa 3 miliardi).
- **Hardware:** training del monitor in circa 5 minuti su una sola RTX A6000 da 48 GB. Il costo maggiore è generare i rollout con la VLA; per LIBERO bastano 210 episodi di training.
- **Esperimenti:** simulatori LIBERO e VLABench, più un braccio reale UFactory xArm 6.
- **Perché interessa a noi:** mostra che le rappresentazioni interne di una policy contengono segnali di fallimento più ricchi delle immagini, e che leggerle costa pochissimo. Il monitor però è una scatola nera: dice che qualcosa sta andando storto, non cosa, e non permette di intervenire.

**Auditing delle istruzioni delle VLA.** ([poster](https://neurips.cc/virtual/2026/poster/155145))

Il titolo del poster coincide quasi con quello del preprint DAERT ([arXiv 2604.05595](https://arxiv.org/abs/2604.05595)), che con ogni probabilità è lo stesso lavoro. Un modello linguistico viene addestrato con reinforcement learning a scrivere istruzioni diverse tra loro, ma con lo stesso significato, che facciano fallire la VLA. Il successo medio delle VLA attaccate scende dal 93,33% al 5,85%.

- **Modello:** il generatore di istruzioni è Qwen3-VL-4B (4 miliardi di parametri); le VLA attaccate sono π0 e OpenVLA, sul simulatore LIBERO.
- **Hardware:** non riportato nelle parti consultate.

**Workshop utili per edizioni future.** L'[8th Robot Learning Workshop](https://neurips.cc/virtual/2026/workshop/137521) e [NeurReps](https://neurreps.org/) (geometria delle rappresentazioni neurali, con apertura esplicita al controllo motorio). Le scadenze 2026 sono chiuse; restano riferimenti per il 2027.

---

## 5. Nuova proposta di tesi

### 5.1 Titolo di lavoro

**Dove intervenire su una policy di locomozione frozen?**

*(EN: Where to Steer a Frozen Locomotion Policy? Adapting to Changing Conditions without Retraining)*

### 5.2 Punto di partenza: cosa contiene davvero la policy ufficiale del Microduck

Prima di formulare la proposta abbiamo verificato il codice pubblico ([microduck_rl](https://github.com/pollen-robotics/microduck_rl), branch `develop`, commit `cfe1c2a` del 29 settembre 2026) e le policy condivise su Hugging Face.

**Quale policy gira sul robot.** La camminata che Pollen Robotics distribuisce con ogni Microduck è `alpha_walking.onnx`. Deriva dal task `Mjlab-Velocity-Flat-MicroDuck`, cioè da un training su pavimento piano ([scheda della policy](https://huggingface.co/tfrere/microduck-move-base-walk)).

**Com'è fatta.** È un MLP con tre layer da 512, 256 e 128 neuroni e circa 198 mila parametri, calcolati dalla configurazione. Decide 50 volte al secondo le posizioni dei 14 servomotori (5 per gamba, 4 per collo e testa) a partire da 61 numeri:
- 48 misure interne: velocità angolare e inclinazione del corpo, posizioni e velocità dei motori, ultima azione;
- 13 comandi: velocità richiesta e posa della testa.

La policy **non ha memoria**: vede solo l'istante corrente, senza la storia dei passi precedenti. Inoltre non riceve la velocità lineare del corpo, che è disponibile solo al critic durante il training.

**Quali condizioni ha visto nel training.** La domain randomization copre:

| Grandezza | Intervallo nel training |
|---|---|
| Attrito dei piedi | da 0,7 a 1,3 volte il valore nominale (restretto rispetto al precedente 0,3-1,2) |
| Massa e inerzia del corpo | ±5% |
| Posizione del baricentro | ±3 mm, portati a ±8 mm durante il training |
| Attrito interno dei motori e inerzia dei rotori | ±10% ciascuno |
| Tensione della batteria | da 6,5 a 8,2 V, fissata all'inizio di ogni episodio, più un calo proporzionale allo sforzo |
| Ritardo dei comandi ai motori | da 3 a 6 passi di controllo (da 60 a 120 ms) |
| Spinte al corpo | variazioni di velocità fino a ±0,3 m/s ogni 3-6 secondi |
| Terreno | solo piano. Esiste una variante "rough" con gradini fino a 1,5 cm, irregolarità fino a 1 cm e pendenze da 1,7° a 5,7°, ma non è la policy distribuita |

**Cosa ne segue.** Una policy senza memoria non può accumulare evidenza nel tempo: non può capire che il pavimento è scivoloso o che la batteria si sta scaricando osservando più passi. Reagisce solo a ciò che vede nell'istante corrente. È quindi improbabile che la policy contenga stime affidabili di attrito o massa costruite nel tempo: la memoria, se serve, va aggiunta dall'esterno.

**Il problema pratico, documentato dalla comunità.** Una variante addestrata su terreno irregolare (`rough-walk-g`) cade 12 volte su 110 prove su terreno accidentato, contro le 32 su 110 della policy distribuita. In cambio consuma il 17% di potenza in più ([scheda della policy](https://huggingface.co/RemiFabre/microduck-rough-walk-g)).

Oggi la scelta è secca: o una camminata efficiente che cade sul terreno difficile, o una camminata robusta che spreca energia sul piano. Il runtime del robot permette di sostituire una policy con un'altra, ma solo a mano e solo tra policy già addestrate. Addestrarne una nuova per ogni condizione non è banale: la guida ufficiale del repository avverte che ogni nuovo training richiede in genere più cicli di correzione della reward, e che i problemi di sim-to-real hanno richiesto settimane di debug ([AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md)).

### 5.3 Il problema

Un robot che cammina fuori dal laboratorio incontra condizioni che cambiano *mentre cammina*:
- passaggio da piastrelle a tappeto o a una zona bagnata;
- salita e discesa, che richiedono passi diversi;
- superfici che vibrano o si muovono (un pavimento di legno sospeso, una piattaforma, un mezzo in movimento);
- una batteria che si scarica nel corso della sessione, rendendo i motori progressivamente più deboli;
- spinte e urti imprevisti.

Le soluzioni oggi disponibili hanno ciascuna un costo:
- **nuovo training con più condizioni e con memoria:** rifare da capo una policy già validata sul robot reale, con il rischio di peggiorare ciò che funzionava;
- **fine-tuning dei pesi**, anche in forma leggera come LoRA: modifica la policy e va rivalidata;
- **sostituzione tra policy specializzate:** discreta, manuale, limitata alle condizioni previste.

Nel 2026 si è affermata un'alternativa: lasciare la policy **frozen**, cioè con i pesi invariati, e imparare solo un piccolo modulo che ne modifica il comportamento durante l'esecuzione. I lavori si distinguono per il **punto in cui il modulo interviene**, e ciascuno lo ha provato in un contesto diverso. Nessuno dei lavori seguenti risulta ancora pubblicato in una conferenza o rivista: sono tutti preprint su arXiv.

- **Sui pesi della policy.** SLowRL ([arXiv, marzo 2026](https://arxiv.org/abs/2603.17092)) corregge il sim-to-real gap di un quadrupede Unitree Go2 aggiungendo ai pesi un aggiornamento LoRA, addestrato direttamente sul robot reale con una policy di recupero che evita le cadute. Mostra che basta un aggiornamento di rango 1 e che in 60 minuti sul robot si migliora più che con il fine-tuning completo. L'adattamento però avviene una volta: il risultato è una nuova policy fissa, non un comportamento che cambia con le condizioni.
- **Sull'input nascosto di una policy generativa.** Le policy generative (diffusion, flow matching) producono l'azione a partire da un vettore casuale di partenza; il modulo sceglie quel vettore invece di estrarlo a caso. SteerGenPO ([arXiv, giugno 2026](https://arxiv.org/pdf/2606.16572)) lo fa per la camminata di un umanoide Unitree G1 in simulazione, migliorando stabilità e inseguimento dei comandi. Principal Steering Subspaces ([arXiv, settembre 2026](https://arxiv.org/abs/2609.33765)) lo fa per la manipolazione, riducendo lo spazio da esplorare a poche direzioni. Questi metodi richiedono una policy generativa: non si applicano a un MLP deterministico come quello del Microduck.
- **Su uno spazio compresso aggiunto alla policy.** ZPRL ([arXiv, maggio 2026](https://arxiv.org/abs/2605.19919)) inserisce durante il training della policy di manipolazione un piccolo "collo di bottiglia" e poi adatta solo quello, su 8 compiti simulati e 4 reali. Richiede di modificare la policy prima del training, quindi non si applica a una policy già distribuita.
- **Sulle activation interne, tramite feature di un SAE.** CLAE ([arXiv, giugno 2026](https://arxiv.org/abs/2606.11489)) è il lavoro più vicino a questa tesi. Addestra un SAE sulle activation di una policy frozen, sceglie poche feature legate ai comportamenti che interessano, e addestra con reinforcement learning un modulo che le modifica a ogni passo, entro limiti fissati, in base allo stato del robot e degli altri robot. Il contesto è la navigazione di più robot insieme, e lo scopo è ottenere comportamenti nuovi (per esempio coordinarsi in gruppo), non reagire a cambiamenti fisici come pendenze o scivolamenti.

**Cosa manca.** Il meccanismo di CLAE, con feature di un SAE modificate in anello chiuso da un modulo addestrato, esiste già: la tesi non lo rivendica come nuovo e lo usa come una delle interfacce da confrontare. Mancano invece tre cose.
1. **Il problema.** Nessuno di questi lavori affronta una policy di locomozione che deve adattarsi *mentre cammina* a condizioni fisiche che cambiano: pavimento, pendenza, vibrazioni, batteria. SLowRL si adatta una volta sola; SteerGenPO migliora la stabilità in generale; CLAE lavora sulla navigazione.
2. **Il confronto.** Ogni lavoro prova la propria interfaccia nel proprio contesto, con pochi confronti. Nessuno mette a confronto, sulla stessa policy e con le stesse condizioni, pesi (LoRA), azioni, activation libere, feature di un SAE, direzioni PCA, sostituzione tra policy e nuovo training. Senza questo confronto, chi deve adattare una policy di locomozione non sa quale strada scegliere.
3. **Il caso reale più comune.** Una policy semplice, deterministica, senza memoria e già distribuita su un robot commerciale, come quella del Microduck. Su di essa i metodi per policy generative e ZPRL non si applicano, e LoRA la modifica.

### 5.4 Idea centrale

Il metodo ha due fasi: la prima si svolge una sola volta, prima di usare il robot; la seconda gira mentre il robot cammina.

**Fase 1, una volta sola: trovare le "manopole".** Si fa camminare la policy frozen in simulazione in tutte le condizioni della sezione 5.7 e se ne registrano le activation. Su queste activation si addestra un SAE. Tra le feature che l'SAE trova se ne scelgono poche, per esempio da 4 a 8, che cambiano il modo di camminare in modo utile e prevedibile. Per ciascuna si verifica:
- a quale grandezza fisica corrisponde, confrontandola con i valori noti del simulatore;
- cosa succede al robot sommandola alle activation con intensità crescenti;
- fino a quale intensità il robot resta stabile. Questo intervallo diventa il limite di sicurezza della feature.

Il risultato è un piccolo insieme di **vettori di steering verificati**. Ciascuno funziona come una manopola: per esempio una che allunga o accorcia il passo, una che alza o abbassa i piedi, una che rende la camminata più rigida o più morbida.

**Fase 2, durante la camminata: girare le manopole al momento giusto.** Un piccolo **controllore di steering** decide, 50 volte al secondo, di quanto girare ciascuna manopola, restando nei limiti di sicurezza. A ogni passo di controllo:
1. guarda la storia recente del robot, per esempio le misure interne degli ultimi due secondi, ed eventualmente il valore attuale delle feature;
2. sceglie un coefficiente per ciascun vettore di steering;
3. somma i vettori, moltiplicati per i coefficienti, alle activation della policy frozen, che poi calcola l'azione come sempre.

Il controllore è una piccola rete neurale con memoria, addestrata con reinforcement learning in simulazione. Lo schema è lo stesso di CLAE (sezione 5.3), applicato qui alla locomozione e a condizioni fisiche che cambiano nel tempo. La reward premia il robot che cammina bene nelle condizioni che cambiano.

**Cosa vuol dire "closed-loop" (anello chiuso).** Il coefficiente non viene deciso una volta all'inizio e poi lasciato fisso. Viene ricalcolato a ogni passo a partire da ciò che il robot sta vivendo in quel momento. L'intervento cambia il modo di camminare, il nuovo modo di camminare cambia ciò che il robot misura, e le nuove misure cambiano la decisione successiva: il ciclo si chiude. È la differenza tra una stufa accesa a potenza fissa e un termostato, che alza o abbassa la potenza continuamente secondo la temperatura che misura.

Un esempio concreto è il robot che scende da una rampa e poi torna in piano:
- **steering fisso (anello aperto):** si decide prima di partire "passo corto" e lo si tiene per tutta la camminata. Va bene in discesa, ma sul piano il robot resta inutilmente lento;
- **steering in anello chiuso:** appena la storia recente mostra che il corpo tende a cadere in avanti, il controllore accorcia il passo; quando la tendenza sparisce, riporta il coefficiente a zero e il robot torna alla camminata normale.

**Monitor o steering?** Le feature trovate nella fase 1 servono soprattutto come manopole, cioè come direzioni lungo cui intervenire. Possono anche servire come segnali da osservare, perché il loro valore attuale fa parte di ciò che il controllore può guardare. La tesi non costruisce però un monitor separato con soglie fissate a mano: è il controllore a imparare quando e quanto intervenire. La versione "monitor più regole" è inclusa come termine di confronto (sezione 5.6): se la feature X supera una soglia, si applica il vettore Y con intensità fissa. Serve a verificare se imparare il controllore porta un vantaggio rispetto a regole scritte a mano.

Rispetto alle alternative della sezione 5.3, l'approccio ha quattro proprietà.
- **La policy validata non cambia.** Se il controllore restituisce coefficienti nulli, il robot si comporta esattamente come prima. Il rischio di peggiorare ciò che funzionava è controllabile.
- **La memoria viene aggiunta dall'esterno.** Una policy senza memoria diventa capace di adattarsi a condizioni che si rivelano solo osservando più passi.
- **Lo spazio di adattamento è piccolo e leggibile.** Il controllore agisce su poche feature, scelte e verificate in anticipo, con intervalli di intensità sicuri misurati in simulazione. Si può sapere in ogni momento cosa sta facendo: "in discesa riduce la lunghezza del passo".
- **Si passa con continuità da un comportamento all'altro.** Invece di scegliere tra la camminata efficiente e quella robusta, il controllore si muove tra le due secondo il terreno.

### 5.5 Stato dell'arte e spazio libero

La tabella riassume i lavori della sezione 5.3 e quelli vicini. La colonna "Pubblicazione" indica dove sono usciti. Tutti sono preprint del 2026 senza una venue dichiarata, segno che l'area è molto attiva ma ancora da consolidare.

| Lavoro | Pubblicazione | Dove interviene | Robot e compito | Perché non copre la tesi |
|---|---|---|---|---|
| [CLAE](https://arxiv.org/abs/2606.11489) | arXiv, giugno 2026 | Feature di un SAE, modificate a ogni passo da un modulo addestrato | Più robot che navigano insieme | Stesso meccanismo, ma per ottenere comportamenti nuovi nella navigazione, non per adattarsi a condizioni fisiche che cambiano; nessun confronto con LoRA o con la sostituzione tra policy. |
| [SteerGenPO](https://arxiv.org/pdf/2606.16572) | arXiv, giugno 2026 | Input nascosto di una policy generativa | Camminata di un umanoide Unitree G1, in simulazione | Richiede una policy generativa; migliora la stabilità in generale, non l'adattamento a condizioni che cambiano. |
| [SLowRL](https://arxiv.org/abs/2603.17092) | arXiv, marzo 2026 | Pesi (LoRA di rango 1) | Trotto e salto di un quadrupede Unitree Go2, robot reale | Adattamento unico al sim-to-real gap; il risultato è una policy fissa e modificata. |
| [ZPRL](https://arxiv.org/abs/2605.19919) | arXiv, maggio 2026 | Spazio compresso aggiunto alla policy durante il training | Manipolazione, 8 compiti simulati e 4 reali | Richiede di modificare la policy prima del training; manipolazione. |
| [Principal Steering Subspaces](https://arxiv.org/abs/2609.33765) | arXiv, settembre 2026 | Poche direzioni dell'input nascosto di una policy generativa | Manipolazione in simulazione; una VLA da 3 miliardi di parametri su umanoide | Richiede una policy generativa; manipolazione. |
| [RL²-VLA](https://arxiv.org/abs/2607.26991) | arXiv, luglio 2026 | Rappresentazioni interne di una VLA, solo quando si prevede un fallimento | Manipolazione | VLA; nessun legame con feature leggibili né con condizioni fisiche. |
| [Latent Activation Editing](https://arxiv.org/abs/2509.20623) | arXiv, 2025 | Activation, modificate quando un classificatore segnala pericolo | Droni che navigano in gruppo | Correzioni non leggibili; nessuna condizione fisica che cambia. |
| [SAE per VLA](https://arxiv.org/abs/2603.19183), [SAE ancorati agli eventi](https://arxiv.org/html/2605.17204) | arXiv, marzo e maggio 2026 | Singole feature di un SAE, intensità fissata a mano | Manipolazione con VLA | Steering fisso, senza modulo che decide a ogni passo; il secondo mostra che l'effetto cambia molto da un modello all'altro. |

**Obiezione da affrontare direttamente.** Rifare il training della policy con memoria e con più condizioni è l'alternativa naturale. Sul Microduck un training completo costa da circa un'ora (piano, 4.000 iterazioni) a circa 5 ore e mezza (terreno irregolare, 8.000 iterazioni) su una RTX 5090, secondo le schede pubblicate dalla comunità ([piano](https://huggingface.co/witcheer/microduck-walk-flat), [irregolare](https://huggingface.co/witcheer/microduck-walk-rough)). Il costo di calcolo quindi non è l'argomento principale. Gli argomenti sono tre:
- la policy distribuita e già validata sul robot reale resta intatta;
- ogni nuova condizione si aggiunge con un modulo piccolo e verificabile, senza rifare tutto;
- l'effetto dell'adattamento è leggibile.

La tesi include comunque la policy riaddestrata come termine di confronto, così il vantaggio o lo svantaggio è misurato e non affermato.

### 5.6 Domande di ricerca

**D1: Qual è lo spazio migliore in cui adattare una policy di locomozione frozen?** A parità di numero di parametri addestrabili e di interazioni con il simulatore, si confrontano cinque interfacce:
- coefficienti su **feature di un SAE**;
- coefficienti su **direzioni PCA** delle activation;
- una **correzione libera delle activation** di un layer, non vincolata a poche direzioni (l'analogo di ResEmb, che per la manipolazione è risultato instabile secondo [ZPRL](https://arxiv.org/abs/2605.19919));
- una **correzione sommata alle azioni** (*action residual*);
- un **aggiornamento dei pesi di rango 1** (LoRA, come in SLowRL).

**D2: L'adattamento in anello chiuso serve davvero rispetto a uno steering fisso?** Nelle condizioni che cambiano durante la camminata si confrontano:
- la policy frozen senza steering;
- steering con coefficienti fissi, scelti per la condizione media (anello aperto);
- monitor più regole: soglie scritte a mano sul valore delle feature che attivano vettori di steering con intensità fissa;
- il controllore in anello chiuso;
- la sostituzione manuale tra policy specializzate (`alpha_walking` e `rough-walk-g`);
- una policy con memoria riaddestrata con le stesse condizioni.

**D3: Lo steering è sicuro e leggibile?** Per ogni interfaccia di D1 si misura:
- quante cadute avvengono durante l'adattamento, non solo alla fine;
- se, vincolando i coefficienti agli intervalli sicuri misurati in simulazione, le cadute diminuiscono senza perdere prestazioni;
- se un osservatore riesce a descrivere cosa fa il controllore guardando solo l'andamento dei coefficienti. Questo è possibile, per costruzione, solo con le interfacce basate su feature.

**D4: Il metodo è generale?** Il controllore si addestra e si valuta su più policy per lo stesso robot, sfruttando il fatto che tutte le policy di camminata del Microduck condividono lo stesso formato di input e output:
- la policy ufficiale;
- le varianti pubbliche della comunità;
- policy addestrate nella tesi con semi e larghezze diverse.

Si ripete poi su almeno un robot diverso tra quelli di [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).

### 5.7 Metodo

**Condizioni dinamiche in simulazione.** Ogni episodio contiene uno o più cambi di condizione *durante* la camminata. Tutti si generano con gli strumenti già presenti in microduck_rl e mjlab, salvo dove indicato.

| Condizione | Come si riproduce | Perché è rilevante |
|---|---|---|
| Transizione di pavimento | L'attrito dei piedi cambia a metà episodio, anche fuori dall'intervallo di training (sotto 0,7). | Passaggio su zona bagnata o su tappeto. |
| Salita e discesa | Terreno a pendenza del generatore di microduck_rl, fino e oltre i 5,7° della variante "rough". | Richiedono lunghezza e ritmo del passo diversi in salita e in discesa. |
| Vibrazioni | Piattaforma che oscilla sotto i piedi con ampiezza e frequenza variabili. Va implementata. | Pavimenti non rigidi, superfici su mezzi in movimento. |
| Batteria che si scarica | La tensione scende in modo continuo durante l'episodio, fino al minimo di 6 V previsto dal modello dei motori. Nel training originale la tensione è fissa per episodio. | È una condizione reale e inevitabile del robot, che la policy distribuita non ha mai visto variare. |
| Carico e spinte | Massa aggiunta al corpo oltre il ±5% del training; spinte più forti di ±0,3 m/s. | Oggetti trasportati, urti. |

**Metriche.** Ciascuna è riportata con media, minimo e massimo sui semi.

| Metrica | Cosa misura | Verso migliore |
|---|---|---|
| Cadute per 100 prove | Robustezza nelle condizioni dinamiche. | Più basso |
| Errore di velocità | Differenza media tra velocità richiesta ed effettiva. | Più basso |
| Potenza media dei motori | Costo energetico, per verificare se si ottiene la robustezza senza pagare il 17% in più. | Più basso |
| Interazioni necessarie | Quanti passi di simulazione servono per addestrare il controllore fino a una prestazione fissata. | Più basso |
| Cadute durante l'adattamento | Sicurezza del processo di adattamento, rilevante per un eventuale adattamento sul robot reale. | Più basso |
| Degrado sul piano | Differenza di prestazioni sul pavimento piano rispetto alla policy frozen, per verificare che l'adattamento non peggiori il caso normale. | Più vicino a zero |

**Costruzione delle feature.** Un SAE viene addestrato sulle activation di un layer della policy, raccolte in tutte le condizioni della tabella. Con il test di corrispondenza, confrontando ogni feature con le grandezze fisiche note dal simulatore, e con curve di risposta a intensità crescenti, si scelgono poche feature da affidare al controllore e se ne misura l'intervallo sicuro. È anche l'occasione per verificare quali grandezze una policy senza memoria rappresenti davvero.

**Addestramento del controllore.** Il controllore si addestra con PPO in simulazione, con la policy frozen. La reward è quella della camminata originale di microduck_rl, così il confronto con la policy riaddestrata è equo.

**Esperimenti sul Microduck** (da dicembre 2026). Il controllore addestrato in simulazione si esegue insieme alla policy ufficiale. Si provano, confrontando con e senza controllore:
- una rampa in legno in salita e in discesa;
- il passaggio da pavimento liscio a tappeto;
- la stessa camminata a batteria carica e a batteria quasi scarica.

Si registrano video, cadute e i coefficienti scelti dal controllore, che mostrano cosa ha fatto il robot e perché. Il robot è assicurato a un supporto.

### 5.8 Contributi attesi

1. **Il primo confronto sistematico tra i punti in cui adattare una policy di locomozione frozen:** pesi (LoRA), azioni, activation libere, feature di un SAE, direzioni PCA, sostituzione tra policy e nuovo training. Tutti sono confrontati sulla stessa policy, con le stesse condizioni che cambiano durante la camminata. Il confronto misura prestazioni, costo, cadute durante l'adattamento e leggibilità.
2. **Un benchmark di condizioni dinamiche per il Microduck:** transizioni di pavimento, pendenze, vibrazioni, scarica della batteria, carichi. È costruito su codice e policy pubblici, quindi riproducibile da chiunque abbia il robot o anche solo il simulatore.
3. **L'estensione alla locomozione dello steering di feature in anello chiuso** (il meccanismo di CLAE): con memoria aggiunta dall'esterno a una policy che non ne ha, e con limiti di sicurezza misurati per ogni feature.
4. **Validazione sul robot reale** su rampa, cambio di pavimento e batteria scarica.

Il contributo 1 è utile qualunque sia l'interfaccia vincente: indica a chi deve adattare una policy di locomozione quale strada scegliere, e a quale costo.

### 5.9 Fattibilità

| Aspetto | Valutazione |
|---|---|
| Calcolo | La policy frozen è piccola (circa 198 mila parametri) e il controllore lo è ancora di più. Il simulatore mjlab gira su GPU con migliaia di ambienti in parallelo; un training completo di camminata richiede circa un'ora su una RTX 5090, come il nodo Moro43. Le RTX 3090 di Faretra permettono di addestrare in parallelo i controllori per le diverse interfacce e i diversi semi. Tempi esatti sulle nostre GPU da misurare nel pilota. |
| Codice | microduck_rl e runtime del robot (Apache-2.0), policy della comunità su Hugging Face con formato comune, MuJoCo Playground, librerie SAE open-source. |
| Dati | Generati in simulazione, senza annotazione umana. |
| Competenze richieste | SAE e steering sono strumenti che il gruppo conosce già dall'NLP. Il training con PPO segue ricette documentate nel repository. |

### 5.10 Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Una policy senza memoria potrebbe avere poche feature utili per lo steering. | D1 confronta lo steering su feature con interfacce che non dipendono da feature leggibili (correzione delle azioni, LoRA). Il risultato è informativo in entrambi i casi. |
| La policy riaddestrata con memoria potrebbe essere nettamente migliore. | È incluso come confronto; il contributo resta sulla possibilità di adattare senza toccare la policy validata e sulla leggibilità. Se il divario è grande, lo si riporta. |
| Il controllore può produrre intensità pericolose sul robot reale. | Coefficienti vincolati agli intervalli sicuri misurati in simulazione; robot assicurato a un supporto. |
| Vibrazioni e scarica continua della batteria non sono nel simulatore attuale. | Sono modifiche circoscritte (una piattaforma mobile, una tensione che varia nel tempo); se richiedessero troppo, si ripiega sulle altre condizioni, già disponibili. |
| Area molto attiva: tutti i lavori della sezione 5.5 sono preprint del 2026, e uno di essi (CLAE) usa già il meccanismo di steering di feature in anello chiuso. | La tesi non rivendica il meccanismo ma il problema (condizioni fisiche che cambiano durante la camminata) e il confronto tra interfacce; controllo mensile di arXiv e aggiornamento della sezione 5.5 prima di ogni sottomissione. |
| Ritardi nella consegna del Microduck. | Tutti i contributi tranne il 4 sono completi in simulazione. |

### 5.11 Primo passo operativo

1. Scaricare `alpha_walking.onnx` e `rough-walk-g`, eseguirli nel simulatore di microduck_rl su una RTX 3090 e riprodurre il confronto di cadute e potenza su terreno irregolare riportato dalla comunità. Questo conferma che il problema esiste nelle nostre mani.
2. Aggiungere al simulatore la scarica continua della batteria e misurare quanto peggiora la policy ufficiale al diminuire della tensione.
3. Addestrare un primo controllore con correzione delle azioni (l'interfaccia più semplice) su una sola condizione, la discesa, per misurare tempi e interazioni necessarie.

Con questi numeri si stabiliscono le milestone e il budget in ore di GPU.

### 5.12 Venue e tempistica

- **Prima sottomissione:** confronto tra interfacce e benchmark dinamico in simulazione su più policy e robot, con i primi esperimenti reali se disponibili. Il contributo è soprattutto empirico, quindi la collocazione più naturale è CoRL o ICRA; ICLR o NeurIPS sono realistiche solo se il confronto produce indicazioni generali e nette (per esempio "sulle policy senza memoria conviene intervenire sulle activation e non sui pesi").
- **Seconda sottomissione:** versione con gli esperimenti sul Microduck, per CoRL, ICRA o come estensione su rivista.
