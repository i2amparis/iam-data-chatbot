# Αδυναμίες Chatbot — TODO Βελτιώσεων

Ευρήματα από έλεγχο κώδικα (manager.py, agents.py, query_extractor.py,
link_router.py, fastapi_app.py, query_normalizer.py, simple_plotter.py).
Ταξινομημένα ανά κατηγορία, με προτεραιότητα (🔴 κρίσιμο, 🟠 σημαντικό, 🟡 βελτίωση).

---

## 1. Routing / Διαχείριση queries

- [x] 🔴 **Bug: το `context` παράμετρος επικαλύπτεται στο `_route_single`** (manager.py:1400).
  Το `context = self.clarification_context` σκιάζει το `context` που πέρασε ο caller
  (`last_entities`). Μετά από clarification, το carried context της συνεδρίας χάνεται.
- [x] 🔴 **Καμία υποστήριξη ελληνικών / μη-αγγλικών queries.** Ο `normalize_query_text`
  (query_normalizer.py) αφαιρεί ό,τι δεν είναι `[a-z0-9|.]` — ελληνικό query καταλήγει
  κενό string και πάει τυχαία στο general_qa. Χρειάζεται detection γλώσσας +
  μετάφραση/χειρισμός πριν το routing, ή τουλάχιστον ευγενικό μήνυμα «ρώτησέ με στα αγγλικά».
- [x] 🔴 **`clarification_context` δεν αρχικοποιείται ποτέ στο `__init__`** — παντού
  ελέγχεται με `hasattr`. Εύθραυστο· αρχικοποίησέ το ως `None` στον constructor.
- [x] 🟠 **Το `_looks_like_site_navigation_request` είναι υπερ-ευρύ** (manager.py:24-130).
  Σκέτο `"contact"` ή `"find " + "database"` στέλνει το query σε navigation. Π.χ.
  «find CO2 emissions in the database» κινδυνεύει να πάει σε general_qa αντί για data_query.
  Οι λίστες hardcoded όρων (workspaces, data stories) πρέπει να φορτώνονται από το link catalog,
  όχι να είναι καρφωτές στον router. *(Το hijacking διορθώθηκε με guard· η μεταφορά των λιστών στο catalog εκκρεμεί.)*
- [x] 🟠 **Comparison ⇒ πάντα plot.** Στο `_deterministic_route_decision` κάθε σύγκριση
  («which is higher, solar or wind in 2050?») δρομολογείται σε data_plotting και επιστρέφει
  γράφημα ενώ ο χρήστης ζήτησε αριθμητική/λεκτική απάντηση.
- [x] 🟠 **`"world" in query` ⇒ region=World άνευ όρων** (manager.py:1623). Πιάνει και
  «world's largest», «world-headed workspace», και πατάει πάνω σε ρητά ζητούμενο region.
  Θέλει word-boundary και να μην κάνει override υπάρχον region.
- [x] 🟠 **Το multi-intent split είναι εύθραυστο** (`_split_multi_intent`). Σπάει στο
  " and " — «show solar and wind capacity and plot it» δίνει λάθος τεμαχισμό. Επίσης το
  context μεταξύ sub-queries ξαναεξάγει entities από το sub-query (νέα κλήση LLM) αντί να
  χρησιμοποιεί τα entities της απάντησης. *(Το split διορθώθηκε· η επανεξαγωγή entities μεταξύ sub-queries παραμένει.)*
- [x] 🟠 **Overzealous nulling των entities** (manager.py:1604-1611 και agents.py:277-298).
  Αν το query λέει "capacity" και το variable δεν περιέχει "capacity" ⇒ μηδενίζεται, ακόμα
  κι αν ο extractor βρήκε σωστό συνώνυμο (π.χ. "installed solar" → Capacity|Electricity|Solar
  περνά, αλλά "power generation capacity" → Secondary Energy… κόβεται). Η ίδια λογική
  υπάρχει διπλή σε manager και agents — ενοποίηση σε ένα σημείο.
- [x] 🟠 **Hardcoded «επιδιορθώσεις»** στο `_repair_comparison_entities`: GHG → `Emissions|GHG`
  και wind/solar → συγκεκριμένα variables χωρίς έλεγχο ότι υπάρχουν στα δεδομένα.
  Αν λείπει το variable, ο χρήστης παίρνει "no data" αντί για το σωστό κοντινό match.
- [x] 🟡 **Routing LLM prompt μπερδεμένο** (manager.py:190-224): το system template
  τελειώνει με "Question: {query} Answer:" ΚΑΙ υπάρχει ξεχωριστό Human message
  "Query: {query}" — το query μπαίνει δύο φορές. Καθάρισμα του template.
- [x] 🟡 **Παρωχημένο μοντέλο `gpt-4-turbo` παντού** (router, agents, extractor).
  Ακριβό/αργό/deprecated. Μετάβαση σε νεότερο μοντέλο· για routing/extraction αρκεί μικρό
  fast μοντέλο (μειώνει latency και κόστος ανά query, γίνονται 2-3 LLM κλήσεις/ερώτηση).

## 2. Clarifications & Follow-ups

- [x] 🔴 **Κάθε μήνυμα ≤4 tokens θεωρείται clarification** (`_is_clarification_followup`).
  «thanks», «hello», «cool» ενώ εκκρεμεί clarification θα «απαντηθούν» ως επιλογή
  variable/region — λάθος συμπεριφορά. Χρειάζεται allowlist για greetings/ευχαριστίες.
- [x] 🟠 **Επιλογή option από τυχαίο αριθμό**: το `_extract_option_choice` πιάνει
  οποιονδήποτε αριθμό στο κείμενο («show me 3 scenarios» ⇒ επιλέγει το option 3).
  Να απαιτεί σκέτο αριθμό ή «option N».
- [x] 🟠 **Το clarification context λήγει μετά από 1 turn** (issued_turn+1). Αν ο χρήστης
  ρωτήσει κάτι ενδιάμεσα και μετά απαντήσει στο clarification, έχει χαθεί. Είτε αύξηση
  παραθύρου είτε ρητό μήνυμα ότι η επιλογή έληξε.
- [x] 🟠 **Στενά regex για generic follow-ups** (`_is_generic_followup`): πιάνει «plot it»
  αλλά όχι «plot them both», «make a chart of that», «graph this data». Επέκταση προτύπων
  ή LLM-based follow-up detection.
- [ ] 🟠 **Το state των follow-ups βασίζεται σε regex parsing της απάντησης**
  (`_persist_last_entities` διαβάζει "### Var in Region" από το markdown). Αν αλλάξει το
  format απάντησης, σπάει σιωπηλά το context. Τα agents να επιστρέφουν δομημένα entities,
  όχι μόνο κείμενο.
- [ ] 🟡 **`_maybe_add_followup_guidance` = σωρός από ειδικές περιπτώσεις** (hardcoded
  "solar capacity", "fit-for-55" strings). Ασυνεπές: κάποιες απαντήσεις παίρνουν το
  «Reply with a scenario…», οι περισσότερες όχι. Θέλει ενιαίο κανόνα βάσει του αν η
  απάντηση είναι πλήρης ή όχι.

## 3. Suggested next questions (API)

- [x] 🔴 **Το «Try the closest valid option» προτείνεται ως ερώτηση αλλά ο router δεν το
  καταλαβαίνει.** Αν ο χρήστης το πατήσει, δεν υπάρχει handler για αυτή τη φράση —
  θα πάρει τυχαία απάντηση. Κάθε suggested question πρέπει να έχει εγγυημένο handling
  (round-trip test: κάθε suggestion → έγκυρη διαδρομή).
- [x] 🟠 **Στατικά, context-blind suggestions** (`_suggested_next_questions`,
  fastapi_app.py:583): «Compare with Baseline» προτείνεται ακόμα κι όταν το τρέχον
  scenario ΕΙΝΑΙ το Baseline ή δεν υπάρχει Baseline για το variable· «Show this for 2050»
  ακόμα κι αν ήδη δείχνει το 2050. Να φιλτράρονται με βάση τα διαθέσιμα δεδομένα
  (variable_scenarios / years) και το τρέχον scope.
- [x] 🟡 **«Use the first option»** προστίθεται με regex στο κείμενο — να προστίθεται μόνο
  όταν υπάρχει ενεργό clarification_context με options.

## 4. Links

- [x] 🟠 **Force-append του γενικού /results link** (`_ensure_results_link_for_data_answer`):
  κόβει πραγματικά σχετικά links στα 2 για να χωρέσει το γενικό /results με confidence 0.25.
  Θόρυβος· να μπαίνει μόνο όταν δεν υπάρχουν ≥2 σχετικά links.
- [x] 🟠 **Fallback πάντα /results ακόμα κι όταν είναι άσχετο** (link_router.py:264-278,
  confidence 0.1). Για γενικές ερωτήσεις κλίματος ο χρήστης βλέπει πάντα ένα link που δεν
  σχετίζεται — καλύτερα κανένα link από λάθος link, βάλε ελάχιστο confidence threshold.
- [x] 🟠 **Διπλή εμφάνιση links**: τα links μπαίνουν ΚΑΙ μέσα στο answer text
  («Relevant IAM PARIS links:») ΚΑΙ στο δομημένο `relevant_links` του API. Το frontend
  θα τα δείξει δύο φορές. Διάλεξε ένα κανάλι (προτείνεται μόνο το structured field).
- [x] 🟡 **Dedup μόνο σε (title, url, search_hint)** στο `suggest_links` — ίδιο URL με
  διαφορετικό τίτλο εμφανίζεται 2 φορές. Dedup ανά URL.
- [x] 🟡 **Καμία runtime επαλήθευση URLs.** Υπάρχει validate_links.py αλλά offline —
  να τρέχει περιοδικά (CI/cron) και να σημαδεύει dead links στο catalog ώστε να μην
  προτείνονται.
- [x] 🟡 **Keyword-μόνο scoring** στο `_score_item` με μαγικούς αριθμούς (28/16/12/…)
  και hardcoded project boosts («ndc aspects», «iam compact»). Δύσκολο να συντηρηθεί —
  τα boosts να μπουν ως πεδία στο link catalog.

## 5. Ποιότητα απαντήσεων

- [x] 🔴 **`ModellingSuggestionsAgent` επιστρέφει στατική λίστα** (agents.py:429-446)
  που αγνοεί τελείως την ερώτηση — «suggest studies about transport» παίρνει τις ίδιες
  6 γενικές προτάσεις. (Υπάρχει και νεκρός κώδικας: το replace αντικαθιστά το URL με τον
  εαυτό του.) Να γίνει data-driven από metadata/workspaces ή LLM με grounding.
- [x] 🟠 **`_workspace_result_answer` hardcoded παράκαμψη**: κάθε αποτυχημένο query που
  περιέχει «net zero» απαντιέται με «IAM COMPACT workspace» — μπορεί να είναι παραπλανητικό
  για άσχετες net-zero ερωτήσεις.
- [x] 🟠 **Γενικό error message** («Sorry, I encountered an error… try rephrasing») για
  κάθε exception — ο χρήστης δεν ξέρει αν φταίει η διατύπωση ή αν έπεσε το service.
  Διαχώρισε provider outage (ξαναδοκίμασε αργότερα) από query problem (αναδιατύπωση).
- [x] 🟠 **Διπλή/ασυνεπής μνήμη στο GeneralQAAgent**: έχει δικό του
  `ConversationSummaryBufferMemory` ΚΑΙ δέχεται `chat_history` από το session — τα δύο
  ιστορικά αποκλίνουν. Επίσης το chain χτίζεται στο `__init__` και πετάει exception αν
  λείπει το vector store (κρασάρει όλο το manager initialization).
- [x] 🟡 **`plot_caption` = πρώτη γραμμή του υπόλοιπου κειμένου** (`_split_answer_payload`)
  — hacky, μπορεί να πάρει τυχαίο κείμενο ως λεζάντα. Ο plotter να επιστρέφει caption ρητά.
- [ ] 🟡 **Το `_classify_no_data_reason` μαντεύει από substrings** του κειμένου απάντησης —
  να επιστρέφεται δομημένος λόγος από το data_query pipeline.

## 6. API / Υποδομή

- [x] 🟠 **Βαρύ per-session state**: κάθε νέο session φτιάχνει ολόκληρο MultiAgentManager
  (QueryEntityExtractor lookups πάνω σε όλα τα ts records + GeneralQA chain). Με
  MAX_SESSIONS=500 η μνήμη εκτοξεύεται και το πρώτο query κάθε χρήστη είναι αργό.
  Τα lookups/prompts είναι ίδια για όλους — να μοιράζονται, ανά session να μένει μόνο
  το ελαφρύ state (history, last_entities, clarification_context). *(Ο extractor μοιράζεται πλέον μεταξύ sessions και το GeneralQA chain χτίζεται lazily· τα agents παραμένουν per-session αλλά είναι ελαφριά.)*
- [x] 🟠 **Rate limiting ανά `request.client.host`**: πίσω από proxy/CDN (η iamparis.eu
  σίγουρα έχει) όλοι οι χρήστες μοιράζονται ένα IP bucket ⇒ 429 σε αθώους χρήστες.
  Χρήση X-Forwarded-For (με trusted proxy) ή per-session limiting.
- [x] 🟡 **Το `history` επιστρέφεται ολόκληρο σε κάθε response** — payload που μεγαλώνει
  απεριόριστα σε μακριές συνομιλίες. Επιστροφή μόνο του τελευταίου turn ή cap.
- [x] 🟡 **Deprecated `@app.on_event("startup")`** — μετάβαση σε lifespan handler.
- [x] 🟡 **Monitoring counters in-memory** — μηδενίζονται σε restart και είναι ανά worker
  (με πολλά uvicorn workers τα νούμερα είναι λάθος). *(Persistence προστέθηκε· το multi-worker θέμα και τα in-memory sessions παραμένουν — θέλουν shared store.)* Επίσης τα sessions είναι in-memory:
  με >1 worker το follow-up context σπάει τυχαία (sticky sessions ή shared store).

## 7. Γενικά / Συντήρηση

- [x] 🟡 **Το todo.md ισχυρίζεται «backend functionally complete»** — να ενημερωθεί με τα
  παραπάνω ευρήματα ώστε να μη δίνει ψευδή εικόνα.
- [ ] 🟡 **Υπερβολική εξάρτηση από hardcoded λίστες strings** σε manager/data_utils
  (workspaces, scenarios, variables) — κάθε νέο workspace/dataset στο IAM PARIS απαιτεί
  αλλαγές κώδικα σε πολλά σημεία. Μεταφορά σε configuration/metadata.
- [x] 🟡 **Δεν υπάρχει χειρισμός small talk / out-of-scope**: «hi», «who are you»,
  «what can you do» πάνε στο general_qa LLM χωρίς οδηγία για capabilities-απάντηση.
  Πρόσθεσε intro/help intent που εξηγεί τι μπορεί να απαντήσει το bot.
