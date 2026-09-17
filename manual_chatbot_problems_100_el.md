# Προβλήματα από 100 ερωτήσεις (IAM PARIS Chatbot)

_Εκτέλεση 101 ερωτήσεων (77 singles + 8 follow-up blocks × 3) στις 2026-07-20._
_Κατηγορίες: model descriptions, data, plots, links, follow-ups._
_Links: **63/63 ζωντανά**. Τα περισσότερα data/plot/link queries με καθαρό variable δουλεύουν σωστά._

Παρακάτω μόνο τα **προβλήματα** (τι είπε το bot που είναι λάθος/ελλιπές), ταξινομημένα κατά σοβαρότητα.

---

## 🔴 Υψηλή προτεραιότητα (λάθος/άχρηστη απάντηση σε βασικό query)

### Model descriptions
1. **`what is the AIM model?`** → «### AIM/Enduse India 3.3 Description: **nan** …» — εμφανίζει το literal `nan` (κενό/NaN metadata δεν φιλτράρεται). Πρέπει: παράλειψη του description ή fallback κείμενο.
2. **`tell me about MESSAGEix-GLOBIOM`** → σωστή περιγραφή, αλλά **το model name διέρρευσε στο region**: `region=MESSAGEix-GLOBIOM 1.1`. Το scope είναι λανθασμένο.
3. **`what kind of model is TIAM?`** → route=`data_query` (όχι model_explanation) και **spurious `region=India`** από το «TIAM». Δίνει περιγραφή αλλά με λάθος scope/route.
4. **`is REMIND a general equilibrium model?`** → «I couldn't match that to a known model» — το REMIND **είναι** γνωστό. Το yes/no misrouteάρει και αποτυγχάνει το matching.
5. **`does WITCH model land use?`** → «no metadata description… Examples: **MUSE**, muse» — για ερώτηση WITCH επιστρέφει MUSE. Λάθος/μπερδεμένο.
6. **`what does the E3ME model do?`** → «matching model names but no metadata description available» ενώ αλλού (`describe the E3ME model`) δίνει κανονική περιγραφή. Ασυνέπεια: το `E3ME 6.1` (data name) δεν συνδέεται με το profile description.

### Links (navigation misroute)
7. **`where is the IAM PARIS homepage?`** → «I could not match this request to a reliable IAM PARIS link.» — το homepage (iamparis.eu) πρέπει να resolveάρει.
8. **`link me to the models page`** → route=`data_query` → «I couldn't match that to a known model.» — navigation misroute σε model lookup.
9. **`where are the data stories?`** → route=`data_query` → «I need one more detail. Which variable should I use?» — navigation misroute σε data query.
10. **`where can I download the scenario data?`** → route=`data_query` → «Which variable should I use?» — το download intent πιάστηκε μόνο για «download the data», όχι «download the **scenario** data».

### Plots
11. **`bar chart of carbon price for the World in 2050`** → «could not confidently match… `bar`, `chart`» — το `carbon price` δεν resolveάρει όταν υπάρχει «bar chart» + year. (Σκέτο `carbon price for the EU` δουλεύει.)
12. **`plot coal and gas primary energy for the EU`** → «could not confidently match all of the requested wording» — multi-variable plot (coal + gas) αποτυγχάνει τελείως.
13. **`graph solar electricity for the EU`** & **`visualize nuclear electricity for France`** → βγάζουν **clarification** αντί για plot· δεν προτείνουν το `Secondary Energy|Electricity|Solar/Nuclear` ως πρώτη επιλογή (δίνουν Capacity).

### Data (year filter σπάει το variable)
14. **`carbon price for the World after 2040`** → «could not confidently match… term: `after`» — το «after 2040» (year filter) σπάει το matching του `carbon price`.

### Follow-ups
15. **Block H — `compare solar and wind`** (μετά `solar electricity for India` → `same for wind`) → **κολλάει σε clarification loop**: το «same for wind» εξακολουθεί να δείχνει Solar, και το «compare solar and wind» δεν βγάζει σύγκριση. Ούτε source switch, ούτε comparison.
16. **Block F — `plot the comparison`** (μετά current policies → `compare with baseline`) → κάνει plot αλλά **χάνει τη σύγκριση** (scope `scenario=Baseline`, «across available scenarios» — όχι current-policies-vs-baseline).

---

## 🟡 Μεσαία προτεραιότητα

### Variable specificity / clarification αντί για auto-resolve
17. **`final energy in transport for the World`** → «could not match» ενώ υπάρχει `Final Energy|Transportation`. Sector-specific δεν πιάνεται.
18. **`nuclear capacity for France`** → clarification αντί για `Capacity|Electricity|Nuclear` (θα έπρεπε να είναι σχεδόν μονοσήμαντο).
19. **`electricity price for Germany`** → clarification με πρώτη επιλογή `Price|Electricity|Steel` (περίεργη). Το `Price|Electricity` βασικό θα ήταν καλύτερο.
20. **`steel production for India`** → clarification με πρώτη `Exports|Steel` (αντί `Production|Steel`).
21. **`cropland area for India`** → «could not match» ενώ υπάρχει `Land Cover|Cropland`-τύπου variable.
22. **`emissions intensity of electricity for the EU`** → διάλεξε ένα περίεργο `IDA|Emissions|CO2|…|CO2 intensity` και «no data». 
23. **`gas consumption` / `oil demand` / `hydro electricity`** → clarification (ambiguous). Ανεκτό, αλλά θα βοηθούσε auto-pick του πιο προφανούς.

### Scenario handling
24. **`primary energy for India in a net zero scenario`** → map σε `NZE_Bench_H` και «no data» αντί για graceful «το net-zero scenario δεν έχει data για αυτό» ή relax.
25. **`plot emissions for the World under net zero`** → «could not match… `net`, `zero`» — το «net zero» + «emissions» σπάει το matching.

### Links (λιγότερο ειδικά)
26. **`link me to the transport transformation results`** → γενικό `[Results]` αντί για το transport-specific workspace link (αλλού το `where can I find NDC ASPECTS results` δίνει ειδικό link — ασυνέπεια).
27. **`link me to the model documentation for GCAM`** → γενικό `[Models]` αντί για GCAM-specific σελίδα.

### Follow-ups
28. **Block B — `plot its CO2 emissions`** (μετά `tell me about GCAM`) → κάνει plot αλλά **χάνει region=World & model=GCAM** (scope μόνο variable). Το «its» (GCAM) δεν εφαρμόστηκε ως model φίλτρο.
29. **Block C — `and its coal share`** → «could not match» (το «coal share» δεν γίνεται `Primary Energy|Coal`)· μετά το «compare them on a chart» απλώς plot-άρει Primary Energy (όχι coal-vs-total).

---

## 🟢 Δουλεύουν σωστά (για αναφορά)

- **Data με καθαρό variable:** CO2/CH4/N2O emissions, Kyoto Gases (greenhouse gas), final/primary/secondary energy, solar/wind generation (`Secondary Energy|Electricity|Solar/Wind`), coal primary energy, GDP|MER, Population, carbon price. Regions: World/EU/CHN/IND/JPN/BRA/NGA/DEU/FRA σωστά.
- **Year filters:** in 2030 / in 2050 / in 2100 / between 2030 and 2060 / from 2020 to 2060 / until 2080 ✅.
- **Plots:** CO2 World, GDP India, population Nigeria, primary energy China, final energy Japan (year range), methane World ✅. Two-region: **compare CO2 for China and India** → σωστό vs-plot ✅.
- **Model descriptions:** GCAM, GEM-E3, IMACLIM, DREEM, «difference GCAM vs REMIND» ✅.
- **Model-scoped:** `CO2 from GCAM` (data), `what data does GCAM have` (variable list) ✅.
- **Model switch follow-up (Block E):** GCAM → «now from MESSAGEix» (τίμιο no-data) → «plot both models» (comparison plot, plots GCAM με note) ✅.
- **Region follow-ups (Block A/D):** «same for China», «what about Brazil», «now show 2050 only» ✅.
- **Links (63/63 ζωντανά):** results, application library, policy catalogue, NDC ASPECTS, industrial transformation, contact, raw data application (Aqueduct), scenario explorer ✅.

---

## Μοτίβα-ρίζες (για μελλοντική διόρθωση)

- **A. Variable matching σπάει από «θόρυβο» στο query:** λέξεις όπως `bar`, `chart`, `after`, `net`, `zero`, `share` περνάνε ως «unmatched terms» και μπλοκάρουν ένα κατά τα άλλα καθαρό variable (carbon price, emissions). Το matching πρέπει να αγνοεί plot-verbs / year-prepositions / scenario-λέξεις πριν κρίνει confidence.
- **B. Navigation misroute:** «homepage», «models page», «data stories», «download the scenario data» δεν πιάνονται ως navigation → πέφτουν σε data_query. Χρειάζεται ευρύτερο nav intent (homepage, «X page», «scenario data»).
- **C. Model description metadata:** `nan` descriptions, model-name→region leak, data-only model names (E3ME 6.1) χωρίς σύνδεση με profile, yes/no ερωτήσεις («is REMIND…», «does WITCH…») που misrouteάρουν.
- **D. Follow-up refinement αδύναμο:** possessive «its coal share» / source switch «same for wind» / comparison preservation «plot the comparison» δεν κρατάνε ή δεν εξειδικεύουν σωστά το scope.
- **E. Multi-variable plots:** «coal and gas primary energy» δεν σχηματίζει 2-variable comparison.
