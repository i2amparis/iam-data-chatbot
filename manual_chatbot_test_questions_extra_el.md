# Extra Manual Test Set για IAM PARIS Chatbot

Συμπληρωματικό αρχείο με αντίστοιχες ερωτήσεις για manual έλεγχο του chatbot.
Οι ερωτήσεις είναι στα Αγγλικά, ενώ οι σημειώσεις/αναμενόμενα είναι στα Ελληνικά.

Για τα follow-up blocks χρησιμοποίησε το ίδιο session/conversation.

## Checklist ποιότητας

- Να καταλαβαίνει σωστά αν ζητάς data, plot, link, model explanation ή clarification.
- Να αναγνωρίζει variable, region, scenario, model και year χωρίς silent λάθος fallback.
- Να μην επινοεί δεδομένα όταν κάτι δεν υπάρχει.
- Να δίνει χρήσιμα links όταν ζητείται documentation ή project navigation.
- Να κρατάει context σε follow-up ερωτήσεις.
- Να επιστρέφει plot όταν ζητάς graph/chart/visualization.

## 1. General IAM και scenario concepts

- [ ] `what problem do integrated assessment models solve?`
- [ ] `how are IAM scenarios different from forecasts?`
- [ ] `why do different IAMs give different results?`
- [ ] `what should I check before comparing model outputs?`
- [ ] `what does a policy scenario mean?`
- [ ] `what is a reference scenario?`
- [ ] `what does decarbonisation mean in these results?`
- [ ] `how should I interpret model uncertainty?`

Αναμενόμενο:
- Conceptual απάντηση, όχι data extraction.
- Προσεκτική διατύπωση για uncertainty και assumptions.
- Χρήσιμες follow-up ερωτήσεις.

## 2. Model explanations

- [ ] `give me an overview of AIM`
- [ ] `what is IMAGE used for?`
- [ ] `explain POLES model`
- [ ] `what kind of model is GEM-E3?`
- [ ] `tell me about COFFEE model`
- [ ] `what is GCAM-PR?`
- [ ] `which model focuses on energy and land systems?`
- [ ] `which models can analyse mitigation pathways?`

Αναμενόμενο:
- Να απαντά με model metadata όπου υπάρχει.
- Αν δεν έχει αρκετές πληροφορίες, να το λέει καθαρά.
- Να δίνει link για model overview/documentation.

## 3. Model assumptions, scope και limitations

- [ ] `what are the main limitations of IMAGE?`
- [ ] `does POLES include energy markets?`
- [ ] `does AIM cover global mitigation pathways?`
- [ ] `does GCAM include land-use change?`
- [ ] `does REMIND represent negative emissions?`
- [ ] `are CCS assumptions fixed across scenarios?`
- [ ] `where can I check assumptions for WITCH?`
- [ ] `which model documentation should I read for MESSAGEix-GLOBIOM?`

Αναμενόμενο:
- Να ξεχωρίζει known facts από scenario-dependent assumptions.
- Να μη φτιάχνει τεχνικές λεπτομέρειες που δεν υπάρχουν.
- Να δίνει σχετικό documentation link.

## 4. Data queries: emissions

- [ ] `greenhouse gas emissions for EU`
- [ ] `CO2 emissions for Japan`
- [ ] `methane emissions for World`
- [ ] `N2O emissions for Brazil`
- [ ] `energy-related CO2 emissions for China`
- [ ] `industrial emissions for India`
- [ ] `land-use CO2 emissions for World`
- [ ] `carbon dioxide removal for EU`
- [ ] `carbon capture and storage for World`
- [ ] `emissions from agriculture for Brazil`

Αναμενόμενο:
- Σωστό emissions variable ή clarification.
- Να μη μετατρέπει όλα τα greenhouse gas queries σε CO2.
- Καθαρή αναφορά αν δεν υπάρχει διαθέσιμο variable.

## 5. Data queries: energy system

- [ ] `renewable electricity generation for World`
- [ ] `solar electricity generation for India`
- [ ] `wind electricity generation for EU`
- [ ] `coal primary energy for China`
- [ ] `gas primary energy for EU`
- [ ] `oil primary energy for World`
- [ ] `nuclear electricity for Japan`
- [ ] `hydro electricity generation for Brazil`
- [ ] `final energy demand in buildings for EU`
- [ ] `final energy demand in industry for China`

Αναμενόμενο:
- Να ξεχωρίζει generation, primary energy και final energy.
- Να μην απαντά με capacity αν ζητείται generation.
- Να ζητά clarification για ambiguous energy queries.

## 6. Data queries: economy, population, prices

- [ ] `GDP MER for World`
- [ ] `GDP PPP for India`
- [ ] `population projection for Brazil`
- [ ] `income for EU`
- [ ] `consumption for China`
- [ ] `investment for World`
- [ ] `carbon price for EU`
- [ ] `electricity price for India`
- [ ] `energy price for China`
- [ ] `trade for World`

Αναμενόμενο:
- Σωστό macroeconomic variable.
- Να μη μπερδεύει GDP MER με GDP PPP.
- Να λέει καθαρά αν το ζητούμενο δεν υπάρχει.

## 7. Data queries: transport, buildings, industry

- [ ] `transport final energy for World`
- [ ] `passenger transport demand for China`
- [ ] `freight transport demand for India`
- [ ] `electric vehicle sales for EU`
- [ ] `building floor area for World`
- [ ] `residential energy demand for China`
- [ ] `commercial building energy demand for EU`
- [ ] `steel production for World`
- [ ] `cement production for India`
- [ ] `chemical industry energy use for China`

Αναμενόμενο:
- Να πιάνει sector-specific intent.
- Να μη γυρίζει generic final energy αν υπάρχει πιο συγκεκριμένο variable.
- Να δίνει επιλογές αν ο όρος είναι ασαφής.

## 8. Data queries: AFOLU και environment

- [ ] `forest land for Brazil`
- [ ] `cropland for World`
- [ ] `pasture area for India`
- [ ] `bioenergy crop area for EU`
- [ ] `food system emissions for World`
- [ ] `agricultural water use for India`
- [ ] `water withdrawal for China`
- [ ] `biodiversity loss indicator for World`
- [ ] `nitrogen pollution from agriculture`
- [ ] `livestock production for Brazil`

Αναμενόμενο:
- Να μένει σε AFOLU/environment scope.
- Να μη μεταφέρει την ερώτηση σε energy/emissions χωρίς λόγο.
- Να εξηγεί data gaps.

## 9. Model-scoped data

- [ ] `show World CO2 emissions from GCAM`
- [ ] `show EU primary energy from IMAGE`
- [ ] `GDP for China using GEM-E3`
- [ ] `electricity generation for India from POLES`
- [ ] `emissions for Japan from AIM`
- [ ] `does COFFEE have final energy data for Brazil?`
- [ ] `what variables are available for IMAGE in EU?`
- [ ] `plot GCAM CO2 emissions for World`
- [ ] `compare IMAGE and GCAM for World emissions`
- [ ] `show data from a model that covers land use`

Αναμενόμενο:
- Να εφαρμόζει model constraint.
- Να μην ανακατεύει models χωρίς να το δηλώνει.
- Αν δεν υπάρχουν timeseries για μοντέλο, να το αναφέρει.

## 10. Years και χρονικά φίλτρα

- [ ] `World CO2 emissions in 2025`
- [ ] `EU GDP in 2040`
- [ ] `China population in 2100`
- [ ] `India final energy between 2030 and 2060`
- [ ] `Brazil emissions after 2040`
- [ ] `Japan electricity generation before 2035`
- [ ] `World primary energy from 2020 to 2050`
- [ ] `EU carbon price in 2030 and 2050`
- [ ] `China methane emissions around 2040`
- [ ] `India GDP by mid-century`

Αναμενόμενο:
- Σωστό filtering ανά year/range.
- Να εξηγεί αν το requested year δεν υπάρχει.
- Να μη χάνει region/variable όταν εφαρμόζει year.

## 11. Scenario tests

- [ ] `World CO2 emissions in baseline scenarios`
- [ ] `EU electricity generation under current policies`
- [ ] `India final energy in net zero scenarios`
- [ ] `China GDP in low ambition scenarios`
- [ ] `Brazil emissions in high ambition mitigation pathways`
- [ ] `Japan primary energy in Paris-aligned scenarios`
- [ ] `compare current policies and baseline for World CO2`
- [ ] `which scenarios are available for EU emissions?`
- [ ] `show scenarios related to NDCs`
- [ ] `show scenarios related to delayed action`

Αναμενόμενο:
- Να κάνει scenario matching ή να ζητά clarification.
- Να μην περνάει scenario σαν model/region.
- Να επιστρέφει filtered scenario info, όχι τεράστια άσχετη λίστα.

## 12. Plot και visualization requests

- [ ] `plot greenhouse gas emissions for EU`
- [ ] `make a chart of solar electricity for India`
- [ ] `visualize carbon price for World`
- [ ] `draw a line chart for China GDP`
- [ ] `graph Brazil population projections`
- [ ] `create a figure for wind electricity in EU`
- [ ] `plot final energy in buildings for China`
- [ ] `chart methane emissions for World`
- [ ] `visualize primary energy by fuel for World`
- [ ] `make a comparison plot for EU and India CO2 emissions`

Αναμενόμενο:
- Να γυρίζει plot/image ή `plot_base64`.
- Να έχει caption με variable, region, scenario, years.
- Να χειρίζεται comparison plots.

## 13. Links και navigation

- [ ] `send me the IAM PARIS homepage`
- [ ] `where is the model documentation page?`
- [ ] `open the scenario explorer`
- [ ] `where can I browse available applications?`
- [ ] `give me the policy catalogue link`
- [ ] `where can I find NDC ASPECTS results?`
- [ ] `where can I find IAM COMPACT outputs?`
- [ ] `link me to transport transformation results`
- [ ] `link me to buildings transformation results`
- [ ] `link me to AFOLU results`
- [ ] `where are industrial transformation results?`
- [ ] `where can I read project methodology?`

Αναμενόμενο:
- Να απαντά με link/navigation.
- Να μην το μετατρέπει σε data query.
- Να δίνει ειδικό link όταν υπάρχει.

## 14. Follow-up block A: region context

1. [ ] `greenhouse gas emissions for World`
2. [ ] `same for EU`
3. [ ] `same for India`
4. [ ] `plot India`

Αναμενόμενο:
- Να κρατά greenhouse gas emissions.
- Να αλλάζει μόνο region.
- Το plot να αφορά India.

## 15. Follow-up block B: variable refinement

1. [ ] `show electricity for China`
2. [ ] `I mean electricity generation from coal`
3. [ ] `same from solar`
4. [ ] `compare coal and solar`

Αναμενόμενο:
- Στο πρώτο να ζητά clarification ή να δίνει επιλογές.
- Μετά να κρατά electricity generation και να αλλάζει μόνο source.
- Να κάνει comparison στο τέλος.

## 16. Follow-up block C: model context

1. [ ] `tell me about IMAGE`
2. [ ] `show its primary energy for World`
3. [ ] `what about EU?`
4. [ ] `make a chart`

Αναμενόμενο:
- Να κρατά model = IMAGE.
- Να αλλάζει World → EU.
- Να κάνει chart για το τελευταίο scope.

## 17. Follow-up block D: scenario context

1. [ ] `World CO2 emissions under current policies`
2. [ ] `same under baseline`
3. [ ] `compare them`
4. [ ] `now show only 2050`

Αναμενόμενο:
- Να κρατά World + CO2.
- Να συγκρίνει scenarios.
- Να εφαρμόζει year filter στο comparison.

## 18. Follow-up block E: link after data

1. [ ] `transport final energy for EU`
2. [ ] `where can I see transport results online?`
3. [ ] `what related questions can I ask?`

Αναμενόμενο:
- Να δίνει transport-related link.
- Να μη χάνει το context του transport.
- Να προτείνει σχετικά next questions.

## 19. Ambiguous και short prompts

- [ ] `solar`
- [ ] `wind`
- [ ] `coal`
- [ ] `GDP`
- [ ] `population`
- [ ] `transport`
- [ ] `water`
- [ ] `agriculture`
- [ ] `models`
- [ ] `documentation`

Αναμενόμενο:
- Να κάνει clarification όπου χρειάζεται.
- Να μην επιλέγει αυθαίρετα variable/region.
- Να προτείνει 2-4 συγκεκριμένες επιλογές.

## 20. Typos και noisy input

- [ ] `grenhouse gas emisions for eruope`
- [ ] `methan emisions wrld`
- [ ] `solr electrcity india`
- [ ] `gdp ppp chna`
- [ ] `populaton brazl 2050`
- [ ] `curent policie co2 eu`
- [ ] `basline primry energy wrld`
- [ ] `plz plot wind eletricity europe`
- [ ] `wher is model documntation`
- [ ] `transport reslts link`

Αναμενόμενο:
- Να αντέχει συνηθισμένα typos.
- Να ζητά clarification όταν το confidence είναι χαμηλό.
- Να μην απαντά με λάθος confident result.

## 21. No-data και impossible requests

- [ ] `CO2 emissions for Gotham`
- [ ] `GDP for Middle Earth`
- [ ] `electricity generation for Atlantis`
- [ ] `population for Mars in 2050`
- [ ] `carbon price for Hogwarts`
- [ ] `rainbow energy for EU`
- [ ] `dragon emissions for World`
- [ ] `plot happiness for China`
- [ ] `show GDP for EU from FakeModel`
- [ ] `compare Atlantis and World emissions`

Αναμενόμενο:
- Καμία επινόηση δεδομένων.
- Σαφές no-data μήνυμα.
- Προτάσεις για valid regions, variables ή models.

## 22. Γρήγορο extra smoke test

- [ ] `what problem do integrated assessment models solve?`
- [ ] `what is IMAGE used for?`
- [ ] `greenhouse gas emissions for EU`
- [ ] `renewable electricity generation for World`
- [ ] `GDP PPP for India`
- [ ] `plot greenhouse gas emissions for EU`
- [ ] `send me the IAM PARIS homepage`
- [ ] `tell me about IMAGE`
- [ ] `show its primary energy for World`
- [ ] `CO2 emissions for Gotham`
