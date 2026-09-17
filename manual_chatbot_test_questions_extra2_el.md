# Extra Manual Test Set 2 για IAM PARIS Chatbot (50 ερωτήσεις)

Δεύτερο συμπληρωματικό αρχείο με 50 νέες ερωτήσεις για manual έλεγχο του chatbot.
Οι ερωτήσεις είναι στα Αγγλικά, οι σημειώσεις/αναμενόμενα στα Ελληνικά.
Καμία ερώτηση δεν επαναλαμβάνεται από τα `manual_test_queries.md` και
`manual_chatbot_test_questions_extra_el.md`.

Κάλυψη: model data, model descriptions, plots, links στο website, follow-up questions.

Για τα follow-up blocks (ενότητες 6Α–6Δ) χρησιμοποίησε το ίδιο session/conversation.

## 1. Model-scoped data (8)

- [ ] `primary energy for World from REMIND`
- [ ] `CO2 emissions for India from MESSAGEix-GLOBIOM`
- [ ] `electricity generation for EU from WITCH`
- [ ] `final energy for China from PROMETHEUS`
- [ ] `does WITCH report carbon price for EU?`
- [ ] `which regions does GCAM cover?`
- [ ] `what years are available for GEM-E3 data?`
- [ ] `list the variables reported by POLES`

Αναμενόμενο:
- Να εφαρμόζει σωστά το model constraint (όχι IMAGE→MANAGE τύπου λάθη).
- Αν το μοντέλο δεν έχει timeseries, καθαρό «no data for this model».
- Coverage ερωτήσεις (regions/years/variables) να απαντούν με metadata, όχι τυχαία data.

## 2. Model descriptions (8)

- [ ] `describe the E3ME model`
- [ ] `what type of model is MUSE?`
- [ ] `is GCAM a general equilibrium model?`
- [ ] `how does REMIND handle technological change?`
- [ ] `who develops the IMAGE model?`
- [ ] `what sectors does PROMETHEUS cover?`
- [ ] `compare GCAM and GEM-E3 as model types`
- [ ] `is TIAM an energy system model?`

Αναμενόμενο:
- Model metadata/description όπου υπάρχει, με link στο model page.
- Yes/no ερωτήσεις να απαντώνται ως explanation, όχι ως data query.
- Να μη μπερδεύει IMAGE με MANAGE ή άλλα κοντινά ονόματα.

## 3. Plots και visualizations (10)

- [ ] `plot nuclear electricity for EU`
- [ ] `bar chart of CO2 emissions for World in 2050`
- [ ] `plot coal primary energy for China from 2020 to 2060`
- [ ] `visualize EU final energy under current policies`
- [ ] `plot CO2 emissions for World from REMIND`
- [ ] `graph electricity generation for India by source`
- [ ] `plot GDP for Japan`
- [ ] `draw emissions for Brazil and India together`
- [ ] `plot carbon capture and storage for World`
- [ ] `chart population for EU until 2100`

Αναμενόμενο:
- Να επιστρέφει `plot_base64` + caption με variable/region/scenario/years.
- Year filters (2050, 2020–2060, until 2100) να εφαρμόζονται στο plot.
- Comparison plot για Brazil+India με δύο σειρές.
- Model-scoped plot να δείχνει μόνο REMIND.

## 4. Links και website navigation (8)

- [ ] `where can I download the data?`
- [ ] `give me the link to the data stories`
- [ ] `where is the contact page?`
- [ ] `link to the IAM COMPACT project website`
- [ ] `where can I see power sector results?`
- [ ] `where can I find publications from the project?`
- [ ] `is there a user guide for the platform?`
- [ ] `take me to the workspaces page`

Αναμενόμενο:
- Link/navigation απάντηση, όχι data query.
- Ζωντανά, on-topic IAM PARIS links.
- Αν δεν υπάρχει τέτοια σελίδα, να το λέει και να δίνει το κοντινότερο.

## 5. Data coverage / metadata (4)

- [ ] `what data do you have for Greece?`
- [ ] `which variables exist for Africa?`
- [ ] `how many scenarios are in the database?`
- [ ] `what is the latest year in the projections?`

Αναμενόμενο:
- Απάντηση από metadata (καταμέτρηση/λίστα), όχι επινόηση.
- Λογικές, φιλτραρισμένες λίστες με «show all» option.

## 6Α. Follow-up: model switch (3)

1. [ ] `show CO2 emissions for World from GCAM`
2. [ ] `now the same from REMIND`
3. [ ] `plot both models together`

Αναμενόμενο:
- Να κρατά variable=CO2, region=World και να αλλάζει μόνο το model.
- Το τελικό plot να συγκρίνει τα δύο models.

## 6Β. Follow-up: year refinement και plot (3)

1. [ ] `EU carbon price under current policies`
2. [ ] `only until 2050`
3. [ ] `plot it`

Αναμενόμενο:
- Να κρατά scope (EU, carbon price, current policies).
- Το year filter να εφαρμοστεί και στο plot.

## 6Γ. Follow-up: από description σε data και link (3)

1. [ ] `what is PROMETHEUS?`
2. [ ] `what data does it have for the EU?`
3. [ ] `where can I read its documentation?`

Αναμενόμενο:
- «it»/«its» να δείχνουν στο PROMETHEUS.
- Στο 2ο βήμα coverage/λίστα variables, στο 3ο documentation link.

## 6Δ. Follow-up: variable switch και comparison (3)

1. [ ] `population for Japan`
2. [ ] `and its GDP?`
3. [ ] `show both on a chart`

Αναμενόμενο:
- Να κρατά region=Japan και να αλλάζει μόνο variable.
- Στο chart δύο μεγέθη (ή καθαρή εξήγηση αν δεν συνδυάζονται λόγω units).
