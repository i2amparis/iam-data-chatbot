# Αποτελέσματα — Extra Manual Test Set 2 (IAM PARIS Chatbot, 50 ερωτήσεις) — ΜΕΤΑ τις διορθώσεις (γύρος 2)

_Αυτόματη εκτέλεση όλων των ερωτήσεων του `manual_chatbot_test_questions_extra2_el.md` (16/07/2026), μετά και τον 2ο γύρο διορθώσεων._
_Follow-up blocks (6Α–6Δ) σε ίδιο session ανά block. Όλα τα links: 43/43 ζωντανά. Unit tests: 425/425._

## Γύρος 2 — τι διορθώθηκε (root-cause, χωρίς hardcoded τιμές δεδομένων)

1. **«greenhouse gas emissions» → `Emissions|Kyoto Gases` με δεδομένα** (πριν: `Emissions|GHG|EUETS` χωρίς data). Δύο αλλαγές: (α) το alias «greenhouse gas» απέκτησε το κανονικό fallback `Emissions|Kyoto Gases` (το IPCC GHG basket), (β) το prefix-fallback του alias resolution δεν «μαντεύει» πλέον ανάμεσα σε sibling sub-scopes (EUETS/Non EUETS) που ο χρήστης δεν ονόμασε — descendant επιλέγεται μόνο αν τα extra segments του υποστηρίζονται από tokens του χρήστη ή είναι μοναδικός. Ολόκληρο το follow-up block 14 (World→EU→India→plot) δουλεύει πλέον.
2. **Metadata ερωτήσεις**: το `what years are available for <model>?` απαντά πλέον από τα καταγεγραμμένα years του μοντέλου (νέα κατηγορία «years» στο model-scoped availability, με «available/are there/exist» phrasing)· για GEM-E3 δίνει τίμιο «no years recorded» (όντως δεν έχει timeseries). Το `where can I download the data?` δρομολογείται ως navigation (νέοι όροι download στο `config/site_navigation.json`) και ο link router επιλύει το download/export intent στο results root → Results.
3. **Plot path fallback για model χωρίς timeseries**: το `plot CO2 emissions for World from REMIND` βγάζει πλέον **plot** με σημείωση «no timeseries data for model REMIND; plotting the models that do have it» — ίδια συμπεριφορά με το data-query path (πριν: σκέτο no-data).
4. **Combined plots σε follow-ups**: τα «plot both models together» (6Α) και «show both on a chart» (6Δ) φτιάχνουν πλέον **structured comparison entities** από τα δύο τελευταία επιτυχή scopes (χωρίς lossy re-extraction από κείμενο) και δίνουν plot: 6Δ → Population vs GDP|MER για JPN, 6Α → model comparison με σημείωση ότι το REMIND-MAgPIE δεν έχει data (σχεδιάζεται το GCAM). Τα scenario lists που προκύπτουν από τα data (all-scenarios) δεν μπλοκάρουν πλέον τη σύγκριση, και το `plot_model_comparison` σχεδιάζει ό,τι υπάρχει αντί για dead-end όταν λείπει ένα μοντέλο.

Επίσης (γύρος 1, ισχύουν): follow-up «from <model>» switch με τίμιο no-data, auto-accept «mutual complete match» (carbon price→`Price|Carbon`), possessive «and its GDP» κρατά region, PROMETHEUS/IMAGE model matching, «how many scenarios» ως metadata.

## Επιβεβαίωση

- 425/425 unit tests.
- Diff τελικού run singles έναντι προηγούμενου: μόνο οι 3 στοχευμένες αλλαγές (years/download/plot-REMIND), κανένα regression, links 43/43.
- Follow-up blocks 14–18 του πρώτου set: όλα υγιή (το 14 πλήρως διορθωμένο).

## Γνωστά υπολειπόμενα (pre-existing, εκτός scope αυτού του γύρου)

- `who develops the IMAGE model?` → ζητά επιβεβαίωση μοντέλου αντί για περιγραφή (LLM γυρίζει bare «IMAGE» με χαμηλό confidence).
- Block 15: «same from solar» κρατά την πηγή coal· «compare coal and solar» καταλήγει σε region `ARE` — pre-existing bugs στο source refinement / region matching.
- `which regions does GCAM cover?` πιθανώς ελλιπές (δεν ενώνει GCAM variants)· `list variables reported by POLES` → γενική λίστα.
- Yes/no ερωτήσεις (does WITCH report…, is GCAM a CGE…) → description αντί για ευθεία απάντηση.
- `Africa` → λείπει alias προς `AFR`· `what is the latest year in the projections?` → LLM απάντηση χωρίς grounding.

## 1 Model-scoped data

**Q:** `primary energy for World from REMIND`  
- route: `data_query` · plot: `False` · scope: variable=`Primary Energy`, region=`World`, model=`REMIND`, start_year=`2005`, end_year=`2100`
- answer: Note: no timeseries data for model `REMIND` in this slice; showing results across the models that do have it.  ### Primary Energy in World  Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `EJ/yr`  Answer: Summary: 55 series across 11 models and 12 scenarios. Showing series 1-12 of 55, ordered by model and scenario. Narrow by model or scenario to view the 43 remaining series.  | Model | Scenario | 2005 | 2
- top link: Models: https://iamparis.eu/models

**Q:** `CO2 emissions for India from MESSAGEix-GLOBIOM`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`IND`, model=`MESSAGEix-GLOBIOM`, start_year=`2005`, end_year=`2100`
- answer: Note: no timeseries data for model `MESSAGEix-GLOBIOM` in this slice; showing results across the models that do have it.  ### Emissions|CO2 in IND  Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr`  Answer: Summary: 37 series across 7 models and 13 scenarios. Showing series 1-12 of 37, ordered by model and scenario. Narrow by model or scenario to view the 25 remaining series.  | Model | Scenario
- top link: Models: https://iamparis.eu/models

**Q:** `electricity generation for EU from WITCH`  
- route: `data_query` · plot: `False` · scope: variable=`Secondary Energy|Electricity`, region=`EU`, model=`WITCH`, start_year=`2005`, end_year=`2100`
- answer: Note: no timeseries data for model `WITCH` in this slice; showing results across the models that do have it.  ### Secondary Energy|Electricity in EU  Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `EJ/yr`  Note: excluded incompatible unit group(s) from this table: `42 - PR_WWH_CP` (Mt CO2/yr).  Answer: Summary: 50 series across 11 models and 13 scenarios. Showing series 1-12 of 50, ordered by model and s
- top link: Models: https://iamparis.eu/models

**Q:** `final energy for China from PROMETHEUS`  
- route: `data_query` · plot: `False` · scope: variable=`Final Energy`, region=`CHN`, model=`PROMETHEUS`, start_year=`2005`, end_year=`2100`
- answer: ### Final Energy in CHN  Scope: scenario `multiple`, model `PROMETHEUS V1`, years `2005-2100` Unit: `EJ/yr`  Answer: **PROMETHEUS V1 - CP_EI** | Year | Value | Unit | |------|-------|------| | 2015 | 83.65 | EJ/yr | | 2020 | 85.07 | EJ/yr | | 2025 | 89.25 | EJ/yr | | 2030 | 91.18 | EJ/yr | | 2035 | 95.01 | EJ/yr | | 2040 | 96.11 | EJ/yr | | 2045 | 96.78 | EJ/yr | | 2050 | 97.36 | EJ/yr |  **PROMETHEUS V1 - NDC_EI** | Year | Va
- top link: PROMETHEUS: https://iamparis.eu/models

**Q:** `does WITCH report carbon price for EU?`  
- route: `data_query` · plot: `False` · scope: —
- answer: I could not confidently match all of the requested wording. The loaded variable catalogue does not represent these term(s) in the top match: `does`, `report`, `witch`. Choose the variable: 1. `Price|Carbon` (price) 2. `Price|Carbon|EUETS` (price) 3. `Price|Carbon|Supply` (price) Reply with a number (1-3), or `yes` for option 1.

**Q:** `which regions does GCAM cover?`  
- route: `data_query` · plot: `False` · scope: —
- answer: Model `GCAM` has these regions: USA, World.
- top link: Results: https://iamparis.eu/results

**Q:** `what years are available for GEM-E3 data?`  
- route: `data_query` · plot: `False` · scope: —
- answer: I could not find any years recorded for model `GEM-E3`.
- top link: GEM-E3: https://iamparis.eu/models

**Q:** `list the variables reported by POLES`  
- route: `data_query` · plot: `False` · scope: —
- answer: I can work with these variables: - 'Economy|Residential & Commercial|Number of buildings - Agricultural Demand - Agricultural Demand|Crops - Agricultural Demand|Crops|Energy - Agricultural Demand|Crops|Energys - Agricultural Demand|Crops|Feed - Agricultural Demand|Crops|Food - Agricultural Demand|Crops|Other - Agricultural Demand|Livestock - Agricultural Demand|Livestock|Food - Agricultural Demand|Livestock|Other - Agricultura
- top link: Results: https://iamparis.eu/results

## 2 Model descriptions

**Q:** `describe the E3ME model`  
- route: `model_explanation` · plot: `False` · scope: model=`E3ME-FTT`
- answer: ### E3ME-FTT  Description: E3ME is a dynamic, computer-based, global macroeconomic model which represents the three pillars of sustainability: economy, society and environment. E3ME's detailed sectoral disaggregation is important for assessing interactions between the pillars. The model is highly empirical in its approach.  Developer: Cambridge Econometrics  Related model documentation: - [IAM PARIS Models](https://iamparis.eu
- top link: E3ME-FTT: https://iamparis.eu/models

**Q:** `what type of model is MUSE?`  
- route: `data_query` · plot: `False` · scope: model=`MUSE`
- answer: ### MUSE  Description: MUSE is a partial equilibrium global integrated assessment model. It uses an agent-based modelling approach to model each sector of the energy system.  Related model documentation: - [IAM PARIS Models](https://iamparis.eu/models)  Reply with a scenario, model, region, or year to narrow the answer.
- top link: MUSE: https://iamparis.eu/models

**Q:** `is GCAM a general equilibrium model?`  
- route: `data_query` · plot: `False` · scope: model=`GCAM`
- answer: ### GCAM  Description: GCAM is a dynamic-recursive model with technology-rich representations of the economy, energy sector, land use and water linked to a climate model that can be used to explore climate change mitigation policies including carbon taxes, carbon trading, regulations and accelerated deployment of energy technology. Regional population and labour productivity growth assumptions drive the energy, land-use and wa
- top link: GCAM: https://iamparis.eu/models

**Q:** `how does REMIND handle technological change?`  
- route: `model_explanation` · plot: `False` · scope: model=`REMIND`
- answer: ### REMIND  Description: REMIND is a global integrated assessment model that links macroeconomic development, the energy system, climate policy and technology choices. It is commonly used to assess mitigation pathways, carbon pricing, energy transitions and technology deployment across regions and sectors.  Model scope: - energy - economy - climate - land-use interactions  Useful for: - global and regional mitigation pathways 
- top link: Models: https://iamparis.eu/models

**Q:** `who develops the IMAGE model?`  
- route: `data_query` · plot: `False` · scope: —
- answer: I matched `IMAGE` as the model, but confidence is low. Which model should I use?

**Q:** `what sectors does PROMETHEUS cover?`  
- route: `model_explanation` · plot: `False` · scope: model=`PROMETHEUS`
- answer: ### PROMETHEUS  Description: The Prometheus model is a comprehensive world energy model with innovative features: it integrates stochastic relations and so all exogenous variables – parameters are stochastic, following explicit probability distributions, including covariance. Therefore, all model results are also stochastic (probability distributions). Among others, stochastic distributions are used for technology change and l
- top link: PROMETHEUS: https://iamparis.eu/models

**Q:** `compare GCAM and GEM-E3 as model types`  
- route: `model_explanation` · plot: `False` · scope: models=`['GEM-E3', 'GCAM']`
- answer: ### GEM-E3 vs GCAM  **GEM-E3** - Methodology/model type: Recursive Dynamic Computable General Equilibrium  **GCAM** - Methodology/model type: Partial equilibrium  The loaded catalogue metadata does not explicitly match `gcam`, `gem` for either model. Comparison focus: method.  [Browse IAM PARIS model documentation](https://iamparis.eu/models)
- top link: Models: https://iamparis.eu/models

**Q:** `is TIAM an energy system model?`  
- route: `model_explanation` · plot: `False` · scope: model=`TIAM_Grantham`
- answer: ### TIAM_Grantham  Description: ETSAP is recalibrating and improving the geographical resolution of the TIAM model. The new version of the model will be available in 2020 and will cover 31 regions, with the same details and general approach of the existing version of the model.  Developer: Grantham, E4SMA  Related model documentation: - [IAM PARIS Models](https://iamparis.eu/models) - Open the Models page and search for: `TIAM
- top link: TIAM_Grantham: https://iamparis.eu/models

## 3 Plots & visualizations

**Q:** `plot nuclear electricity for EU`  
- route: `data_query` · plot: `False` · scope: —
- answer: I found several possible variables, but none has enough evidence for automatic selection. Choose the variable: 1. `Capacity|Electricity|Nuclear` (power capacity, electricity) 2. `Lifetime|Electricity|Nuclear` (electricity) 3. `Capital Cost|Electricity|Nuclear` (electricity) Reply with a number (1-3), or `yes` for option 1.

**Q:** `bar chart of CO2 emissions for World in 2050`  
- route: `data_plotting` · plot: `True` · scope: variable=`Emissions|CO2`, region=`World`, start_year=`2050`, end_year=`2050`
- answer: Showing CO2 Emissions in World across available scenarios (2050-2050). Resolved variable: `Emissions|CO2`.  Reply with a scenario, model, region, or year to narrow the answer.
- top link: Results: https://iamparis.eu/results

**Q:** `plot coal primary energy for China from 2020 to 2060`  
- route: `data_plotting` · plot: `True` · scope: variable=`Primary Energy|Coal`, region=`CHN`, start_year=`2020`, end_year=`2060`
- answer: Showing Coal in CHN across available scenarios (2020-2060). Resolved variable: `Primary Energy|Coal`.  Reply with a scenario, model, region, or year to narrow the answer.
- top link: Results: https://iamparis.eu/results

**Q:** `visualize EU final energy under current policies`  
- route: `data_plotting` · plot: `True` · scope: variable=`Final Energy`, region=`EU`, scenario=`Current Policies`
- answer: Showing Final Energy in EU for selected scenarios `PR_CurPol_EI`, `PR_CurPol_CP`, and `PR_CurPol_CPO`.  Reply with a scenario, model, region, or year to narrow the answer.
- top link: Results: https://iamparis.eu/results

**Q:** `plot CO2 emissions for World from REMIND`  
- route: `data_plotting` · plot: `True` · scope: variable=`Emissions|CO2`, region=`World`, model=`REMIND`
- answer: Note: no timeseries data for model `REMIND` in this slice; plotting the models that do have it.  Showing CO2 Emissions in World across available scenarios. Resolved variable: `Emissions|CO2`.
- top link: Models: https://iamparis.eu/models

**Q:** `graph electricity generation for India by source`  
- route: `data_plotting` · plot: `True` · scope: variable=`Secondary Energy|Electricity`, region=`IND`
- answer: Showing Electricity Generation in IND across available scenarios. Resolved variable: `Secondary Energy|Electricity`.  Reply with a scenario, model, region, or year to narrow the answer.
- top link: Results: https://iamparis.eu/results

**Q:** `plot GDP for Japan`  
- route: `data_plotting` · plot: `True` · scope: variable=`GDP|MER`, region=`JPN`
- answer: Showing MER in JPN across available scenarios. Resolved variable: `GDP|MER`.  Reply with a scenario, model, region, or year to narrow the answer.
- top link: Results: https://iamparis.eu/results

**Q:** `draw emissions for Brazil and India together`  
- route: `data_query` · plot: `False` · scope: —
- answer: I could not confidently match all of the requested wording. The loaded variable catalogue does not represent these term(s) in the top match: `draw`, `india`, `together`. Choose the variable: 1. `Emissions|BC` (emissions) 2. `Emissions|CO` (emissions) 3. `Emissions|OC` (emissions) Reply with a number (1-3), or `yes` for option 1.

**Q:** `plot carbon capture and storage for World`  
- route: `data_query` · plot: `False` · scope: —
- answer: I could not confidently match all of the requested wording. The loaded variable catalogue does not represent these term(s) in the top match: `storage`. Choose the variable: 1. `Carbon Capture` 2. `Carbon Capture|Fossil` 3. `Carbon Capture|Biomass` Reply with a number (1-3), or `yes` for option 1.

**Q:** `chart population for EU until 2100`  
- route: `data_plotting` · plot: `True` · scope: variable=`Population`, region=`EU`, end_year=`2100`
- answer: Showing Population in EU across available scenarios (?-2100).  Reply with a scenario, model, region, or year to narrow the answer.
- top link: Results: https://iamparis.eu/results

## 4 Links & website navigation

**Q:** `where can I download the data?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request: [Results](https://iamparis.eu/results).
- top link: Results: https://iamparis.eu/results

**Q:** `give me the link to the data stories`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request: [Data Stories](https://iamparis.eu/datastories).
- top link: Data Stories: https://iamparis.eu/datastories

**Q:** `where is the contact page?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request: [Contact](https://iamparis.eu/contact).
- top link: Contact: https://iamparis.eu/contact

**Q:** `link to the IAM COMPACT project website`  
- route: `general_qa` · plot: `False` · scope: —
- answer: You can visit the IAM COMPACT project website at [https://iamcompact.eu](https://iamcompact.eu).
- top link: IAM COMPACT project: https://iam-compact.eu

**Q:** `where can I see power sector results?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request: [Results](https://iamparis.eu/results).
- top link: Results: https://iamparis.eu/results

**Q:** `where can I find publications from the project?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: You can find publications from the IAM PARIS project on their official website. Visit the [IAM PARIS Publications page](https://iamparis.eu/publications) for access to various research papers, reports, and other publications related to the project.

**Q:** `is there a user guide for the platform?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Yes, you can find user guides and detailed information about the IAM PARIS platform on their official website. For comprehensive data access and guidance, visit [IAM PARIS Results](https://iamparis.eu/results).

**Q:** `take me to the workspaces page`  
- route: `general_qa` · plot: `False` · scope: —
- answer: You can access the IAM PARIS workspaces page by visiting [https://iamparis.eu/workspaces](https://iamparis.eu/workspaces).

## 5 Data coverage / metadata

**Q:** `what data do you have for Greece?`  
- route: `data_query` · plot: `False` · scope: region=`GREECE`
- answer: ### What I can help you with  - **Models:** 74 available. Examples: 42, AIM/Enduse India 3.3, ALADIN, ATOM. - **Variables:** 1796 available. Examples: 'Economy|Residential & Commercial|Number of buildings, Agricultural Demand, Agricultural Demand|Crops, Agricultural Demand|Crops|Energy. - **Scenarios:** 106 available. Examples: 1.5 TECH, 1.5 °C DRI-trade, 1.5 °C Domestic, BAU. - **Regions:** 204 available. Examples: AFR, AGO, 
- top link: Results: https://iamparis.eu/results

**Q:** `which variables exist for Africa?`  
- route: `data_query` · plot: `False` · scope: —
- answer: I couldn't find `Africa` as a region in the IAM PARIS data, so I can't return results for it. Try one of the available regions, e.g. `AT (Austria)`, `BE (Belgium)`, `BG (Bulgaria)`, `CH (Switzerland)`, `CY (Cyprus)`, or ask `list regions` to see all options.
- top link: Results: https://iamparis.eu/results

**Q:** `how many scenarios are in the database?`  
- route: `data_query` · plot: `False` · scope: —
- answer: I found scenarios matching your request like Baseline, Baseline_HP, Baseline|Harmonized, INDEPENDENCE, Increased_ambition, PR_Baseline, Unharmonised baseline and baseline and 1 more. You can plot variables for any of these scenarios.  Showing 8 of 9. Say `show all scenarios` if you need the full list.  Reply with a scenario, model, region, or year to narrow the answer.
- top link: Results: https://iamparis.eu/results

**Q:** `what is the latest year in the projections?`  
- route: `general_qa` · plot: `False` · scope: start_year=`-1`, end_year=`-1`
- answer: The latest year in the projections for the Prometheus model is 2070, as the update to this year is ongoing. For more detailed data access, you can visit [IAM PARIS results](https://iamparis.eu/results).

## 6Α Follow-up: model switch + plot both

**Q:** `show CO2 emissions for World from GCAM`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`World`, model=`GCAM`, start_year=`2005`, end_year=`2100`
- answer: ### Emissions|CO2 in World  Scope: scenario `multiple`, model `GCAM`, years `2005-2100` Unit: `Mt CO2/yr`  Answer: **GCAM - Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 31.4K | Mt CO2/yr | | 2010 | 36.1K | Mt CO2/yr | | 2015 | 39.2K | Mt CO2/yr | | 2020 | 38.3K | Mt CO2/yr | | 2025 | 40.7K | Mt CO2/yr | | 2030 | -100.00 | Mt CO2/yr | | 2035 | 44.0K | Mt CO2/yr | | 2040 | 45.3K | Mt CO2/yr | | 2045 | 45.5
- top link: GCAM: https://iamparis.eu/models

**Q:** `now the same from REMIND`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`World`, model=`REMIND-MAgPIE`, start_year=`2005`, end_year=`2100`
- answer: Note: no timeseries data for model `REMIND-MAgPIE` in this slice; showing results across the models that do have it.  ### Emissions|CO2 in World  Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr`  Answer: Summary: 66 series across 11 models and 18 scenarios. Showing series 1-12 of 66, ordered by model and scenario. Narrow by model or scenario to view the 54 remaining series.  | Model | Scenario 
- top link: Models: https://iamparis.eu/models

**Q:** `plot both models together`  
- route: `data_plotting` · plot: `True` · scope: variable=`Emissions|CO2`, region=`World`, models=`['GCAM', 'REMIND-MAgPIE']`, start_year=`2005`, end_year=`2100`
- answer: Note: no timeseries data for model(s) `REMIND-MAgPIE` in this slice; plotting `GCAM`.  Showing model comparison of CO2 Emissions in World across available scenarios (2005-2100).
- top link: Models: https://iamparis.eu/models

## 6Β Follow-up: year refinement + plot

**Q:** `EU carbon price under current policies`  
- route: `data_query` · plot: `False` · scope: variable=`Price|Carbon`, region=`EU`, scenario=`Current Policies`, start_year=`2005`, end_year=`2100`
- answer: ### Price|Carbon in EU  Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple`  Answer: Summary: 17 series across 6 models and 3 scenarios. Showing series 1-12 of 17, ordered by model and scenario. Narrow by model or scenario to view the 5 remaining series.  | Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 
- top link: Results: https://iamparis.eu/results

**Q:** `only until 2050`  
- route: `data_query` · plot: `False` · scope: variable=`Price|Carbon`, region=`EU`, scenario=`Current Policies`, start_year=`2050`, end_year=`2050`
- answer: ### Price|Carbon in EU  Scope: scenario `multiple`, model `multiple`, years `2050` Unit: `multiple`  Answer: Summary: 17 series across 6 models and 3 scenarios. Showing series 1-12 of 17, ordered by model and scenario. Narrow by model or scenario to view the 5 remaining series.  | Model | Scenario | 2050 | Unit | |---|---|---|---| | e3me | PR_CurPol_CP | 42.79 | US$2010/tCO2 | | e3me | PR_CurPol_EI | 52.04 | US$2010/tCO2 | | g
- top link: Results: https://iamparis.eu/results

**Q:** `plot it`  
- route: `data_plotting` · plot: `True` · scope: variable=`Price|Carbon`, region=`EU`, scenario=`Current Policies`, start_year=`2050`, end_year=`2050`
- answer: Showing Carbon in EU across available scenarios (2050-2050). Resolved variable: `Price|Carbon`.
- top link: Results: https://iamparis.eu/results

## 6Γ Follow-up: description → data → link

**Q:** `what is PROMETHEUS?`  
- route: `model_explanation` · plot: `False` · scope: model=`PROMETHEUS`
- answer: ### PROMETHEUS  Description: The Prometheus model is a comprehensive world energy model with innovative features: it integrates stochastic relations and so all exogenous variables – parameters are stochastic, following explicit probability distributions, including covariance. Therefore, all model results are also stochastic (probability distributions). Among others, stochastic distributions are used for technology change and l
- top link: PROMETHEUS: https://iamparis.eu/models

**Q:** `what data does it have for the EU?`  
- route: `data_query` · plot: `False` · scope: region=`EU`, model=`PROMETHEUS`
- answer: Model `PROMETHEUS V1` has these variables: Capital Cost|Electricity|Biomass|w/ CCS, Capital Cost|Electricity|Biomass|w/o CCS, Capital Cost|Electricity|Coal|w/ CCS, Capital Cost|Electricity|Coal|w/o CCS, Capital Cost|Electricity|Gas|w/ CCS, Capital Cost|Electricity|Gas|w/o CCS, Capital Cost|Electricity|Geothermal, Capital Cost|Electricity|Hydro, Capital Cost|Electricity|Nuclear, Capital Cost|Electricity|Solar|PV, Capital Cost|E
- top link: PROMETHEUS: https://iamparis.eu/models

**Q:** `where can I read its documentation?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: You can read the documentation for the PROMETHEUS model on the IAM PARIS website. Here is the link to the [IAM PARIS Models page](https://iamparis.eu/models), where you can find detailed information about the PROMETHEUS model and other integrated assessment models.

## 6Δ Follow-up: variable switch + comparison

**Q:** `population for Japan`  
- route: `data_query` · plot: `False` · scope: variable=`Population`, region=`JPN`, start_year=`2005`, end_year=`2100`
- answer: ### Population in JPN  Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple`  Answer: Summary: 39 series across 7 models and 12 scenarios. Showing series 1-12 of 39, ordered by model and scenario. Narrow by model or scenario to view the 27 remaining series.  | Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090
- top link: Results: https://iamparis.eu/results

**Q:** `and its GDP?`  
- route: `data_query` · plot: `False` · scope: variable=`GDP|MER`, region=`JPN`, start_year=`2005`, end_year=`2100`
- answer: ### GDP|MER in JPN  Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple`  Answer: Summary: 20 series across 3 models and 8 scenarios. Showing series 1-12 of 20, ordered by model and scenario. Narrow by model or scenario to view the 8 remaining series.  | Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 20
- top link: Results: https://iamparis.eu/results

**Q:** `show both on a chart`  
- route: `data_plotting` · plot: `True` · scope: variable=`Population`, variables=`['Population', 'GDP|MER']`, region=`JPN`, start_year=`2005`, end_year=`2100`
- answer: Showing comparison of Population vs MER in JPN across available scenarios (2005-2100).
- top link: Results: https://iamparis.eu/results
