# Αποτελέσματα — Extra Manual Test Set (IAM PARIS Chatbot)

_Αυτόματη εκτέλεση όλων των ερωτήσεων του `manual_chatbot_test_questions_extra_el.md`._
_Follow-up blocks (14–18) σε ίδιο session. 28 links ελέγχθηκαν με HTTP → 0 νεκρά._

## Σύνοψη ευρημάτων

Αυτό το set είναι πιο απαιτητικό και ξεσκεπάζει αρκετές αδυναμίες πέρα από το βασικό.

**Δουλεύουν καλά ✅**
- **Links & navigation (ενότητα 13): όλα** → σωστά IAM PARIS links (homepage, model docs, scenario explorer, applications, policy catalogue, NDC ASPECTS, IAM COMPACT, transport/buildings/AFOLU/industrial transformation, methodology). 28/28 ζωντανά.
- Concepts (ενότητα 1): «what problem do IAMs solve», «decarbonisation», «model uncertainty» → σωστά general_qa.
- Βασικά emissions: greenhouse gas→`Emissions|Kyoto Gases` (όχι CO2!), methane→CH4, industrial→CO2|Energy and Industrial Processes.
- Core data: GDP MER, primary energy coal/gas/oil, carbon price, consumption, investment.
- Model descriptions: POLES, GEM-E3, WITCH σωστά· cross-model «no timeseries» τίμιο (GEM-E3, REMIND).
- Plots: τα περισσότερα επιστρέφουν εικόνα (greenhouse gas, carbon price, GDP, methane).

**Σοβαρά προβλήματα ❌**
1. **Ψεύτικες/άγνωστες περιοχές δεν απορρίπτονται** (ενότητα 21): `CO2 for Gotham`, `GDP for Middle Earth`, `electricity for Atlantis`, `population for Mars`, `carbon price for Hogwarts` → επιστρέφουν **παγκόσμια δεδομένα** σαν έγκυρα. Παραβιάζει το «καμία επινόηση». (Fake models & fake variables απορρίπτονται σωστά.)
2. **Follow-up blocks 14–17 σπάνε**: εμφανίζεται spurious σενάριο `NZE`/`NZE_EUPol_Stand` και χάνεται το context (`same for EU`, `what about EU?`, `same under baseline`, `compare them`, `plot India` → no-data ή «choose variable»).
3. **Λάθος variable specificity**: `solar/wind/hydro electricity generation` → γενικό `Secondary Energy|Electricity` (χάνει την πηγή)· `final energy in industry`→γενικό Final Energy· **`GDP PPP`→`GDP|MER`**· `energy price`→Price|Biomass· `transport final energy`→Price variable· `residential/commercial energy`→`Emissions|BC`.
4. **Year ranges**: `between 2030 and 2060` → μόνο 2030· `in 2100` → δεν φιλτράρει· `in 2030 and 2050` → μόνο 2030.
5. **Conceptual/yes-no misroute**: `what is a reference scenario?`, `does POLES include energy markets?`, `does GCAM include land-use change?` → data_query αντί general_qa.
6. **IMAGE → MANAGE** (λάθος model match)· `emissions from AIM` → gcam.

**Μέτρια ⚠️**
- Ambiguous μονές λέξεις (`solar`, `wind`, `transport`, `coal`) → διαλέγουν αυθαίρετα variable αντί για 2–4 επιλογές (`coal`→region Colombia!).
- Typos ασυνεπή: κάποια δουλεύουν (`curent policie co2 eu`, `basline primry energy wrld`, `plz plot wind eletricity europe`), πολλά όχι (`methan emisions wrld`, `solr electrcity india`, `gdp ppp chna`).
- `which scenarios are available for EU emissions?` → γενική επισκόπηση· `scenarios related to NDCs` → λάθος σενάρια (total_FiT…).
- AFOLU (ενότητα 8): πολλά «need detail» (pasture, bioenergy crop, water, biodiversity, livestock).

**Πιο κρίσιμα για διόρθωση:** (1) απόρριψη άγνωστων περιοχών, (2) follow-up spurious-scenario, (3) year ranges, (4) variable specificity (solar/wind generation, GDP PPP).


## 1 IAM/scenario concepts

**Q:** `what problem do integrated assessment models solve?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Integrated Assessment Models (IAMs) are designed to address complex, interdisciplinary problems related to climate change, energy systems, and sustainable development. They integrate knowledge from various fields such as economics, environmental science, and technology to provide comprehensive insights into the following key areas: 1. **Climate Change Mitigation and Adaptation**: IAMs help in understanding the potential impacts of climate change and evaluating strategies for reducing greenhouse …
- top link: Model Comparison: https://iamparis.eu/models/comparison (200)

**Q:** `how are IAM scenarios different from forecasts?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: IAM scenarios and forecasts are both tools used to understand potential future outcomes, but they differ in their approach and purpose: ### IAM Scenarios - **Purpose**: Scenarios in Integrated Assessment Models (IAMs) are used to explore a range of possible futures based on different assumptions about key drivers such as technology, policy, and socio-economic factors. - **Nature**: They are not predictions but rather "what-if" analyses that help policymakers understand the implications of differ…
- top link: FORECAST: https://iamparis.eu/models (200)

**Q:** `why do different IAMs give different results?`  
- route: `data_query` · plot: `False` · scope: —
- answer: Based on your wording, I think one of these is the closest match. Choose the variable: 1. `Emissions|BC` (emissions) 2. `Emissions|CO` (emissions) 3. `Emissions|OC` (emissions) Reply with a number (1-3), or `yes` for option 1.
- top link: Results: https://iamparis.eu/results (200)

**Q:** `what should I check before comparing model outputs?`  
- route: `modelling_suggestions` · plot: `False` · scope: —
- answer: Here are some modelling study suggestions you could explore: 1. Explore the impact of different carbon pricing scenarios on emission reductions. 2. Investigate the role of renewable energy adoption in achieving climate targets. 3. Analyze the effects of land-use changes on greenhouse gas emissions. 4. Study the implications of energy efficiency improvements across sectors. 5. Examine the potential of negative emissions technologies in climate mitigation pathways. 6. Assess the outcomes of differ…
- top link: Results: https://iamparis.eu/results (200)

**Q:** `what does a policy scenario mean?`  
- route: `data_query` · plot: `False` · scope: scenario=`Policy`
- answer: Based on your wording, I think one of these is the closest match. Choose the variable: 1. `Policy Cost|GDP Loss` 2. `Temperature|Global Mean` 3. `Policy Cost|Consumption Loss` Reply with a number (1-3), or `yes` for option 1.
- top link: Behavioural changes for climate policy: https://iamparis.eu/results/iam-compact/behavioural-changes-for-climate-policy/policy_questions (200)

**Q:** `what is a reference scenario?`  
- route: `data_query` · plot: `False` · scope: scenario=`Reference`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Results: https://iamparis.eu/results (200)

**Q:** `what does decarbonisation mean in these results?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Decarbonisation in the context of IAM PARIS results refers to the process of reducing carbon dioxide (CO2) emissions associated with energy production and consumption. This involves transitioning from fossil fuel-based energy sources to low-carbon or renewable energy sources, improving energy efficiency, and implementing technologies and practices that reduce carbon emissions across various sectors. In the IAM PARIS models, decarbonisation is often analyzed through scenarios that project future …
- top link: Circularity & decarbonisation interventions: https://iamparis.eu/datastories/CE_intervention (200)

**Q:** `how should I interpret model uncertainty?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Interpreting model uncertainty involves understanding the range of possible outcomes that a model predicts, given the inherent variability and assumptions in its inputs and structure. Here are some key points to consider: ### 1. **Types of Uncertainty:** - **Parameter Uncertainty:** Variability in model inputs or parameters, such as economic growth rates or technology costs. - **Structural Uncertainty:** Differences in model structure or the way processes are represented. - **Scenario Uncertaint…

## 2 Model explanations

**Q:** `give me an overview of AIM`  
- route: `data_query` · plot: `False` · scope: model=`AIM/Enduse India 3.3`
- answer: I need one more detail. Which variable should I use?
- top link: AIM/Enduse India 3.3: https://iamparis.eu/models (200)

**Q:** `what is IMAGE used for?`  
- route: `model_explanation` · plot: `False` · scope: model=`MANAGE`
- answer: I matched `MANAGE`, but IAM PARIS does not expose a dedicated assumptions/metadata page for that model in the local model catalog. Use the IAM PARIS model and results links below to inspect related documentation or available data.
- top link: MANAGE: https://iamparis.eu/models/manage (200)

**Q:** `explain POLES model`  
- route: `model_explanation` · plot: `False` · scope: model=`POLES`
- answer: ### POLES Description: POLES is a dynamic-recursive model with technology-rich representations of the energy secto that can be used to explore climate change mitigation policies including carbon taxes, regulations and accelerated deployment of energy technology. Regional population and GDP growth assumptions drive the energy sector employing numerous technology options to produce, transform and provide energy services. Related model documentation: - [IAM PARIS Models](https://iamparis.eu/models)
- top link: POLES: https://iamparis.eu/models (200)

**Q:** `what kind of model is GEM-E3?`  
- route: `data_query` · plot: `False` · scope: region=`India`, model=`GEM-E3`
- answer: ### GEM-E3 Description: The GEM-E3 model is a multi-regional, multi-sectoral, recursive dynamic computable general equilibrium (CGE) model which provides details on the macro-economy and its interaction with the environment and the energy system. It is an empirical, large scale model, written entirely in structural form. GEM-E3 allows for a consistent comparative analysis of policy scenarios since it ensures that in all scenarios, the economic system remains in general equilibrium. In addition, …
- top link: GEM-E3: https://iamparis.eu/models (200)

**Q:** `tell me about COFFEE model`  
- route: `data_query` · plot: `False` · scope: model=`COFFEE`
- answer: I matched `COFFEE` as the model, but confidence is low. Which model should I use?

**Q:** `what is GCAM-PR?`  
- route: `model_explanation` · plot: `False` · scope: model=`GCAM-PR`
- answer: ### GCAM Description: GCAM is a dynamic-recursive model with technology-rich representations of the economy, energy sector, land use and water linked to a climate model that can be used to explore climate change mitigation policies including carbon taxes, carbon trading, regulations and accelerated deployment of energy technology. Regional population and labour productivity growth assumptions drive the energy, land-use and water systems employing numerous technology options to produce, transform…
- top link: Models: https://iamparis.eu/models (200)

**Q:** `which model focuses on energy and land systems?`  
- route: `data_query` · plot: `False` · scope: —
- answer: ### Models covering Land Use 6 model(s) of 90 report at least one land use variable in IAM PARIS: GCAM 7.0, GCAM-PR 5.3, GCAM-PR 7.0, GEMINI-E3 7.0, gcam, gemini_e3 Ask for a specific model (e.g. `tell me about GCAM 7.0`) or a data query (e.g. `land use emissions for Europe`) to go deeper.
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `which models can analyse mitigation pathways?`  
- route: `data_query` · plot: `False` · scope: —
- answer: There are 74 models available. Examples: 42, AIM/Enduse India 3.3, ALADIN, ATOM, CHANCE, CLEWs, CLEWs - Ethiopia, CLEWs - Kenya. You can ask for details about a specific model using `info [model name]`, or say `list variables` to see the kinds of outputs available. Showing 8 of 74. Say `show all models` if you need the full list.
- top link: Cost of capital evolution effects on regional mitigation pathways: https://iamparis.eu/results/iam-compact/cost-of-capital-evolution-effects-on-regional-mitigation-pathways/policy_questions (200)

## 3 Assumptions/scope/limits

**Q:** `what are the main limitations of IMAGE?`  
- route: `data_query` · plot: `False` · scope: model=`MANAGE`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: MANAGE: https://iamparis.eu/models/manage (200)

**Q:** `does POLES include energy markets?`  
- route: `data_query` · plot: `False` · scope: —
- answer: Based on your wording, I think one of these is the closest match. Choose the variable: 1. `Secondary Energy|Liquids|Fossil` (secondary energy) 2. `Final Energy|Residential and Commercial|Commercial|Liquids|Fossil` (final energy use, buildings) 3. `Price|Primary Energy|Gas` (primary energy, gas) Reply with a number (1-3), or `yes` for option 1.
- top link: POLES: https://iamparis.eu/models (200)

**Q:** `does AIM cover global mitigation pathways?`  
- route: `data_query` · plot: `False` · scope: region=`World`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Cost of capital evolution effects on regional mitigation pathways: https://iamparis.eu/results/iam-compact/cost-of-capital-evolution-effects-on-regional-mitigation-pathways/policy_questions (200)

**Q:** `does GCAM include land-use change?`  
- route: `data_query` · plot: `False` · scope: variable=`Final Energy|Non-Energy Use`, model=`gcam`
- answer: ### Final Energy|Non-Energy Use Scope: scenario `multiple`, model `gcam`, years `2005-2100` Unit: `EJ/yr` Answer: **gcam - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 0.22 | EJ/yr | | 2010 | 0.22 | EJ/yr | | 2015 | 0.24 | EJ/yr | | 2020 | 0.24 | EJ/yr | | 2025 | 0.28 | EJ/yr | | 2030 | 0.32 | EJ/yr | | 2035 | 0.35 | EJ/yr | | 2040 | 0.39 | EJ/yr | | 2045 | 0.43 | EJ/yr | | 2050 | 0.46 | EJ/yr | | 2055 | 0.49 | EJ/yr | | 2060 | 0.52 | EJ/yr | | 2065 | 0.55 | EJ/yr | | 2…
- top link: GCAM: https://iamparis.eu/models (200)

**Q:** `does REMIND represent negative emissions?`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2|Energy and Industrial Processes`, model=`REMIND`
- answer: Note: no timeseries data for model `REMIND` in this slice; showing results across the models that do have it. ### Emissions|CO2|Energy and Industrial Processes Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **E3ME 6.1 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2010 | 4.3K | Mt CO2/yr | | 2015 | 4.0K | Mt CO2/yr | | 2020 | 3.4K | Mt CO2/yr | | 2025 | 3.2K | Mt CO2/yr | | 2030 | 3.0K | Mt CO2/yr | | 2035 | 2.9K | Mt CO2/yr | | 2040 | 2…
- top link: FORECAST: https://iamparis.eu/models (200)

**Q:** `are CCS assumptions fixed across scenarios?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: To determine if Carbon Capture and Storage (CCS) assumptions are fixed across scenarios, we need to look at the specific Integrated Assessment Models (IAMs) and scenarios in question. In general, CCS assumptions can vary across scenarios depending on the model's design and the specific scenario narratives. For detailed information on CCS assumptions in specific models and scenarios, you can explore the IAM PARIS database at [https://iamparis.eu/results](https://iamparis.eu/results). This resourc…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `where can I check assumptions for WITCH?`  
- route: `model_explanation` · plot: `False` · scope: model=`WITCH`
- answer: ### WITCH Description: WITCH is an integrated assessment model focused on the interaction between the economy, energy system, climate policy and technological change. It is often used for regional mitigation strategies, policy design and technology innovation analysis. Model scope: - economy - energy - climate policy - technology innovation Useful for: - regional mitigation and burden-sharing analysis - carbon pricing and policy assessment - technology innovation and diffusion scenarios - long-t…
- top link: Models: https://iamparis.eu/models (200)

**Q:** `which model documentation should I read for MESSAGEix-GLOBIOM?`  
- route: `data_query` · plot: `False` · scope: —
- answer: There are 74 models available. Examples: 42, AIM/Enduse India 3.3, ALADIN, ATOM, CHANCE, CLEWs, CLEWs - Ethiopia, CLEWs - Kenya. You can ask for details about a specific model using `info [model name]`, or say `list variables` to see the kinds of outputs available. Showing 8 of 74. Say `show all models` if you need the full list.
- top link: GLOBIO: https://iamparis.eu/models (200)

## 4 Emissions

**Q:** `greenhouse gas emissions for EU`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|Kyoto Gases`, region=`EU`
- answer: ### Emissions|Kyoto Gases in EU Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2-equiv/yr` Answer: **GCAM-PR 5.3 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2005 | 5.2K | Mt CO2-equiv/yr | | 2010 | 4.7K | Mt CO2-equiv/yr | | 2015 | 4.4K | Mt CO2-equiv/yr | | 2020 | 4.2K | Mt CO2-equiv/yr | | 2025 | 3.7K | Mt CO2-equiv/yr | | 2030 | 3.4K | Mt CO2-equiv/yr | | 2035 | 2.7K | Mt CO2-equiv/yr | | 2040 | 2.4K | Mt CO2-equiv/yr | | 2045 | 2.0K | Mt CO2-e…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `CO2 emissions for Japan`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`Japan`, model=`gcam`
- answer: ### Emissions|CO2 in Japan Scope: scenario `multiple`, model `gcam`, years `2005-2100` Unit: `Mt CO2/yr` Answer: **gcam - Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 1.2K | Mt CO2/yr | | 2010 | 1.2K | Mt CO2/yr | | 2015 | 1.2K | Mt CO2/yr | | 2020 | 1.1K | Mt CO2/yr | | 2025 | 1.1K | Mt CO2/yr | | 2030 | 1.1K | Mt CO2/yr | | 2035 | 1.0K | Mt CO2/yr | | 2040 | 973.98 | Mt CO2/yr | | 2045 | 897.64 | Mt CO2/yr | | 2050 | 833.21 | Mt CO2/yr | | 2055 | 794.47 | Mt CO2/yr | | 2…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `methane emissions for World`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CH4`, region=`World`
- answer: ### Emissions|CH4 in World Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CH4/yr` Answer: **gcam - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 345.08 | Mt CH4/yr | | 2010 | 367.57 | Mt CH4/yr | | 2015 | 380.58 | Mt CH4/yr | | 2020 | 409.93 | Mt CH4/yr | | 2025 | 426.25 | Mt CH4/yr | | 2030 | 434.59 | Mt CH4/yr | | 2035 | 440.83 | Mt CH4/yr | | 2040 | 446.37 | Mt CH4/yr | | 2045 | 456.36 | Mt CH4/yr | | 2050 | 461.74 | Mt CH4/yr | | 2055 | 464…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `N2O emissions for Brazil`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|N2O`, region=`Brazil`
- answer: I could not find data for `Emissions|N2O` in `Brazil`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. variable `Emissions|CO2` (CO2 emissions) 2. variable `Emissions|CO2|AFOLU` (CO2 emissions, land use) 3. variable `Emissions|CO2|Energy` (CO2 emissions) Reply with `1`, `2`, or `3`, or type the variable you want.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `energy-related CO2 emissions for China`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`CHN`
- answer: ### Emissions|CO2 in CHN Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr` Answer: **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2010 | 7.8K | Mt CO2/yr | | 2015 | 9.4K | Mt CO2/yr | | 2020 | 10.3K | Mt CO2/yr | | 2025 | 10.8K | Mt CO2/yr | | 2030 | 11.1K | Mt CO2/yr | | 2035 | 10.8K | Mt CO2/yr | | 2040 | 9.5K | Mt CO2/yr | | 2045 | 8.1K | Mt CO2/yr | | 2050 | 7.5K | Mt CO2/yr | **e3me - PR_CurPol_CP** | Year | Value | Unit | |------|--…
- top link: EU-CHINA BRIDGE: https://iamparis.eu/results/eu-china-bridge (200)

**Q:** `industrial emissions for India`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2|Energy and Industrial Processes`, region=`IND`
- answer: ### Emissions|CO2|Energy and Industrial Processes in IND Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2010 | 1.7K | Mt CO2/yr | | 2015 | 2.3K | Mt CO2/yr | | 2020 | 2.7K | Mt CO2/yr | | 2025 | 3.2K | Mt CO2/yr | | 2030 | 3.9K | Mt CO2/yr | | 2035 | 4.7K | Mt CO2/yr | | 2040 | 5.7K | Mt CO2/yr | | 2045 | 6.6K | Mt CO2/yr | | 2050 | 7.6K | Mt CO2/yr | **e3me - PR_CurPol_CP** | Year …
- top link: Industrial transformation: https://iamparis.eu/results/ndc-aspects/industrial-transformation/policy_questions (200)

**Q:** `land-use CO2 emissions for World`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`World`
- answer: ### Emissions|CO2 in World Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr` Answer: **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2010 | 31.5K | Mt CO2/yr | | 2015 | 34.4K | Mt CO2/yr | | 2020 | 34.4K | Mt CO2/yr | | 2025 | 35.4K | Mt CO2/yr | | 2030 | 36.9K | Mt CO2/yr | | 2035 | 38.1K | Mt CO2/yr | | 2040 | 38.8K | Mt CO2/yr | | 2045 | 39.1K | Mt CO2/yr | | 2050 | 40.4K | Mt CO2/yr | **e3me - PR_CurPol_CP** | Year | Value | Unit | |--…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `carbon dioxide removal for EU`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2|Energy`, region=`EU (European Union)`
- answer: I found the variable `Emissions|CO2|Energy` (CO2 emissions) in `EU (European Union)`. I still need the scenario. Choose the scenario: 1. `PR_Baseline` (baseline) 2. `PR_CurPol_CP` (current policy) 3. `PR_CurPol_CPO` (current policy) Reply with a number (1-3), or `yes` for option 1.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `carbon capture and storage for World`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2|Energy`, region=`World`
- answer: I found the variable `Emissions|CO2|Energy` (CO2 emissions) in `World`. I still need the scenario. Choose the scenario: 1. `Baseline` (baseline) 2. `CP_EI` 3. `NDC_EI` (NDC) Reply with a number (1-3), or `yes` for option 1.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `emissions from agriculture for Brazil`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CH4|AFOLU|Agriculture`, region=`Brazil`
- answer: I could not find data for `Emissions|CH4|AFOLU|Agriculture` in `Brazil`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. variable `Emissions|CO2` (CO2 emissions) 2. variable `Emissions|CO2|AFOLU` (CO2 emissions, land use) 3. variable `Emissions|CO2|Energy` (CO2 emissions) Reply with `1`, `2`, or `3`, or type the variable you want.
- top link: AFOLU transformation: https://iamparis.eu/results/ndc-aspects/afolu-transformation/policy_questions (200)

## 5 Energy system

**Q:** `renewable electricity generation for World`  
- route: `data_query` · plot: `False` · scope: variable=`Secondary Energy|Electricity`, region=`World`
- answer: ### Secondary Energy|Electricity in World Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `EJ/yr` Answer: **42 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 54.56 | EJ/yr | | 2010 | 64.41 | EJ/yr | | 2015 | 72.64 | EJ/yr | | 2020 | 82.71 | EJ/yr | | 2025 | 93.95 | EJ/yr | | 2030 | 105.44 | EJ/yr | | 2035 | 116.99 | EJ/yr | | 2040 | 128.05 | EJ/yr | | 2045 | 138.96 | EJ/yr | **42 - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 20…
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `solar electricity generation for India`  
- route: `data_query` · plot: `False` · scope: variable=`Secondary Energy|Electricity`, region=`IND`
- answer: ### Secondary Energy|Electricity in IND Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `EJ/yr` Answer: **42 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 1.76 | EJ/yr | | 2010 | 2.59 | EJ/yr | | 2015 | 3.70 | EJ/yr | | 2020 | 4.90 | EJ/yr | | 2025 | 6.43 | EJ/yr | | 2030 | 8.16 | EJ/yr | | 2035 | 10.05 | EJ/yr | | 2040 | 12.03 | EJ/yr | | 2045 | 14.10 | EJ/yr | **42 - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 2005 | 1.76 | …
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `wind electricity generation for EU`  
- route: `data_query` · plot: `False` · scope: variable=`Secondary Energy|Electricity`, region=`EU`
- answer: ### Secondary Energy|Electricity in EU Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **42 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2005 | -9.97 | Mt CO2/yr | | 2010 | -10.21 | Mt CO2/yr | | 2015 | -9.82 | Mt CO2/yr | | 2020 | -9.90 | Mt CO2/yr | | 2025 | -10.03 | Mt CO2/yr | | 2030 | -10.39 | Mt CO2/yr | | 2035 | -10.92 | Mt CO2/yr | | 2040 | -11.49 | Mt CO2/yr | | 2045 | -12.19 | Mt CO2/yr | **E3ME 6.1 - PR_WWH_CP** | Year | Valu…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `coal primary energy for China`  
- route: `data_query` · plot: `False` · scope: variable=`Primary Energy|Coal`, region=`CHN`
- answer: ### Primary Energy|Coal in CHN Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **42 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 50.40 | EJ/yr | | 2010 | 74.90 | EJ/yr | | 2015 | 82.58 | EJ/yr | | 2020 | 85.37 | EJ/yr | | 2025 | 91.97 | EJ/yr | | 2030 | 93.82 | EJ/yr | | 2035 | 94.99 | EJ/yr | | 2040 | 93.53 | EJ/yr | | 2045 | 90.92 | EJ/yr | **42 - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 2005 | 50.40 |…
- top link: EU-CHINA BRIDGE: https://iamparis.eu/results/eu-china-bridge (200)

**Q:** `gas primary energy for EU`  
- route: `data_query` · plot: `False` · scope: variable=`Primary Energy|Gas`, region=`EU`
- answer: ### Primary Energy|Gas in EU Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **42 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2005 | 18.63 | EJ/yr | | 2010 | 18.72 | EJ/yr | | 2015 | 14.97 | EJ/yr | | 2020 | 14.96 | EJ/yr | | 2025 | 15.31 | EJ/yr | | 2030 | 15.31 | EJ/yr | | 2035 | 15.43 | EJ/yr | | 2040 | 15.29 | EJ/yr | | 2045 | 15.38 | EJ/yr | **E3ME 6.1 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2010 | 18.77 | …
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `oil primary energy for World`  
- route: `data_query` · plot: `False` · scope: variable=`Primary Energy|Oil`, region=`World`
- answer: I found the variable `Primary Energy|Oil` (primary energy, oil) in `World`. I still need the scenario. Choose the scenario: 1. `PR_Baseline` (baseline) 2. `PR_CurPol_CP` (current policy) 3. `PR_CurPol_CPO` (current policy) Reply with a number (1-3), or `yes` for option 1.
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `nuclear electricity for Japan`  
- route: `data_query` · plot: `False` · scope: variable=`Secondary Energy|Electricity|Nuclear`, region=`Japan`
- answer: I could not find data for `Secondary Energy|Electricity|Nuclear` in `Japan`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. region `AGO` (AGO (Angola)) 2. region `ARE` (ARE (United Arab Emirates)) 3. region `ARG` (ARG (Argentina)) Reply with `1`, `2`, or `3`, or type the region you want.
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `hydro electricity generation for Brazil`  
- route: `data_query` · plot: `False` · scope: variable=`Secondary Energy|Electricity`, region=`Brazil`
- answer: I could not find data for `Secondary Energy|Electricity` in `Brazil`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. region `AGO` (AGO (Angola)) 2. region `ARE` (ARE (United Arab Emirates)) 3. region `ARG` (ARG (Argentina)) Reply with `1`, `2`, or `3`, or type the region you want.
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `final energy demand in buildings for EU`  
- route: `data_query` · plot: `False` · scope: variable=`Final Energy|Buildings|Residential`, region=`EU`, scenario=`PR_WWH_CP`, model=`GEMINI-E3 7.0`
- answer: ### Final Energy|Buildings|Residential in EU Scope: scenario `PR_WWH_CP`, model `GEMINI-E3 7.0`, years `2005-2050` Unit: `EJ/yr` Answer: **GEMINI-E3 7.0 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2015 | 8.91 | EJ/yr | | 2020 | 9.02 | EJ/yr | | 2025 | 8.95 | EJ/yr | | 2030 | 8.74 | EJ/yr | | 2035 | 8.80 | EJ/yr | | 2040 | 8.87 | EJ/yr | | 2045 | 8.92 | EJ/yr | | 2050 | 8.96 | EJ/yr | Next: - Ask `plot Final Energy|Buildings|Residential for EU` to visualize this result.
- top link: Buildings Transformation: https://iamparis.eu/results/ndc-aspects/buildings-transformation/policy_questions (200)

**Q:** `final energy demand in industry for China`  
- route: `data_query` · plot: `False` · scope: variable=`Final Energy`, region=`CHN`
- answer: ### Final Energy in CHN Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **42 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 49.92 | EJ/yr | | 2010 | 66.91 | EJ/yr | | 2015 | 80.61 | EJ/yr | | 2020 | 91.84 | EJ/yr | | 2025 | 103.99 | EJ/yr | | 2030 | 112.96 | EJ/yr | | 2035 | 119.93 | EJ/yr | | 2040 | 124.03 | EJ/yr | | 2045 | 127.06 | EJ/yr | **42 - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 2005 | 49.92 | E…
- top link: Buildings Transformation: https://iamparis.eu/results/ndc-aspects/buildings-transformation/policy_questions (200)

## 6 Economy/pop/prices

**Q:** `GDP MER for World`  
- route: `data_query` · plot: `False` · scope: variable=`GDP|MER`, region=`World`
- answer: ### GDP|MER in World Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2010 | 65.9K | billion US$2010/yr | | 2015 | 75.6K | billion US$2010/yr | | 2020 | 87.2K | billion US$2010/yr | | 2025 | 99.7K | billion US$2010/yr | | 2030 | 113.3K | billion US$2010/yr | | 2035 | 127.6K | billion US$2010/yr | | 2040 | 142.4K | billion US$2010/yr | | 2045 | 158.2K | billion US$2010/yr | | 2050 | 17…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `GDP PPP for India`  
- route: `data_query` · plot: `False` · scope: variable=`GDP|MER`, region=`IND`
- answer: ### GDP|MER in IND Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2010 | 1.7K | billion US$2010/yr | | 2015 | 2.3K | billion US$2010/yr | | 2020 | 3.5K | billion US$2010/yr | | 2025 | 4.9K | billion US$2010/yr | | 2030 | 6.2K | billion US$2010/yr | | 2035 | 7.7K | billion US$2010/yr | | 2040 | 9.5K | billion US$2010/yr | | 2045 | 11.5K | billion US$2010/yr | | 2050 | 13.8K | billion…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `population projection for Brazil`  
- route: `data_query` · plot: `False` · scope: variable=`Population`, region=`Brazil`
- answer: I could not find data for `Population` in `Brazil`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. region `AGO` (AGO (Angola)) 2. region `ARE` (ARE (United Arab Emirates)) 3. region `ARG` (ARG (Argentina)) Reply with `1`, `2`, or `3`, or type the region you want.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `income for EU`  
- route: `data_query` · plot: `False` · scope: region=`EU`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `consumption for China`  
- route: `data_query` · plot: `False` · scope: variable=`Consumption`, region=`CHN`, model=`gemini_e3`
- answer: ### Consumption in CHN Scope: scenario `multiple`, model `gemini_e3`, years `2005-2100` Unit: `billion US$2010/yr` Answer: **gemini_e3 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2015 | 9.8K | billion US$2010/yr | | 2020 | 13.1K | billion US$2010/yr | | 2025 | 16.6K | billion US$2010/yr | | 2030 | 20.5K | billion US$2010/yr | | 2035 | 24.1K | billion US$2010/yr | | 2040 | 27.8K | billion US$2010/yr | | 2045 | 31.2K | billion US$2010/yr | | 2050 | 34.6K | billion US$2010/yr …
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `investment for World`  
- route: `data_query` · plot: `False` · scope: variable=`Investment`, region=`World`
- answer: ### Investment in World Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **gemini_e3 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2015 | 28.1K | billion US$2010/yr OR local currency | | 2020 | 34.1K | billion US$2010/yr OR local currency | | 2025 | 40.4K | billion US$2010/yr OR local currency | | 2030 | 47.3K | billion US$2010/yr OR local currency | | 2035 | 54.1K | billion US$2010/yr OR local currency | | 2040 | 61.3K | billion US$201…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `carbon price for EU`  
- route: `data_query` · plot: `False` · scope: variable=`Price|Carbon`, region=`EU`
- answer: ### Price|Carbon in EU Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **MUSE - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2015 | 0.00 | US$2010/t CO2 | | 2020 | 0.00 | US$2010/t CO2 | | 2025 | 0.00 | US$2010/t CO2 | | 2030 | 50.00 | US$2010/t CO2 | | 2035 | 107.00 | US$2010/t CO2 | | 2040 | 113.00 | US$2010/t CO2 | | 2045 | 122.00 | US$2010/t CO2 | | 2050 | 132.00 | US$2010/t CO2 | **GCAM-PR 5.3 - PR_WWH_CP** | Year | Value | Unit | |-…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `electricity price for India`  
- route: `data_query` · plot: `False` · scope: variable=`Price|Secondary Energy|Electricity`, region=`IND`, model=`gcam`
- answer: ### Price|Secondary Energy|Electricity in IND Scope: scenario `multiple`, model `gcam`, years `2005-2100` Unit: `US$2010/GJ or local currency/GJ` Answer: **gcam - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 19.68 | US$2010/GJ or local currency/GJ | | 2010 | 21.42 | US$2010/GJ or local currency/GJ | | 2015 | 21.35 | US$2010/GJ or local currency/GJ | | 2020 | 20.24 | US$2010/GJ or local currency/GJ | | 2025 | 20.11 | US$2010/GJ or local currency/GJ | | 2030 | 20.18 | US$…
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `energy price for China`  
- route: `data_query` · plot: `False` · scope: variable=`Price|Primary Energy|Biomass`, region=`CHN`, model=`gcam`
- answer: ### Price|Primary Energy|Biomass in CHN Scope: scenario `multiple`, model `gcam`, years `2005-2100` Unit: `US$2010/GJ or local currency/GJ` Answer: **gcam - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 5.94 | US$2010/GJ or local currency/GJ | | 2010 | 6.12 | US$2010/GJ or local currency/GJ | | 2015 | 6.22 | US$2010/GJ or local currency/GJ | | 2020 | 6.09 | US$2010/GJ or local currency/GJ | | 2025 | 6.16 | US$2010/GJ or local currency/GJ | | 2030 | 6.15 | US$2010/GJ or l…
- top link: EU-CHINA BRIDGE: https://iamparis.eu/results/eu-china-bridge (200)

**Q:** `trade for World`  
- route: `data_query` · plot: `False` · scope: variable=`Primary Energy|Secondary Energy Trade`, region=`World`
- answer: I could not find data for `Primary Energy|Secondary Energy Trade` in `World`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. variable `Primary Energy` (primary energy) 2. variable `Primary Energy|Biomass` (primary energy) 3. variable `Primary Energy|Biomass|Electricity` (primary energy, electricity) Reply with `1`, `2`, or `3`, or type the variable you want.
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

## 7 Transport/buildings/industry

**Q:** `transport final energy for World`  
- route: `data_query` · plot: `False` · scope: variable=`Price|Final Energy|Transportation|Electricity`, region=`World`, model=`gcam`
- answer: ### Price|Final Energy|Transportation|Electricity in World Scope: scenario `multiple`, model `gcam`, years `2005-2100` Unit: `US$2010/GJ or local currency/GJ` Answer: **gcam - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 34.50 | US$2010/GJ or local currency/GJ | | 2010 | 34.67 | US$2010/GJ or local currency/GJ | | 2015 | 33.65 | US$2010/GJ or local currency/GJ | | 2020 | 33.04 | US$2010/GJ or local currency/GJ | | 2025 | 32.78 | US$2010/GJ or local currency/GJ | | 2030 …
- top link: Transportation Transformation: https://iamparis.eu/results/ndc-aspects/transportation-transformation/policy_questions (200)

**Q:** `passenger transport demand for China`  
- route: `data_query` · plot: `False` · scope: variable=`Energy Service|Transportation|Passenger`, region=`CHN`, model=`gcam`
- answer: ### Energy Service|Transportation|Passenger in CHN Scope: scenario `multiple`, model `gcam`, years `2005-2100` Unit: `bn pkm/yr` Answer: **gcam - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 6.0K | bn pkm/yr | | 2010 | 8.6K | bn pkm/yr | | 2015 | 12.5K | bn pkm/yr | | 2020 | 16.5K | bn pkm/yr | | 2025 | 20.7K | bn pkm/yr | | 2030 | 25.1K | bn pkm/yr | | 2035 | 28.7K | bn pkm/yr | | 2040 | 31.8K | bn pkm/yr | | 2045 | 34.4K | bn pkm/yr | | 2050 | 36.8K | bn pkm/yr | | 20…
- top link: Transportation Transformation: https://iamparis.eu/results/ndc-aspects/transportation-transformation/policy_questions (200)

**Q:** `freight transport demand for India`  
- route: `data_query` · plot: `False` · scope: variable=`Energy Service|Transportation|Freight`, region=`IND`, model=`gcam`
- answer: ### Energy Service|Transportation|Freight in IND Scope: scenario `multiple`, model `gcam`, years `2005-2100` Unit: `bn tkm/yr` Answer: **gcam - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 912.32 | bn tkm/yr | | 2010 | 1.4K | bn tkm/yr | | 2015 | 1.8K | bn tkm/yr | | 2020 | 2.3K | bn tkm/yr | | 2025 | 2.9K | bn tkm/yr | | 2030 | 3.6K | bn tkm/yr | | 2035 | 4.3K | bn tkm/yr | | 2040 | 5.1K | bn tkm/yr | | 2045 | 5.9K | bn tkm/yr | | 2050 | 6.7K | bn tkm/yr | | 2055 | 7.5…
- top link: Transportation Transformation: https://iamparis.eu/results/ndc-aspects/transportation-transformation/policy_questions (200)

**Q:** `electric vehicle sales for EU`  
- route: `data_query` · plot: `False` · scope: variable=`Demand|Transportation|Road|Passenger|Battery Electric Car`, region=`EU`
- answer: I could not find data for `Demand|Transportation|Road|Passenger|Battery Electric Car` in `EU`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. region `Germany` 2. region `USA` (USA (United States)) 3. scenario `Baseline` (baseline) Reply with `1`, `2`, or `3`, or type the region you want.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `building floor area for World`  
- route: `data_query` · plot: `False` · scope: variable=`Energy Service|Commercial|Floor Space`, region=`World`
- answer: I could not find data for `Energy Service|Commercial|Floor Space` in `World`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. variable `Energy Service|Transportation|Freight` (transport) 2. variable `Energy Service|Transportation|Passenger` (transport) 3. region `EU27noUK` Reply with `1`, `2`, or `3`, or type the variable you want.
- top link: Buildings Transformation: https://iamparis.eu/results/ndc-aspects/buildings-transformation/policy_questions (200)

**Q:** `residential energy demand for China`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|BC|Energy|Demand|Residential and Commercial`, region=`CHN`
- answer: I could not find data for `Emissions|BC|Energy|Demand|Residential and Commercial` in `CHN`. Reason: the requested variable is unavailable in the current scope. Closest valid options: 1. variable `Emissions|C2F6` (emissions) 2. variable `Emissions|CH4` (methane emissions) 3. variable `Emissions|CH4|AFOLU` (methane emissions, land use) Reply with `1`, `2`, or `3`, or type the variable you want.
- top link: Buildings Transformation: https://iamparis.eu/results/ndc-aspects/buildings-transformation/policy_questions (200)

**Q:** `commercial building energy demand for EU`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|BC|Energy|Demand|Residential and Commercial`, region=`EU`
- answer: I could not find data for `Emissions|BC|Energy|Demand|Residential and Commercial` in `EU`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. variable `Emissions per capita|CO2|Energy` (emissions) 2. variable `Emissions/GDP|CO2|Energy` (emissions) 3. variable `Emissions|C2F6` (emissions) Reply with `1`, `2`, or `3`, or type the variable you want.
- top link: Buildings Transformation: https://iamparis.eu/results/ndc-aspects/buildings-transformation/policy_questions (200)

**Q:** `steel production for World`  
- route: `data_query` · plot: `False` · scope: variable=`Carbon Intensity|Production|Steel`, region=`World`, model=`WISEE-EDM`
- answer: ### Carbon Intensity|Production|Steel in World Scope: scenario `multiple`, model `WISEE-EDM`, years `2020-2050` Unit: `Mt CO2/Mt` Answer: **WISEE-EDM - Current Ambition** | Year | Value | Unit | |------|-------|------| | 2020 | 1.43 | Mt CO2/Mt | | 2030 | 1.26 | Mt CO2/Mt | | 2040 | 1.00 | Mt CO2/Mt | | 2050 | 0.67 | Mt CO2/Mt | **WISEE-EDM - 1.5 °C Domestic** | Year | Value | Unit | |------|-------|------| | 2020 | 1.43 | Mt CO2/Mt | | 2030 | 1.15 | Mt CO2/Mt | | 2040 | 0.69 | Mt CO2/Mt | | 205…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `cement production for India`  
- route: `data_query` · plot: `False` · scope: variable=`Production|Cement`, region=`IND`
- answer: I could not find data for `Production|Cement` in `IND`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. region `China` 2. region `EU-27` (EU-27 (European Union (EU-27))) 3. region `EU27` (EU27 (European Union (EU27))) Reply with `1`, `2`, or `3`, or type the region you want.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `chemical industry energy use for China`  
- route: `data_query` · plot: `False` · scope: variable=`Benchmarking|Industry|Energy intensity`, region=`CHN`, model=`China-MORE`
- answer: No data found for **Benchmarking|Industry|Energy intensity** in region `CHN` under scenario ``. Choose the variable: 1. `Benchmarking|Industry|CO2 intensity (net)` (industry) 2. `Benchmarking|Industry|CO2 intensity (gross)` (industry) Reply with a number (1-2), or `yes` for option 1.
- top link: China-MORE: https://iamparis.eu/models (200)

## 8 AFOLU/environment

**Q:** `forest land for Brazil`  
- route: `data_query` · plot: `False` · scope: variable=`Land Cover|Forest`, region=`Brazil`
- answer: I could not find data for `Land Cover|Forest` in `Brazil`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. region `ARG` (ARG (Argentina)) 2. region `BRA` (BRA (Brazil)) 3. region `CAN` (CAN (Canada)) Reply with `1`, `2`, or `3`, or type the region you want.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `cropland for World`  
- route: `data_query` · plot: `False` · scope: variable=`Land Cover|Cropland`, region=`World`, model=`gcam`
- answer: ### Land Cover|Cropland in World Scope: scenario `multiple`, model `gcam`, years `2005-2100` Unit: `million ha` Answer: **gcam - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 1.5K | million ha | | 2010 | 1.5K | million ha | | 2015 | 1.6K | million ha | | 2020 | 1.6K | million ha | | 2025 | 1.6K | million ha | | 2030 | 1.6K | million ha | | 2035 | 1.7K | million ha | | 2040 | 1.7K | million ha | | 2045 | 1.7K | million ha | | 2050 | 1.7K | million ha | | 2055 | 1.7K | mil…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `pasture area for India`  
- route: `data_query` · plot: `False` · scope: region=`IND`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `bioenergy crop area for EU`  
- route: `data_query` · plot: `False` · scope: region=`EU`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `food system emissions for World`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`World`
- answer: ### Emissions|CO2 in World Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr` Answer: **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2010 | 31.5K | Mt CO2/yr | | 2015 | 34.4K | Mt CO2/yr | | 2020 | 34.4K | Mt CO2/yr | | 2025 | 35.4K | Mt CO2/yr | | 2030 | 36.9K | Mt CO2/yr | | 2035 | 38.1K | Mt CO2/yr | | 2040 | 38.8K | Mt CO2/yr | | 2045 | 39.1K | Mt CO2/yr | | 2050 | 40.4K | Mt CO2/yr | **e3me - PR_CurPol_CP** | Year | Value | Unit | |--…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `agricultural water use for India`  
- route: `data_query` · plot: `False` · scope: region=`IND`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `water withdrawal for China`  
- route: `data_query` · plot: `False` · scope: region=`CHN`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `biodiversity loss indicator for World`  
- route: `data_query` · plot: `False` · scope: region=`World`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `nitrogen pollution from agriculture`  
- route: `general_qa` · plot: `False` · scope: —
- answer: To address nitrogen pollution from agriculture, you can explore models that focus on land use, agriculture, and environmental impacts. Here are some models from the IAM PARIS database that might be relevant: 1. **CLEWs**: This model integrates Climate, Land, Energy, and Water systems, which can help analyze the impact of agricultural practices on nitrogen pollution. 2. **GCAM**: The Global Change Assessment Model includes detailed representations of agriculture and land use, which can be used to…
- top link: AFOLU transformation: https://iamparis.eu/results/ndc-aspects/afolu-transformation/policy_questions (200)

**Q:** `livestock production for Brazil`  
- route: `data_query` · plot: `False` · scope: region=`Brazil`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

## 9 Model-scoped data

**Q:** `show World CO2 emissions from GCAM`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`World`, model=`gcam`
- answer: ### Emissions|CO2 in World Scope: scenario `multiple`, model `gcam`, years `2005-2100` Unit: `Mt CO2/yr` Answer: **gcam - Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 31.4K | Mt CO2/yr | | 2010 | 36.1K | Mt CO2/yr | | 2015 | 39.2K | Mt CO2/yr | | 2020 | 38.3K | Mt CO2/yr | | 2025 | 40.7K | Mt CO2/yr | | 2030 | 42.1K | Mt CO2/yr | | 2035 | 44.0K | Mt CO2/yr | | 2040 | 45.3K | Mt CO2/yr | | 2045 | 45.5K | Mt CO2/yr | | 2050 | 45.6K | Mt CO2/yr | | 2055 | 46.7K | Mt CO2/yr | …
- top link: GCAM: https://iamparis.eu/models (200)

**Q:** `show EU primary energy from IMAGE`  
- route: `data_query` · plot: `False` · scope: variable=`Primary Energy`, region=`EU`
- answer: ### Primary Energy in EU Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **42 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2005 | 79.33 | EJ/yr | | 2010 | 76.15 | EJ/yr | | 2015 | 70.57 | EJ/yr | | 2020 | 67.43 | EJ/yr | | 2025 | 64.31 | EJ/yr | | 2030 | 61.21 | EJ/yr | | 2035 | 59.02 | EJ/yr | | 2040 | 57.02 | EJ/yr | | 2045 | 55.35 | EJ/yr | **MUSE - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2015 | 66.78 | EJ/y | |…
- top link: EU-CHINA BRIDGE: https://iamparis.eu/results/eu-china-bridge (200)

**Q:** `GDP for China using GEM-E3`  
- route: `data_query` · plot: `False` · scope: variable=`GDP|MER`, region=`CHN`, model=`GEM-E3`
- answer: Note: no timeseries data for model `GEM-E3` in this slice; showing results across the models that do have it. ### GDP|MER in CHN Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2010 | 6.1K | billion US$2010/yr | | 2015 | 8.9K | billion US$2010/yr | | 2020 | 12.6K | billion US$2010/yr | | 2025 | 15.8K | billion US$2010/yr | | 2030 | 18.7K | billion US$2010/yr | | 2035 | 21.5K | billio…
- top link: GEM-E3: https://iamparis.eu/models (200)

**Q:** `electricity generation for India from POLES`  
- route: `data_query` · plot: `False` · scope: variable=`Secondary Energy|Electricity`, region=`IND`
- answer: ### Secondary Energy|Electricity in IND Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `EJ/yr` Answer: **42 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 1.76 | EJ/yr | | 2010 | 2.59 | EJ/yr | | 2015 | 3.70 | EJ/yr | | 2020 | 4.90 | EJ/yr | | 2025 | 6.43 | EJ/yr | | 2030 | 8.16 | EJ/yr | | 2035 | 10.05 | EJ/yr | | 2040 | 12.03 | EJ/yr | | 2045 | 14.10 | EJ/yr | **42 - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 2005 | 1.76 | …
- top link: POLES: https://iamparis.eu/models (200)

**Q:** `emissions for Japan from AIM`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2|Energy and Industrial Processes`, region=`Japan`, scenario=`Baseline`, model=`gcam`
- answer: ### Emissions|CO2|Energy and Industrial Processes in Japan Scope: scenario `Baseline`, model `gcam`, years `2005-2100` Unit: `Mt CO2/yr` Answer: **gcam - Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 1.3K | Mt CO2/yr | | 2010 | 1.2K | Mt CO2/yr | | 2015 | 1.2K | Mt CO2/yr | | 2020 | 1.2K | Mt CO2/yr | | 2025 | 1.2K | Mt CO2/yr | | 2030 | 1.1K | Mt CO2/yr | | 2035 | 1.0K | Mt CO2/yr | | 2040 | 979.25 | Mt CO2/yr | | 2045 | 901.25 | Mt CO2/yr | | 2050 | 836.38 | Mt CO2/yr | |…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `does COFFEE have final energy data for Brazil?`  
- route: `data_query` · plot: `False` · scope: variable=`Final Energy`, region=`Brazil`
- answer: I could not find data for `Final Energy` in `Brazil`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. variable `Final Energy|Industry|Chemicals|Ammonia` (final energy use, industry) 2. variable `Final Energy|Industry|Chemicals|Ammonia|Gases` (final energy use, industry, gas) 3. variable `Final Energy|Industry|Chemicals|Ammonia|Hydrogen` (final energy use, industry, hydrogen) Reply with `1`, `2`, or `3`, or type the variable you want.
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `what variables are available for IMAGE in EU?`  
- route: `data_query` · plot: `False` · scope: —
- answer: I can work with these variables: - 'Economy|Residential & Commercial|Number of buildings - Agricultural Demand - Agricultural Demand|Crops - Agricultural Demand|Crops|Energy - Agricultural Demand|Crops|Feed - Agricultural Demand|Crops|Food - Agricultural Demand|Crops|Other - Agricultural Demand|Livestock - Agricultural Demand|Livestock|Food - Agricultural Demand|Livestock|Other - Agricultural Production - Annual Refurbishment Rate|Commercial and 1144 more Try queries like 'Capacity|Electricity|S…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `plot GCAM CO2 emissions for World`  
- route: `data_plotting` · plot: `True` · scope: variable=`Emissions|CO2`, region=`World`, model=`GCAM`
- caption: Showing CO2 Emissions in World across available scenarios.
- answer: Showing CO2 Emissions in World across available scenarios. Reply with a scenario, model, region, or year to narrow the answer.
- top link: GCAM: https://iamparis.eu/models (200)

**Q:** `compare IMAGE and GCAM for World emissions`  
- route: `data_plotting` · plot: `False` · scope: variable=`Emissions|CO2`, region=`World`, model=`GCAM`
- answer: Could not identify enough variables to compare. Found: [] Reply with a scenario, model, region, or year to narrow the answer.
- top link: GCAM: https://iamparis.eu/models (200)

**Q:** `show data from a model that covers land use`  
- route: `data_query` · plot: `False` · scope: —
- answer: ### Models covering Land Use 6 model(s) of 90 report at least one land use variable in IAM PARIS: GCAM 7.0, GCAM-PR 5.3, GCAM-PR 7.0, GEMINI-E3 7.0, gcam, gemini_e3 Ask for a specific model (e.g. `tell me about GCAM 7.0`) or a data query (e.g. `land use emissions for Europe`) to go deeper.
- top link: AFOLU transformation: https://iamparis.eu/results/ndc-aspects/afolu-transformation/policy_questions (200)

## 10 Years/time filters

**Q:** `World CO2 emissions in 2025`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`World`, start_year=`2025`, end_year=`2025`
- answer: ### Emissions|CO2 in World Scope: scenario `multiple`, model `multiple`, years `2025` Unit: `Mt CO2/yr` Answer: **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2025 | 35.4K | Mt CO2/yr | **e3me - PR_CurPol_CP** | Year | Value | Unit | |------|-------|------| | 2025 | 35.4K | Mt CO2/yr | **e3me - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 2025 | 35.4K | Mt CO2/yr | **e3me - PR_NDC_CP** | Year | Value | Unit | |------|-------|------| | 2025 | 34.6K | …
- top link: Global impacts of NDCs: https://iamparis.eu/results/ndc-aspects/global-impacts-of-ndcs/policy_questions (200)

**Q:** `EU GDP in 2040`  
- route: `data_query` · plot: `False` · scope: variable=`GDP|MER`, region=`EU`, start_year=`2040`, end_year=`2040`
- answer: ### GDP|MER in EU Scope: scenario `multiple`, model `multiple`, years `2040` Unit: `multiple` Answer: **E3ME 6.1 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2040 | 25.6K | billion US$2010/yr | **E4SMA-EU-TIMES 1.0 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2040 | 25.7K | billion US$2010/yr | **GCAM-PR 5.3 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2040 | 25.6K | billion US$2010/yr OR local currency | **GEMINI-E3 7.0 - PR_WWH_CP** | Ye…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `China population in 2100`  
- route: `data_query` · plot: `False` · scope: variable=`Population`, region=`CHN`
- answer: ### Population in CHN Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `million` Answer: **42 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 1.3K | million | | 2010 | 1.4K | million | | 2015 | 1.4K | million | | 2020 | 1.4K | million | | 2025 | 1.4K | million | | 2030 | 1.4K | million | | 2035 | 1.4K | million | | 2040 | 1.4K | million | | 2045 | 1.4K | million | **42 - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 2005 | 1.3K | m…
- top link: EU-CHINA BRIDGE: https://iamparis.eu/results/eu-china-bridge (200)

**Q:** `India final energy between 2030 and 2060`  
- route: `data_query` · plot: `False` · scope: variable=`Final Energy`, region=`IND`, start_year=`2030`, end_year=`2030`
- answer: ### Final Energy in IND Scope: scenario `multiple`, model `multiple`, years `2030` Unit: `multiple` Answer: **42 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2030 | 38.85 | EJ/yr | **42 - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 2030 | 38.85 | EJ/yr | **42 - PR_NDC_EI** | Year | Value | Unit | |------|-------|------| | 2030 | 35.89 | EJ/yr | **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2030 | 40.14 | EJ/yr | **e3me - PR_Cur…
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `Brazil emissions after 2040`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`Brazil`, model=`gcam`, start_year=`2041`
- answer: ### Emissions|CO2 in Brazil Scope: scenario `multiple`, model `gcam`, years `2041-latest available` Unit: `Mt CO2/yr` Answer: **gcam - Baseline** | Year | Value | Unit | |------|-------|------| | 2045 | 773.21 | Mt CO2/yr | | 2050 | 744.89 | Mt CO2/yr | | 2055 | 682.21 | Mt CO2/yr | | 2060 | 663.75 | Mt CO2/yr | | 2065 | 655.90 | Mt CO2/yr | | 2070 | 640.56 | Mt CO2/yr | | 2075 | 628.46 | Mt CO2/yr | | 2080 | 633.57 | Mt CO2/yr | | 2085 | 642.13 | Mt CO2/yr | | 2090 | 637.92 | Mt CO2/yr | | 2095…
- top link: Global impacts of NDCs: https://iamparis.eu/results/ndc-aspects/global-impacts-of-ndcs/policy_questions (200)

**Q:** `Japan electricity generation before 2035`  
- route: `data_query` · plot: `False` · scope: variable=`Secondary Energy|Electricity`, region=`Japan`
- answer: I could not find data for `Secondary Energy|Electricity` in `Japan`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. region `AGO` (AGO (Angola)) 2. region `ARE` (ARE (United Arab Emirates)) 3. region `ARG` (ARG (Argentina)) Reply with `1`, `2`, or `3`, or type the region you want.
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `World primary energy from 2020 to 2050`  
- route: `data_query` · plot: `False` · scope: variable=`Primary Energy`, region=`World`, start_year=`2020`, end_year=`2050`
- answer: ### Primary Energy in World Scope: scenario `multiple`, model `multiple`, years `2020-2050` Unit: `multiple` Answer: **42 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2020 | 603.42 | EJ/yr | | 2025 | 650.43 | EJ/yr | | 2030 | 688.45 | EJ/yr | | 2035 | 723.31 | EJ/yr | | 2040 | 749.64 | EJ/yr | | 2045 | 770.99 | EJ/yr | **42 - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 2020 | 602.06 | EJ/yr | | 2025 | 639.58 | EJ/yr | | 2030 | 670.30 | EJ/yr | | 2035 | 6…
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `EU carbon price in 2030 and 2050`  
- route: `data_query` · plot: `False` · scope: variable=`Price|Carbon`, region=`EU`, start_year=`2030`, end_year=`2030`
- answer: ### Price|Carbon in EU Scope: scenario `multiple`, model `multiple`, years `2030` Unit: `multiple` Answer: **MUSE - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2030 | 50.00 | US$2010/t CO2 | **GCAM-PR 5.3 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2030 | 26.38 | US$2010/t CO2 | **GEMINI-E3 7.0 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2030 | 45.09 | US$2010/t CO2 | **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| …
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `China methane emissions around 2040`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CH4`, region=`CHN`, start_year=`2040`, end_year=`2040`
- answer: ### Emissions|CH4 in CHN Scope: scenario `multiple`, model `multiple`, years `2040` Unit: `Mt CH4/yr` Answer: **gcam - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2040 | 57.78 | Mt CH4/yr | **gcam - PR_CurPol_CP** | Year | Value | Unit | |------|-------|------| | 2040 | 41.50 | Mt CH4/yr | **gcam - PR_CurPol_CPO** | Year | Value | Unit | |------|-------|------| | 2040 | 42.50 | Mt CH4/yr | **gcam - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 2040 | 41.24 …
- top link: EU-CHINA BRIDGE: https://iamparis.eu/results/eu-china-bridge (200)

**Q:** `India GDP by mid-century`  
- route: `data_query` · plot: `False` · scope: variable=`GDP|MER`, region=`IND`
- answer: ### GDP|MER in IND Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2010 | 1.7K | billion US$2010/yr | | 2015 | 2.3K | billion US$2010/yr | | 2020 | 3.5K | billion US$2010/yr | | 2025 | 4.9K | billion US$2010/yr | | 2030 | 6.2K | billion US$2010/yr | | 2035 | 7.7K | billion US$2010/yr | | 2040 | 9.5K | billion US$2010/yr | | 2045 | 11.5K | billion US$2010/yr | | 2050 | 13.8K | billion…
- top link: PowerPlan: https://iamparis.eu/models (200)

## 11 Scenario tests

**Q:** `World CO2 emissions in baseline scenarios`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`World`, scenario=`Baseline`
- answer: ### Emissions|CO2 in World Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr` Answer: **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2010 | 31.5K | Mt CO2/yr | | 2015 | 34.4K | Mt CO2/yr | | 2020 | 34.4K | Mt CO2/yr | | 2025 | 35.4K | Mt CO2/yr | | 2030 | 36.9K | Mt CO2/yr | | 2035 | 38.1K | Mt CO2/yr | | 2040 | 38.8K | Mt CO2/yr | | 2045 | 39.1K | Mt CO2/yr | | 2050 | 40.4K | Mt CO2/yr | **e3me - Unharmonised baseline** | Year | Value | U…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `EU electricity generation under current policies`  
- route: `data_query` · plot: `False` · scope: variable=`Secondary Energy|Electricity`, region=`EU`, scenario=`Current Policies`
- answer: ### Secondary Energy|Electricity in EU Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `EJ/yr` Answer: **42 - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 2005 | 9.97 | EJ/yr | | 2010 | 10.21 | EJ/yr | | 2015 | 9.82 | EJ/yr | | 2020 | 9.90 | EJ/yr | | 2025 | 10.03 | EJ/yr | | 2030 | 10.39 | EJ/yr | | 2035 | 10.92 | EJ/yr | | 2040 | 11.49 | EJ/yr | | 2045 | 12.19 | EJ/yr | **e3me - PR_CurPol_CP** | Year | Value | Unit | |------|-------|------| | 2010 | 11…
- top link: EU-CHINA BRIDGE: https://iamparis.eu/results/eu-china-bridge (200)

**Q:** `India final energy in net zero scenarios`  
- route: `data_query` · plot: `False` · scope: variable=`Final Energy`, region=`IND`, scenario=`NZE_Bench_L`
- answer: I could not find data for `Final Energy` in `IND` under `NZE_Bench_L`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. region `EU27` (EU27 (European Union (EU27))) 2. scenario `CP_EI` 3. scenario `NDC_EI` (NDC) Reply with `1`, `2`, or `3`, or type the region you want.
- top link: Tech-constrained EU pathways to net zero: https://iamparis.eu/results/iam-compact/tech-constrained-eu-pathways-to-net-zero/policy_questions (200)

**Q:** `China GDP in low ambition scenarios`  
- route: `data_query` · plot: `False` · scope: variable=`GDP|MER`, region=`CHN`
- answer: ### GDP|MER in CHN Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2010 | 6.1K | billion US$2010/yr | | 2015 | 8.9K | billion US$2010/yr | | 2020 | 12.6K | billion US$2010/yr | | 2025 | 15.8K | billion US$2010/yr | | 2030 | 18.7K | billion US$2010/yr | | 2035 | 21.5K | billion US$2010/yr | | 2040 | 23.9K | billion US$2010/yr | | 2045 | 25.9K | billion US$2010/yr | | 2050 | 27.8K | bi…
- top link: Post-Glasgow targets: https://iamparis.eu/results/iam-compact/post-glasgow-targets/policy_questions (200)

**Q:** `Brazil emissions in high ambition mitigation pathways`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`Brazil`, model=`gcam`
- answer: ### Emissions|CO2 in Brazil Scope: scenario `multiple`, model `gcam`, years `2005-2100` Unit: `Mt CO2/yr` Answer: **gcam - Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 948.17 | Mt CO2/yr | | 2010 | 1.0K | Mt CO2/yr | | 2015 | 1.4K | Mt CO2/yr | | 2020 | 830.38 | Mt CO2/yr | | 2025 | 737.11 | Mt CO2/yr | | 2030 | 698.82 | Mt CO2/yr | | 2035 | 744.46 | Mt CO2/yr | | 2040 | 750.47 | Mt CO2/yr | | 2045 | 773.21 | Mt CO2/yr | | 2050 | 744.89 | Mt CO2/yr | | 2055 | 682.21 | Mt C…
- top link: Cost of capital evolution effects on regional mitigation pathways: https://iamparis.eu/results/iam-compact/cost-of-capital-evolution-effects-on-regional-mitigation-pathways/policy_questions (200)

**Q:** `Japan primary energy in Paris-aligned scenarios`  
- route: `data_query` · plot: `False` · scope: variable=`Primary Energy`, region=`Japan`
- answer: I could not find data for `Primary Energy` in `Japan`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. region `AGO` (AGO (Angola)) 2. region `ARE` (ARE (United Arab Emirates)) 3. region `ARG` (ARG (Argentina)) Reply with `1`, `2`, or `3`, or type the region you want.
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `compare current policies and baseline for World CO2`  
- route: `data_plotting` · plot: `True` · scope: variable=`Emissions|CO2`, region=`World`, scenario=`Current Policies`
- caption: Showing CO2 Emissions in World across available scenarios.
- answer: Showing CO2 Emissions in World across available scenarios. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `which scenarios are available for EU emissions?`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`EU`
- answer: ### What I can help you with - **Models:** 74 available. Examples: 42, AIM/Enduse India 3.3, ALADIN, ATOM. - **Variables:** 1152 available. Examples: Benchmarking|Buildings|CO2 intensity, Benchmarking|Industry|CO2 intensity (gross), Benchmarking|Industry|CO2 intensity (net), Benchmarking|Transport|CO2 intensity. - **Scenarios:** 101 available. Examples: 1.5 TECH, 1.5 °C DRI-trade, 1.5 °C Domestic, BAU. - **Regions:** 180 available. Examples: AGO, ARE, ARG, AUS. - **Workspaces:** 20 available. As…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `show scenarios related to NDCs`  
- route: `data_query` · plot: `False` · scope: —
- answer: I found scenarios matching your request like total_FiT, total_NEM and total_NEMBESS. You can plot variables for any of these scenarios. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Global impacts of NDCs: https://iamparis.eu/results/ndc-aspects/global-impacts-of-ndcs/policy_questions (200)

**Q:** `show scenarios related to delayed action`  
- route: `data_query` · plot: `False` · scope: —
- answer: I found scenarios matching your request like total_FiT, total_NEM and total_NEMBESS. You can plot variables for any of these scenarios. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

## 12 Plots

**Q:** `plot greenhouse gas emissions for EU`  
- route: `data_plotting` · plot: `True` · scope: variable=`Emissions|Kyoto Gases`, region=`EU`
- caption: Showing Kyoto Gases Emissions in EU across available scenarios.
- answer: Showing Kyoto Gases Emissions in EU across available scenarios. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `make a chart of solar electricity for India`  
- route: `data_plotting` · plot: `True` · scope: variable=`Capacity|Electricity|Solar`, region=`IND`
- caption: Showing Solar Capacity in IND across available scenarios.
- answer: Showing Solar Capacity in IND across available scenarios. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `visualize carbon price for World`  
- route: `data_plotting` · plot: `True` · scope: variable=`Price|Carbon`, region=`World`
- caption: Showing Carbon in World across available scenarios.
- answer: Showing Carbon in World across available scenarios. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `draw a line chart for China GDP`  
- route: `data_plotting` · plot: `True` · scope: variable=`GDP|MER`, region=`CHN`
- caption: Showing MER in CHN across available scenarios.
- answer: Showing MER in CHN across available scenarios. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `graph Brazil population projections`  
- route: `data_plotting` · plot: `False` · scope: variable=`Population`, region=`Brazil`
- answer: No data found for **Population** in region `Brazil` under scenario `None`. - Closest regions: `AGO`, `ARE`, `ARG` - Closest scenarios: `PR_Baseline`, `PR_CurPol_EI`, `PR_NDC_EI` Reply with the option you want to use.
- top link: Results: https://iamparis.eu/results (200)

**Q:** `create a figure for wind electricity in EU`  
- route: `data_query` · plot: `False` · scope: variable=`Secondary Energy|Electricity|Wind`, region=`EU`
- answer: ### Secondary Energy|Electricity|Wind in EU Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **42 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2005 | 0.25 | Mt CO2/yr | | 2010 | 0.54 | Mt CO2/yr | | 2015 | 1.09 | Mt CO2/yr | | 2020 | 1.32 | Mt CO2/yr | | 2025 | 1.64 | Mt CO2/yr | | 2030 | 2.06 | Mt CO2/yr | | 2035 | 2.36 | Mt CO2/yr | | 2040 | 2.69 | Mt CO2/yr | | 2045 | 3.06 | Mt CO2/yr | **E3ME 6.1 - PR_WWH_CP** | Year | Value | Unit |…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `plot final energy in buildings for China`  
- route: `data_plotting` · plot: `False` · scope: variable=`Final Energy|Buildings|Residential`, region=`CHN`
- answer: No data found for variable 'Final Energy|Buildings|Residential'. Recommended regions: EU (European Union) Recommended scenarios: PR_WWH_CP Reply with a scenario, model, region, or year to narrow the answer.
- top link: Buildings Transformation: https://iamparis.eu/results/ndc-aspects/buildings-transformation/policy_questions (200)

**Q:** `chart methane emissions for World`  
- route: `data_plotting` · plot: `True` · scope: variable=`Emissions|CH4`, region=`World`
- caption: Showing CH4 Emissions in World across available scenarios.
- answer: Showing CH4 Emissions in World across available scenarios. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `visualize primary energy by fuel for World`  
- route: `data_plotting` · plot: `True` · scope: variable=`Primary Energy`, region=`World`
- caption: Showing Primary Energy in World across available scenarios.
- answer: Showing Primary Energy in World across available scenarios. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `make a comparison plot for EU and India CO2 emissions`  
- route: `data_plotting` · plot: `True` · scope: variable=`Emissions|CO2`, region=`IND`
- caption: Showing CO2 Emissions.
- answer: Showing CO2 Emissions. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

## 13 Links/navigation

**Q:** `send me the IAM PARIS homepage`  
- route: `general_qa` · plot: `False` · scope: —
- answer: You can visit the IAM PARIS homepage at [https://iamparis.eu/](https://iamparis.eu/).
- top link: search: IAM Validation UI

**Q:** `where is the model documentation page?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request:
- top link: Home: https://iamparis.eu/ (200)

**Q:** `open the scenario explorer`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request: If the direct detail page is not available, open the Application Library and search for: AR6 Scenario Explorer and Database.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `where can I browse available applications?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: You can browse available applications and models on the IAM PARIS website. Visit [https://iamparis.eu/results](https://iamparis.eu/results) to explore detailed data and model information.

**Q:** `give me the policy catalogue link`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request:
- top link: Policy Catalogue Interactive Explorer: https://iamparis.eu/datastories/policyCatalog (200)

**Q:** `where can I find NDC ASPECTS results?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request:
- top link: NDC ASPECTS: https://iamparis.eu/results/ndc-aspects (200)

**Q:** `where can I find IAM COMPACT outputs?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: You can find IAM COMPACT outputs on the IAM PARIS website. Visit [https://iamparis.eu/results](https://iamparis.eu/results) for detailed data access and to explore the outputs of various Integrated Assessment Models, including IAM COMPACT if available.
- top link: IAM COMPACT: https://iamparis.eu/results/iam-compact (200)

**Q:** `link me to transport transformation results`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request:
- top link: Transportation Transformation: https://iamparis.eu/results/ndc-aspects/transportation-transformation/policy_questions (200)

**Q:** `link me to buildings transformation results`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request:
- top link: Buildings Transformation: https://iamparis.eu/results/ndc-aspects/buildings-transformation/policy_questions (200)

**Q:** `link me to AFOLU results`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request:
- top link: AFOLU transformation: https://iamparis.eu/results/ndc-aspects/afolu-transformation/policy_questions (200)

**Q:** `where are industrial transformation results?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request:
- top link: Industrial transformation: https://iamparis.eu/results/ndc-aspects/industrial-transformation/policy_questions (200)

**Q:** `where can I read project methodology?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: You can read about the project methodology on the IAM PARIS website. Visit the [IAM PARIS Methodology page](https://iamparis.eu/methodology) for detailed information on the methodologies used in the various models and analyses.

## 19 Ambiguous/short

**Q:** `solar`  
- route: `data_query` · plot: `False` · scope: variable=`Capacity|Electricity|Solar`
- answer: ### Capacity|Electricity|Solar Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `GW` Answer: **E4SMA-EU-TIMES 1.0 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2010 | 30.73 | GW | | 2020 | 113.75 | GW | | 2025 | 137.49 | GW | | 2030 | 303.52 | GW | | 2040 | 303.62 | GW | | 2050 | 350.83 | GW | **GCAM-PR 5.3 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2005 | 0.59 | GW | | 2010 | 9.41 | GW | | 2015 | 43.87 | GW | | 2020 | 221.29 | GW | | 2…
- top link: Results: https://iamparis.eu/results (200)

**Q:** `wind`  
- route: `data_query` · plot: `False` · scope: variable=`Capacity Additions|Electricity|Wind`, region=`IND`, model=`gcam`
- answer: ### Capacity Additions|Electricity|Wind in IND Scope: scenario `multiple`, model `gcam`, years `2005-2100` Unit: `GW/yr` Answer: **gcam - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 0.00 | GW/yr | | 2010 | 0.00 | GW/yr | | 2015 | 0.00 | GW/yr | | 2020 | 3.15 | GW/yr | | 2025 | 3.29 | GW/yr | | 2030 | 3.89 | GW/yr | | 2035 | 4.79 | GW/yr | | 2040 | 6.66 | GW/yr | | 2045 | 8.49 | GW/yr | | 2050 | 14.01 | GW/yr | | 2055 | 12.32 | GW/yr | | 2060 | 12.32 | GW/yr | | 2065 | …
- top link: Results: https://iamparis.eu/results (200)

**Q:** `coal`  
- route: `data_query` · plot: `False` · scope: region=`COL`
- answer: Based on your wording, I think you may mean one of these for `COL (Colombia)`. Choose the variable: 1. `Secondary Energy|Electricity|Coal` (electricity output) 2. `Primary Energy|Coal|w/o CCS` (primary energy) 3. `Capacity|Electricity|Coal|w/ CCS` (power capacity, electricity) Reply with a number (1-3), or `yes` for option 1.
- top link: Results: https://iamparis.eu/results (200)

**Q:** `GDP`  
- route: `data_query` · plot: `False` · scope: variable=`GDP|MER`
- answer: ### GDP|MER Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **E3ME 6.1 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2010 | 17.0K | billion US$2010/yr | | 2015 | 18.0K | billion US$2010/yr | | 2020 | 19.7K | billion US$2010/yr | | 2025 | 20.9K | billion US$2010/yr | | 2030 | 22.4K | billion US$2010/yr | | 2035 | 23.8K | billion US$2010/yr | | 2040 | 25.6K | billion US$2010/yr | | 2045 | 27.7K | billion US$2010/yr | | 2050 | 29.9K | billi…
- top link: Results: https://iamparis.eu/results (200)

**Q:** `population`  
- route: `data_query` · plot: `False` · scope: variable=`Population`
- answer: ### Population Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **42 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2005 | 495.48 | EJ/yr | | 2010 | 503.87 | EJ/yr | | 2015 | 507.49 | EJ/yr | | 2020 | 512.40 | EJ/yr | | 2025 | 516.12 | EJ/yr | | 2030 | 519.09 | EJ/yr | | 2035 | 521.63 | EJ/yr | | 2040 | 523.28 | EJ/yr | | 2045 | 523.72 | EJ/yr | **E4SMA-EU-TIMES 1.0 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2010 | 503…
- top link: Results: https://iamparis.eu/results (200)

**Q:** `transport`  
- route: `data_query` · plot: `False` · scope: variable=`Energy Service|Transportation|Freight`
- answer: ### Energy Service|Transportation|Freight Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **GCAM-PR 5.3 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2005 | 3.1K | bn tkm/yr | | 2010 | 3.0K | bn tkm/yr | | 2015 | 2.9K | bn tkm/yr | | 2020 | 3.1K | bn tkm/yr | | 2025 | 3.2K | bn tkm/yr | | 2030 | 3.3K | bn tkm/yr | | 2035 | 3.4K | bn tkm/yr | | 2040 | 3.6K | bn tkm/yr | | 2045 | 3.8K | bn tkm/yr | | 2050 | 4.0K | bn tkm/yr | **GCAM 7.0 - …
- top link: Transportation Transformation: https://iamparis.eu/results/ndc-aspects/transportation-transformation/policy_questions (200)

**Q:** `water`  
- route: `data_query` · plot: `False` · scope: region=`IRL`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Results: https://iamparis.eu/results (200)

**Q:** `agriculture`  
- route: `general_qa` · plot: `False` · scope: —
- answer: It seems like you're interested in agriculture within the context of IAM PARIS models. Here's how agriculture is typically represented in integrated assessment models (IAMs): ### Agriculture in IAMs 1. **Land Use and Crop Production**: - Models often include major land cover types such as forests, grasslands, and agricultural lands. - Agricultural land is typically divided into different crop types, such as maize, rice, and wheat. - Crop production can be influenced by factors like climate condi…
- top link: AFOLU transformation: https://iamparis.eu/results/ndc-aspects/afolu-transformation/policy_questions (200)

**Q:** `models`  
- route: `data_query` · plot: `False` · scope: —
- answer: I couldn't match that to a known model. Try `list models` to see available options.
- top link: Model Comparison: https://iamparis.eu/models/comparison (200)

**Q:** `documentation`  
- route: `general_qa` · plot: `False` · scope: —
- answer: For detailed documentation and data access related to IAM PARIS models and results, please visit the [IAM PARIS Results page](https://iamparis.eu/results). This page provides comprehensive information on the models, scenarios, and data available in the IAM PARIS database.

## 20 Typos/noisy

**Q:** `grenhouse gas emisions for eruope`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|Kyoto Gases|AFOLU`, region=`EU`
- answer: I could not find data for `Emissions|Kyoto Gases|AFOLU` in `EU`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. variable `Emissions per capita|CO2|Energy` (emissions) 2. variable `Emissions/GDP|CO2|Energy` (emissions) 3. variable `Emissions|C2F6` (emissions) Reply with `1`, `2`, or `3`, or type the variable you want.
- top link: AFOLU transformation: https://iamparis.eu/results/ndc-aspects/afolu-transformation/policy_questions (200)

**Q:** `methan emisions wrld`  
- route: `data_query` · plot: `False` · scope: region=`World`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Results: https://iamparis.eu/results (200)

**Q:** `solr electrcity india`  
- route: `data_query` · plot: `False` · scope: region=`IND`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Results: https://iamparis.eu/results (200)

**Q:** `gdp ppp chna`  
- route: `data_query` · plot: `False` · scope: variable=`GDP|MER`, region=`China`
- answer: I could not find data for `GDP|MER` in `China`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. region `ARG` (ARG (Argentina)) 2. region `AUS` (AUS (Australia)) 3. region `AUT` (AUT (Austria)) Reply with `1`, `2`, or `3`, or type the region you want.
- top link: EU-CHINA BRIDGE: https://iamparis.eu/results/eu-china-bridge (200)

**Q:** `populaton brazl 2050`  
- route: `data_query` · plot: `False` · scope: region=`Brazil`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Results: https://iamparis.eu/results (200)

**Q:** `curent policie co2 eu`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`EU`
- answer: ### Emissions|CO2 in EU Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr` Answer: **E3ME 6.1 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2010 | 4.3K | Mt CO2/yr | | 2015 | 4.0K | Mt CO2/yr | | 2020 | 3.4K | Mt CO2/yr | | 2025 | 3.2K | Mt CO2/yr | | 2030 | 3.0K | Mt CO2/yr | | 2035 | 2.9K | Mt CO2/yr | | 2040 | 2.7K | Mt CO2/yr | | 2045 | 2.5K | Mt CO2/yr | | 2050 | 2.4K | Mt CO2/yr | **E4SMA-EU-TIMES 1.0 - PR_WWH_CP** | Year | Value | Unit | |-…
- top link: Implications of EU steel industry relocation: https://iamparis.eu/results/iam-compact/implications-of-eu-steel-industry-relocation/policy_questions (200)

**Q:** `basline primry energy wrld`  
- route: `data_query` · plot: `False` · scope: variable=`Primary Energy`, region=`World`, scenario=`Baseline`
- answer: ### Primary Energy in World Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **42 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 481.23 | EJ/yr | | 2010 | 538.79 | EJ/yr | | 2015 | 571.32 | EJ/yr | | 2020 | 603.42 | EJ/yr | | 2025 | 650.43 | EJ/yr | | 2030 | 688.45 | EJ/yr | | 2035 | 723.31 | EJ/yr | | 2040 | 749.64 | EJ/yr | | 2045 | 770.99 | EJ/yr | **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2010 | …
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `plz plot wind eletricity europe`  
- route: `data_plotting` · plot: `True` · scope: variable=`Capacity Additions|Electricity|Wind`, region=`EU`
- caption: Showing Wind in EU across available scenarios.
- answer: Showing Wind in EU across available scenarios. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `wher is model documntation`  
- route: `general_qa` · plot: `False` · scope: —
- answer: You can find the model documentation and detailed information on the IAM PARIS website. Visit [IAM PARIS Models](https://iamparis.eu/models) for comprehensive documentation on each model available in the IAM PARIS database.

**Q:** `transport reslts link`  
- route: `data_query` · plot: `False` · scope: variable=`Energy Service|Transportation|Freight`
- answer: ### Energy Service|Transportation|Freight Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **GCAM-PR 5.3 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2005 | 3.1K | bn tkm/yr | | 2010 | 3.0K | bn tkm/yr | | 2015 | 2.9K | bn tkm/yr | | 2020 | 3.1K | bn tkm/yr | | 2025 | 3.2K | bn tkm/yr | | 2030 | 3.3K | bn tkm/yr | | 2035 | 3.4K | bn tkm/yr | | 2040 | 3.6K | bn tkm/yr | | 2045 | 3.8K | bn tkm/yr | | 2050 | 4.0K | bn tkm/yr | **GCAM 7.0 - …
- top link: Transportation Transformation: https://iamparis.eu/results/ndc-aspects/transportation-transformation/policy_questions (200)

## 21 No-data/impossible

**Q:** `CO2 emissions for Gotham`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`
- answer: ### Emissions|CO2 Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr` Answer: **E3ME 6.1 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2010 | 4.3K | Mt CO2/yr | | 2015 | 4.0K | Mt CO2/yr | | 2020 | 3.4K | Mt CO2/yr | | 2025 | 3.2K | Mt CO2/yr | | 2030 | 3.0K | Mt CO2/yr | | 2035 | 2.9K | Mt CO2/yr | | 2040 | 2.7K | Mt CO2/yr | | 2045 | 2.5K | Mt CO2/yr | | 2050 | 2.4K | Mt CO2/yr | **E4SMA-EU-TIMES 1.0 - PR_WWH_CP** | Year | Value | Unit | |------|…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `GDP for Middle Earth`  
- route: `data_query` · plot: `False` · scope: variable=`GDP|MER`
- answer: ### GDP|MER Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **E3ME 6.1 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2010 | 17.0K | billion US$2010/yr | | 2015 | 18.0K | billion US$2010/yr | | 2020 | 19.7K | billion US$2010/yr | | 2025 | 20.9K | billion US$2010/yr | | 2030 | 22.4K | billion US$2010/yr | | 2035 | 23.8K | billion US$2010/yr | | 2040 | 25.6K | billion US$2010/yr | | 2045 | 27.7K | billion US$2010/yr | | 2050 | 29.9K | billi…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `electricity generation for Atlantis`  
- route: `data_query` · plot: `False` · scope: variable=`Secondary Energy|Electricity`
- answer: ### Secondary Energy|Electricity Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **OSeMOSYS-GR - BAU** | Year | Value | Unit | |------|-------|------| | 2025 | 0.23 | EJ/yr | | 2030 | 0.28 | EJ/yr | | 2035 | 0.39 | EJ/yr | | 2040 | 0.50 | EJ/yr | | 2045 | 0.58 | EJ/yr | | 2050 | 0.61 | EJ/yr | **OSeMOSYS-GR - BAU_CL1** | Year | Value | Unit | |------|-------|------| | 2025 | 0.22 | EJ/yr | | 2030 | 0.27 | EJ/yr | | 2035 | 0.36 | EJ/yr | | 2040 | 0.46 | EJ…
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `population for Mars in 2050`  
- route: `data_query` · plot: `False` · scope: variable=`Population`, start_year=`2050`, end_year=`2050`
- answer: ### Population Scope: scenario `multiple`, model `multiple`, years `2050` Unit: `multiple` Answer: **42 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| **E4SMA-EU-TIMES 1.0 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2050 | 523.71 | million | **MUSE - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2050 | 524.21 | million | **GCAM-PR 5.3 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2050 | 521.20 | million | **GEMINI-E3 7.0 - PR_W…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `carbon price for Hogwarts`  
- route: `data_query` · plot: `False` · scope: variable=`Price|Carbon`
- answer: ### Price|Carbon Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **MUSE - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2015 | 0.00 | US$2010/t CO2 | | 2020 | 0.00 | US$2010/t CO2 | | 2025 | 0.00 | US$2010/t CO2 | | 2030 | 50.00 | US$2010/t CO2 | | 2035 | 107.00 | US$2010/t CO2 | | 2040 | 113.00 | US$2010/t CO2 | | 2045 | 122.00 | US$2010/t CO2 | | 2050 | 132.00 | US$2010/t CO2 | **GCAM-PR 5.3 - PR_WWH_CP** | Year | Value | Unit | |------|…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `rainbow energy for EU`  
- route: `data_query` · plot: `False` · scope: region=`EU`
- answer: Based on your wording, I think one of these is the closest match. Choose the variable: 1. `Secondary Energy|Hydrogen|Biomass` (secondary energy, hydrogen) 2. `Secondary Energy|Liquids|Fossil` (secondary energy) 3. `Primary Energy|Gases` (primary energy, gas) Reply with a number (1-3), or `yes` for option 1.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `dragon emissions for World`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`World`
- answer: ### Emissions|CO2 in World Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr` Answer: **e3me - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2010 | 31.5K | Mt CO2/yr | | 2015 | 34.4K | Mt CO2/yr | | 2020 | 34.4K | Mt CO2/yr | | 2025 | 35.4K | Mt CO2/yr | | 2030 | 36.9K | Mt CO2/yr | | 2035 | 38.1K | Mt CO2/yr | | 2040 | 38.8K | Mt CO2/yr | | 2045 | 39.1K | Mt CO2/yr | | 2050 | 40.4K | Mt CO2/yr | **e3me - PR_CurPol_CP** | Year | Value | Unit | |--…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `plot happiness for China`  
- route: `data_plotting` · plot: `False` · scope: region=`CHN`
- answer: Could not identify a variable to plot. Did you mean: Land Cover|Forest|Natural Forest, Land Cover|Forest, Land Cover|Forest|Managed? Reply with a scenario, model, region, or year to narrow the answer.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `show GDP for EU from FakeModel`  
- route: `data_query` · plot: `False` · scope: variable=`GDP|MER`, region=`EU`
- answer: I couldn't match that to a known model. Try `list models` to see available options.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `compare Atlantis and World emissions`  
- route: `data_plotting` · plot: `False` · scope: variable=`Emissions|CO2`, region=`World`
- answer: Could not identify enough variables to compare. Found: [] Reply with a scenario, model, region, or year to narrow the answer.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

# Follow-up blocks (ίδιο session)

## 14 FU region

**Q:** `greenhouse gas emissions for World`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|Kyoto Gases`, region=`World`, model=`gcam`
- answer: ### Emissions|Kyoto Gases in World Scope: scenario `multiple`, model `gcam`, years `2005-2100` Unit: `Mt CO2-equiv/yr` Answer: **gcam - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 44.4K | Mt CO2-equiv/yr | | 2010 | 49.7K | Mt CO2-equiv/yr | | 2015 | 53.5K | Mt CO2-equiv/yr | | 2020 | 55.6K | Mt CO2-equiv/yr | | 2025 | 58.6K | Mt CO2-equiv/yr | | 2030 | 61.2K | Mt CO2-equiv/yr | | 2035 | 63.6K | Mt CO2-equiv/yr | | 2040 | 65.6K | Mt CO2-equiv/yr | | 2045 | 67.0K | Mt CO…
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `same for EU`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|Kyoto Gases`, region=`World`, scenario=`NZE`, model=`GCAM`
- answer: I could not find data for `Emissions|Kyoto Gases` in `World` under `NZE_EUPol_Stand`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. scenario `PR_Baseline` (baseline) 2. scenario `PR_CurPol_CP` (current policy) 3. scenario `PR_CurPol_CPO` (current policy) Reply with `1`, `2`, or `3`, or type the scenario you want.
- top link: GCAM: https://iamparis.eu/models (200)

**Q:** `same for India`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|Kyoto Gases`, region=`IND`, scenario=`NZE_Bench_L`, model=`GCAM`
- answer: I could not find data for `Emissions|Kyoto Gases` in `IND` under `NZE_EUPol_Stand`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. scenario `PR_Baseline` (baseline) 2. scenario `PR_CurPol_CP` (current policy) 3. scenario `PR_CurPol_CPO` (current policy) Reply with `1`, `2`, or `3`, or type the scenario you want.
- top link: GCAM: https://iamparis.eu/models (200)

**Q:** `plot India`  
- route: `data_plotting` · plot: `False` · scope: region=`IND`
- answer: Could not identify a variable to plot. Please specify a variable like 'solar capacity' or 'CO2 emissions'. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Results: https://iamparis.eu/results (200)

## 15 FU variable refine

**Q:** `show electricity for China`  
- route: `data_query` · plot: `False` · scope: region=`CHN`
- answer: No data found for **Secondary Energy|Electricity** in region `CHN` under scenario `None`. - Closest variables: `Secondary Energy`, `Secondary Energy|Electricity|Biomass`, `Secondary Energy|Electricity|Biomass|w/ CCS` Reply with the option you want to use.
- top link: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU: https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions (200)

**Q:** `I mean electricity generation from coal`  
- route: `data_query` · plot: `False` · scope: variable=`Secondary Energy|Electricity`, region=`COL`
- answer: ### Secondary Energy|Electricity in COL Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `EJ/yr` Answer: **42 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 0.15 | EJ/yr | | 2010 | 0.17 | EJ/yr | | 2015 | 0.19 | EJ/yr | | 2020 | 0.22 | EJ/yr | | 2025 | 0.24 | EJ/yr | | 2030 | 0.28 | EJ/yr | | 2035 | 0.31 | EJ/yr | | 2040 | 0.35 | EJ/yr | | 2045 | 0.39 | EJ/yr | **42 - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 2005 | 0.15 | EJ/…
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `same from solar`  
- route: `data_query` · plot: `False` · scope: variable=`Capacity|Electricity|Solar`
- answer: ### Capacity|Electricity|Solar Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `GW` Answer: **E4SMA-EU-TIMES 1.0 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2010 | 30.73 | GW | | 2020 | 113.75 | GW | | 2025 | 137.49 | GW | | 2030 | 303.52 | GW | | 2040 | 303.62 | GW | | 2050 | 350.83 | GW | **GCAM-PR 5.3 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2005 | 0.59 | GW | | 2010 | 9.41 | GW | | 2015 | 43.87 | GW | | 2020 | 221.29 | GW | | 2…
- top link: Results: https://iamparis.eu/results (200)

**Q:** `compare coal and solar`  
- route: `data_plotting` · plot: `False` · scope: variable=`Capacity|Electricity|Solar`
- answer: No data found for the requested variables in region 'ARE'. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

## 16 FU model context

**Q:** `tell me about IMAGE`  
- route: `model_explanation` · plot: `False` · scope: model=`MANAGE`
- answer: The IMAGE model, developed by PBL Netherlands Environmental Assessment Agency, is an integrated assessment model that simulates the global environmental consequences of human activities. It is designed to inform and support policy-makers by providing insights into the interactions between human development and the environment. The model is used to assess the impacts of various policies on climate change, biodiversity, and ecosystems. ### Key Features of IMAGE: - **Integration**: Combines socio-e…
- top link: MANAGE: https://iamparis.eu/models/manage (200)

**Q:** `show its primary energy for World`  
- route: `data_query` · plot: `False` · scope: variable=`Primary Energy`, region=`World`
- answer: ### Primary Energy in World Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `multiple` Answer: **42 - PR_Baseline** | Year | Value | Unit | |------|-------|------| | 2005 | 481.23 | EJ/yr | | 2010 | 538.79 | EJ/yr | | 2015 | 571.32 | EJ/yr | | 2020 | 603.42 | EJ/yr | | 2025 | 650.43 | EJ/yr | | 2030 | 688.45 | EJ/yr | | 2035 | 723.31 | EJ/yr | | 2040 | 749.64 | EJ/yr | | 2045 | 770.99 | EJ/yr | **42 - PR_CurPol_EI** | Year | Value | Unit | |------|-------|------| | 2005 | 4…
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `what about EU?`  
- route: `data_query` · plot: `False` · scope: variable=`Primary Energy`, region=`World`, scenario=`NZE`
- answer: I could not find data for `Primary Energy` in `World` under `NZE_EUPol_Stand`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. scenario `PR_Baseline` (baseline) 2. scenario `PR_CurPol_CP` (current policy) 3. scenario `PR_CurPol_CPO` (current policy) Reply with `1`, `2`, or `3`, or type the scenario you want.
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

**Q:** `make a chart`  
- route: `data_query` · plot: `False` · scope: variable=`Primary Energy`, region=`World`, scenario=`NZE`
- answer: I could not find data for `Primary Energy` in `World` under `NZE_EUPol_Stand`. Reason: the exact scenario combination is not available for this data slice. Closest valid options: 1. scenario `PR_Baseline` (baseline) 2. scenario `PR_CurPol_CP` (current policy) 3. scenario `PR_CurPol_CPO` (current policy) Reply with `1`, `2`, or `3`, or type the scenario you want.
- top link: Enhancing renewable energy system metrics: https://iamparis.eu/results/iam-compact/enhancing-renewable-energy-system-metrics/policy_questions (200)

## 17 FU scenario context

**Q:** `World CO2 emissions under current policies`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`World`, scenario=`Current Policies`
- answer: ### Emissions|CO2 in World Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr` Answer: **e3me - PR_CurPol_CP** | Year | Value | Unit | |------|-------|------| | 2010 | 31.5K | Mt CO2/yr | | 2015 | 34.4K | Mt CO2/yr | | 2020 | 34.4K | Mt CO2/yr | | 2025 | 35.4K | Mt CO2/yr | | 2030 | 36.9K | Mt CO2/yr | | 2035 | 38.0K | Mt CO2/yr | | 2040 | 38.5K | Mt CO2/yr | | 2045 | 38.8K | Mt CO2/yr | | 2050 | 40.1K | Mt CO2/yr | **e3me - PR_CurPol_EI** | Year | Value | Unit | |-…
- top link: Policy Catalogue Interactive Explorer: https://iamparis.eu/datastories/policyCatalog (200)

**Q:** `same under baseline`  
- route: `data_query` · plot: `False` · scope: variable=`Emissions|CO2`, region=`World`, scenario=`Baseline`
- answer: I need one more detail to continue. Please specify the variable, region, or scenario.
- top link: Results: https://iamparis.eu/results (200)

**Q:** `compare them`  
- route: `data_plotting` · plot: `False` · scope: —
- answer: Could not identify a variable to plot. Please specify a variable like 'solar capacity' or 'CO2 emissions'. Reply with a scenario, model, region, or year to narrow the answer.
- top link: Results: https://iamparis.eu/results (200)

**Q:** `now show only 2050`  
- route: `data_query` · plot: `False` · scope: start_year=`2050`, end_year=`2050`
- answer: Based on your wording, I think one of these is the closest match. Choose the variable: 1. `IDA|Emissions|CO2|Energy|Supply|Electricity|Emission end level 2050` (CO2 emissions, electricity) 2. `IDA|Emissions|CO2|Energy|Demand|Industry|Emission end level 2050` (CO2 emissions, industry) 3. `IDA|Emissions|CO2|Energy|Demand|Buildings|Emission end level 2050` (CO2 emissions, buildings) Reply with a number (1-3), or `yes` for option 1.
- top link: Results: https://iamparis.eu/results (200)

## 18 FU link after data

**Q:** `transport final energy for EU`  
- route: `data_query` · plot: `False` · scope: variable=`Price|Final Energy|Transportation|Electricity`, region=`EU`
- answer: ### Price|Final Energy|Transportation|Electricity in EU Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `US$2010/GJ or local currency/GJ` Answer: **GCAM-PR 5.3 - PR_WWH_CP** | Year | Value | Unit | |------|-------|------| | 2005 | 34.42 | US$2010/GJ or local currency/GJ | | 2010 | 35.92 | US$2010/GJ or local currency/GJ | | 2015 | 35.32 | US$2010/GJ or local currency/GJ | | 2020 | 37.31 | US$2010/GJ or local currency/GJ | | 2025 | 40.23 | US$2010/GJ or local currency/GJ | |…
- top link: Transportation Transformation: https://iamparis.eu/results/ndc-aspects/transportation-transformation/policy_questions (200)

**Q:** `where can I see transport results online?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Use these IAM PARIS links for this request:
- top link: Transportation Transformation: https://iamparis.eu/results/ndc-aspects/transportation-transformation/policy_questions (200)

**Q:** `what related questions can I ask?`  
- route: `general_qa` · plot: `False` · scope: —
- answer: Here are some related questions you can ask about transport final energy for the EU: 1. **Time-Series Data:** - What are the trends in transport final energy consumption in the EU over the past decade? - How does transport final energy consumption in the EU compare across different scenarios? 2. **Model Comparisons:** - Which models provide projections for transport final energy in the EU? - How do different models predict the future of transport final energy in the EU? 3. **Sectoral Analysis:**…
- top link: Home: https://iamparis.eu/ (200)
