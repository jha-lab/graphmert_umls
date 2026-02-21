MEANING_EXPL_qwen = """isa and inverse_isa are exact inverses of each other.
* isa - points up the hierarchy: “Diabetic retinopathy” isa “Retinal disease.” (specific → general)
* inverse_isa - points down the hierarchy: “Retinal disease” inverse_isa “Diabetic retinopathy.”
* cause_of - directional link where the source concept is understood to directly or indirectly produce, trigger, or give rise to the target concept.
* due_to - causal link – the subject condition, finding, or situation results from the object. Inverse: cause_of.
* associated_with - non‐directional link indicating that two concepts are statistically or clinically linked without asserting a clear cause-and-effect direction. 
* has_associated_morphology -links a pathological or clinical entity (typically a disease, syndrome or injury) to the characteristic structural change (“morphology”) it produces. Concretely: source: a disorder concept; target: a Morphologic Abnormality concept (e.g., “Necrosis”, “Hyperplasia”, “Fibrosis”). Inverse: associated_morphology_of.
* associated_finding_of - it reads:
    “X associated_finding_of Y”
    “Finding X is the clinical finding for which procedure Y is performed.”"""


MEANING_EXPL_GENERAL = """isa and inverse_isa are exact inverses of each other.
* isa - points up the hierarchy: “Diabetic retinopathy” isa “Retinal disease.” (specific → general)
* inverse_isa - points down the hierarchy: “Retinal disease” inverse_isa “Diabetic retinopathy.”

* plays_role - links a concept that acts in a particular functional capacity to the concept that names that capacity or role.
The relation is non-hierarchical (not broader / narrower) and is often used for chemicals and gene products that serve a pharmacological or biological role.

* has_clinical_course - attaches a disorder to its typical time-course descriptor (acute, chronic, relapsing, cyclic, etc.). Inverse: clinical_course_of.

* method_of	- describes how something is done – “the manner and sequence of events in performing a procedure.” Typically, it links a concrete test/technique to the broader activity it serves.

* cause_of - directional link where the source concept is understood to directly or indirectly produce, trigger, or give rise to the target concept.

* due_to - causal link – the subject condition, finding, or situation results from the object. Inverse: cause_of.

* associated_with - non‐directional link indicating that two concepts are statistically or clinically linked without asserting a clear cause-and-effect direction. 

* has_associated_morphology -links a pathological or clinical entity (typically a disease, syndrome or injury) to the characteristic structural change (“morphology”) it produces. Concretely: source: a disorder concept; target: a Morphologic Abnormality concept (e.g., “Necrosis”, “Hyperplasia”, “Fibrosis”). Inverse: associated_morphology_of.

* associated_finding_of - it reads:
    “X associated_finding_of Y”
    “Finding X is the clinical finding for which procedure Y is performed.”

* interprets - a relationship where a diagnostic or evaluative concept (e.g., a lab test, imaging study) is used to explain or make sense of a clinical finding or observation."""

#------

SYSTEM_CONTEXT_TEMPLATE = """You are completing triples for a medical knowledge graph on diabetes and its comorbidities. For each sample, you're given a sequence, a head entity in that sequence, a relation and a list of candidate tokens. The relations are from UMLS and have the same meaning.

Your task is to output a filtered list of high-quality and factual tails in the format: ["tail 1", "tail 2", ...] or [].

To form the list of candidates:
Step 1: Analyze the sequence to understand the context and identify the head entity and relation.
Step 2: Choose candidate tails. You can combine tokens from the candidate list to get the most precise, relevant and meaningful tails in the context of the head and relation. Combine subword tokens, too.
Step 3: Verify each candidate.


Verification. Each tail must:
- Be causally and factually related to the head via the specified relation. Make sure the relation direction is correct: the head implies the tail given the relation. Note that 'isa' is a subclass->class relation, 'inverse_isa' is a class->subclass relation.
- Be supported by the sequence, but you can rely on well-established medical knowledge even if the sequence doesn’t spell it out verbatim. If no reliable support exists, reject the tail.
- Add valuable medical knowledge to the graph. Tails must be non-redundant. When all tails are factual, prefer specific tails over general and vague, e.g, "proliferative diabetic retinopathy" over "retinopathy" . Terms that include "level", "disease", "disorder", "complication" are too vague and rarely add useful knowledge to KG.
- Include only tokens from the list of candidates.

Reason step by step within <think>...</think>. 

You will see incorrect outputs labeled "Output (Incorrect)". Avoid similar errors.

Before finalizing:
- Ensure all output constraints are met
- Validate that each tail is logically, contextually, and factually aligned with the head and relation. 
- Confirm that each triple adds meaningful knowledge to the graph.

Note the meaning of some relations in UMLS that you may encounter in the input:
{MEANING_EXPL}"""

# SYSTEM_CONTEXT = SYSTEM_CONTEXT_TEMPLATE.format(MEANING_EXPL=MEANING_EXPL_GENERAL)
SYSTEM_CONTEXT = SYSTEM_CONTEXT_TEMPLATE.format(MEANING_EXPL=MEANING_EXPL_qwen)



example_user_1 = f"""Input:
2 + binding to s100a1 ef - hand motifs, the conformation of s100a1 changes and promotes interactions with target proteins. rage consists of three domains : the cytoplasmic, transmembrane, and extracellular domains. the extracellular domain consists of c1, c2, and v domains. v domains are the primary receptors for the s100 protein. it was reported several years ago that s100a1 and rage v domains interact in a pathway involving s100a1 - rage signaling, whereby s100a1 binds to the v domain, resulting in rage dimerization. the autophosphorylation of the cytoplasmic domain initiates a signaling cascade that regulates cell proliferation, cell growth, and tumor formation. in this study

head: s100a1
relation: associated_with
predictions: protein receptor hydrolase structure process pathway factor complex glycoprotein s100a family domain oxidoreductase proteinase ligand extracellular signaling calcium apolipoprotein s100
Output:"""

example_assistant_1 = """["calcium signaling pathway"]"""

example_explanation_1 = """Available tokens let us build terms such as:
• “calcium signaling pathway” (calcium + signaling + pathway)
• “extracellular signaling” (extracellular + signaling) – too broad
• “s100a family” – classification, not an association
Other tokens (“glycoprotein”, “oxidoreductase”, etc.) do not represent well-known processes linked to S100A1.

– Preference & verification
“S100A1 modulates intracellular Ca²⁺ handling and is well-documented to participate in the calcium signaling pathway, especially in cardiac and skeletal muscle.” This is a concrete biological pathway association that adds value to the graph."""


# -----

example_user_2 = f"""Input:
3. 31 ] ) and approximately 2 times more likely to experience stroke ( hr 1. 83 [ 1. 77 – 1. 89 ] ) or mi ( hr 2. 27 [ 2. 20 – 2. 34 ] ). conclusion ascvd is associated with both economic and clinical impacts. people with ascvd incurred considerably higher costs than matched controls, with indirect costs resulting from work absence and inpatient admissions being major cost drivers, and were also more likely to experience additional ascvd events. supplementary information the online version contains supplementary material available at 10. 1186 / s12872 - 023 - 03518 - y. rabies is a

head: stroke
relation: cause_of
predictions: ischemic stroke acute coronary heart myocardial infarction accident cerebrovascular cardiac cardiovascular artery following ischaemic of angina chronic disease disorder limb

Output:"""

example_assistant_2 = """[]"""

example_explanation_2 = """The excerpt states that individuals with ASCVD are more likely to experience stroke, but it never describes stroke causing any subsequent condition or event. None of the candidate tokens (e.g., “myocardial infarction,” “limb,” “cerebrovascular accident”) are presented in the text as outcomes caused by stroke. Because “cause of” relation lacks support, every candidate must be rejected."""
# -----

example_user_3 = f"""Input:
in the old recipient rats and made the metabolic profile of the old recipients more similar to that of the young recipients. finally, combining the above metabolomic data with the transcriptomic data from the geo, we found that the altered metabolites and genes in the liver were enriched in 9 metabolic pathways, including glycerophospholipid, arachidonic acid, histidine and linoleate. thus, this study revealed important age ‐ related metabolites and potential pathways as well as the interaction between the liver and the whole ‐ body environment. aging is related to metabolic disorder. to better understand the underlying mechanisms, we transplanted livers of young rats to young and old rats. analyses of

head: metabolic pathways
relation: has_part
predictions: metabolic acid metabolism lipid process pathway ##olipid disorder ##ophosph catabolic glycer biosynthetic carbohydrate fatty amino sphing lipids structure of metabol

Output:"""

example_assistant_3 = """["glycerophospholipid metabolism"]"""

example_explanation_3 = """The passage specifies that the enriched metabolic pathways include glycerophospholipid.
From the sub-tokens glycer + ##ophosph + ##olipid + metabolism, we can reconstruct the precise pathway name “glycerophospholipid metabolism,” which is explicitly supported and represents a component (part) of the broader set of metabolic pathways discussed."""
# -----

example_user_4 = f"""Input:
cyclooxygenase ( cox ) - 2, inducible nitric oxide synthase ( inos ) 2, tumor necrosis factor ( tnf ) - α, and interleukin ( il ) - 6 in lps - stimulated raw 264. 7 cells. the fge treatment also decreased the production of nitric oxide, tnf - α, and il - 6 significantly in a dose - dependent manner. in addition, fge downregulated phosphorylation of mapk and nf - κb signaling pathway - related proteins. the chromatographic and mass spectrometry results showed that fge contained bioactive flavonoids such as ( + ) - catechin, miquelianin, quercitrin, and afzelin, which 

head: inducible nitric oxide synthase ( inos )
relation: has_disposition
predictions: oxidoreductase hydrolase enzyme transferase cyclooxygenase ##ase ligase substance oxidase peroxisome ##ulator agonist receptor - protease proliferator structure prostaglandin lyase antiviral

Output:"""

example_assistant_4 = """["oxidoreductase"]"""

example_explanation_4 = """1 — Context & relation** *Head*: **inducible nitric oxide synthase (iNOS)** — an enzyme that catalyzes a redox reaction to generate nitric oxide. *Relation*: **has_disposition** links a material entity to the functional capacity it inherently possesses.

2 — Build candidate tails from provided tokens
Tokens permit several enzyme-class words. The most precise, function-type term is “oxidoreductase.”

3 — Verification
iNOS is classified biochemically as an oxidoreductase because it transfers electrons during the oxidation of L-arginine to produce NO. Thus the disposition “oxidoreductase” is factual and specific. Other options (e.g., “enzyme”) are too general."""
# -----

example_user_5 = f"""Input:
. 0085 ), beyond which this association was not observed ( adjusted β −0. 43, 95 % ci −1. 57, 0. 71 ; p = 0. 4600 ). conclusion we found positive associations between urinary arsenic metabolites udma, uas levels and poor hearing threshold shifts in us adults. this study provides new evidence for the association between arsenic exposure and auditory function. care groups organize integrated cardiovascular risk management programs in primary care for high risk patients. results of long term cardiovascular risk management are scarce. the aim was to describe changes in low density lipoprotein cholesterol, systolic blood pressure and smoking between 2011 and 2018 in patients participating 

head: low density lipoprotein cholesterol
relation: part_of
predictions: density lipoprotein - low and cholesterol high very , dense serum total ldl to ‐ sensitivity : binding hdl plasma

Output:"""

example_assistant_5 = """["total cholesterol"]"""

example_explanation_5 = """**Step 1 – Context & relation** *Head*: **low-density lipoprotein cholesterol (LDL-C)**. *Relation*: **part_of** – the head is a component contained within the tail.

Step 2 – Build candidate tails (must use only provided tokens)
Tokens allow the construction of:
    “total cholesterol” (tokens “total” + “cholesterol”)
    “serum total cholesterol” (adds “serum”)
    “plasma total cholesterol” (adds “plasma”)

Step 3 – Verify
LDL-C is a quantitative fraction that contributes to total cholesterol measured in blood. Thus “total cholesterol” is a valid superclass that contains LDL-C.
Adding “serum” or “plasma” does not materially improve specificity for the KG, so the concise concept “total cholesterol” is preferred.

Step 4 – Output
Return the single verified tail."""
# -----

example_user_6 = f"""Input:
m3 and m3 - 12, in addition to recipient age. corticosteroids, diabetes at baseline, and asat yielded inconstant results between d8 - m3 and m3 - m12. we found no ethnicity effect when cyp3a phenotype was accounted for, and no food effect. intra - individual variability over m3 - m12 was moderate, and significantly lower in patients with chronic hepatic disorder ( p = 0. 0196 ) or cancer ( p = 0. 0132 ). graphical abstract the triglyceride – glucose ( tyg ) index has been shown to be an independent predictor for the progression and prognosis of coronary artery disease ( cad ). whether 

head: triglyceride – glucose ( tyg ) index
relation: associated_with
associated_with
predictions: triglyceride ##idemia glucose insulin hypertriglycer hyperglycemia dyslipidemia lipid diabetes ##mia hyperlipidemia lipoprotein glycemic metabolism mellitus hyperglycemic plasma fatty resistance metabolic

Output:"""

example_assistant_6 = """["insulin resistance"]"""

example_explanation_6 = """1 — Understand context & relation** *Head*: **triglyceride–glucose (TyG) index** – a lipid-glucose composite biomarker. *Relation*: **associated_with** (non-directional statistical/clinical link).

2 — Generate candidate tails using provided tokens
Available tokens enable:
    “insulin resistance” (“insulin” + “resistance”)
    “diabetes mellitus” (“diabetes” + “mellitus”)
    “hypertriglyceridemia” (“hypertriglycer” + “##idemia”)

3 — Verify candidates
    Insulin resistance – The TyG index is widely validated as a surrogate marker of insulin resistance; strong, well-established association.
    Diabetes mellitus – TyG predicts incident diabetes but this adds little beyond the better-supported link with insulin resistance; acceptable yet less specific.
    Hypertriglyceridemia – Trivial/tautological, as serum triglyceride is a component of the index; offers no new knowledge."""
# -----

example_user_7 = f"""Input:
tyg index and the risk of complicated cad ( syntax score > 22 ; nonlinear p = 0. 200 ). the risk for a mid / high syntax score in the t2 and t3 groups was significantly higher in normoglycemia, prediabetes mellitus, and diabetes mellitus subgroups. conclusions a higher tyg index was associated with the presence of a higher coronary anatomical complexity ( syntax score > 22 ) in acs patients, irrespective of diabetes mellitus status. the tyg index might serve as a noninvasive predictor of cad complexity in acs patients and could potentially influence the management and therapeutic approach. supplementary information the online version contains supplementary material 

head: complicated cad
relation: has_associated_morphology
has_associated_morphology
predictions: ##osclerosis ##ific atherosclerosis calcification ischemia coronary disease myocardial arteri structure obstruction segmental stenosis to calcified fibrosis fibrous atherosclerotic - wall

Output:"""

example_assistant_7 = """["atherosclerosis", "calcification", "stenosis"]"""

example_explanation_7 = """**1 — Context & relation** *Head*: **complicated coronary artery disease (CAD)** — a disorder. *Relation*: **has_associated_morphology** (disorder → characteristic structural change).

2 — Build tails from candidate tokens
Tokens permit these UMLS morphologic-abnormality concepts:
    atherosclerosis
    calcification
    stenosis

3 — Verification
    atherosclerosis: CAD arises from atherosclerotic plaque formation.
    calcification: complex CAD often features calcified plaques.
    stenosis: luminal narrowing is the key structural change driving CAD severity."""
# -----

example_user_8 = f"""Input:
cyclooxygenase ( cox ) - 2, inducible nitric oxide synthase ( inos ) 2, tumor necrosis factor ( tnf ) - α, and interleukin ( il ) - 6 in lps - stimulated raw 264. 7 cells. the fge treatment also decreased the production of nitric oxide, tnf - α, and il - 6 significantly in a dose - dependent manner. in addition, fge downregulated phosphorylation of mapk and nf - κb signaling pathway - related proteins. the chromatographic and mass spectrometry results showed that fge contained bioactive flavonoids such as ( + ) - catechin, miquelianin, quercitrin, and afzelin, which 

head: ( + ) - catechin
relation: isa
predictions: flavonoid compound acid containing ##ic phenol ##flav product ##pen catech - plant ##in derivative phenolic rich tann polyphenol ##oid inhibitor

Output:"""

example_assistant_8 = """["flavonoid", "polyphenol"]"""

example_explanation_8 = """**Step 1 – Context & relation** Head concept: **(+)-catechin** (a specific flavan-3-ol). Relation: **isa** → the head is a more specific instance/subclass of the tail.

Step 2 – Construct candidate tails from supplied tokens
Tokens permit coherent class concepts:
    “flavonoid”
    “polyphenol”

Step 3 – Verify candidates
    flavonoid – Catechin is classified as a flavonoid (flavan-3-ol subclass).
    polyphenol – Flavonoids, including catechin, belong to the broader polyphenol family.
    Both are established hierarchical parents of catechin and are non-redundant (different granularity levels)."""
# -----

example_user_9 = f"""Input:
- hydroxypyrimidine to induce enos uncoupling abolishes the protective effect of fam3d deficiency against angii - induced hypertension. furthermore, antagonism of formyl peptide receptor 1 ( fpr1 ) and fpr2 or the suppression of oxidative stress blunts fam3d - induced enos uncoupling. translationally, targeting endothelial fam3d by adeno - associated virus or intraperitoneal injection of fam3d - neutralizing antibodies markedly ameliorates angii - or deoxycorticosterone acetate ( doca ) - salt - induced hypertension. conclusively, fam3d causes enos uncoupling through fpr1 - and fpr2 - mediated oxidative stress, thereby exacerbating the development of hypertension. fam3d 

head: fam3d
relation: cause_of
predictions: - haploinsu ##tr deficiency ##ep hereditary product ##pi rub cassette human fam of inhibitor containing protein family ##amyl ##ocysteine ##3d

Output:"""

example_assistant_9 = """[]"""

example_explanation_9 = """**1 — Context & relation** Head: **FAM3D** (a cytokine-like protein). Relation: **cause_of** – the head must produce or induce the tail condition.

2 — Token review & tail construction
Allowed tokens: “deficiency”, “hereditary”, “inhibitor”, “protein”, etc.
Crucially, tokens for the actual disorders FAM3D is known to cause (e.g., “hypertension”, “oxidative stress”, “eNOS uncoupling”) are not present.
No combination of the provided tokens yields a medically meaningful disease or pathologic state attributable to FAM3D.

3 — Verification
No valid concept can be formed.

4 — Decision
Reject all"""
# -----
# Combined examples list for easy import
PROMPT_EXAMPLES = [
    (example_user_1, example_assistant_1, example_explanation_1),
    (example_user_2, example_assistant_2, example_explanation_2),
    (example_user_3, example_assistant_3, example_explanation_3),
    (example_user_4, example_assistant_4, example_explanation_4),
    (example_user_5, example_assistant_5, example_explanation_5),
    (example_user_6, example_assistant_6, example_explanation_6),
    (example_user_7, example_assistant_7, example_explanation_7),
    (example_user_8, example_assistant_8, example_explanation_8),
    (example_user_9, example_assistant_9, example_explanation_9),
]