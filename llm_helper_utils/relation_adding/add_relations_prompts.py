ALLOWED_RELATIONS = """associated_with
plays_role
isa
inverse_isa
cause_of
focus_of
due_to
part_of
interprets
has_disposition
has_component
has_associated_morphology
has_finding_site
associated_finding_of
has_part
causative_agent_of
is_interpreted_by
associated_morphology_of
finding_site_of
has_method
has_modification
possibly_equivalent_to
has_causative_agent
same_as
occurs_before
has_focus
has_clinical_course
occurs_in"""


REL_USAGE_EXAMPLES = """carotid artery stenosis | associated_finding_of | history of carotid artery stenosis
fibrosis | associated_morphology_of | endomyocardial fibrosis
cancer | associated_with | anemia in malignant neoplastic disease
Mycobacterium tuberculosis | causative_agent_of | Tuberculosis
diabetes mellitus | cause_of | diabetic foot
hypoglycemic alcoholic ketoacidosis | due_to | acute alcohol intoxication
adipose tissue | finding_site_of | lipoatrophy
renal failure | focus_of | emergency hemofiltration
hepatitis a | has_associated_morphology | Hepatocellular necrosis
influenza | has_causative_agent | Influenza A virus
acute myocardial infarction | has_clinical_course | Abrupt onset
fasting triglyceride | has_component | triacylglycerol
tumor necrosis factor | has_disposition | immunomodulator
melanoma | has_finding_site | skin
vertebroplasty | has_focus | Vertebral fracture
bariatric surgery | has_method | surgical action
glucagon | has_modification | glucagon hydrochloride
nephron | has_part | glomerulus
overweight | interprets | body weight measure
adiponectin | inverse_isa | high molecular weight adiponectin
blood eosinophil counts | is_interpreted_by | asthmatic pulmonary eosinophilia
empagliflozin | isa | sodium glucose cotransporter subtype 2 inhibitor
mastectomy | occurs_before | post - mastectomy chronic pain syndrome
cardiac amyloidosis | occurs_in | old age
postpartum anxiety status | occurs_in | maternal postpartum
cellular senescence | part_of | cellular response to stress
agomelatine | plays_role | antidepressant therapeutic role
coronary syndrome | possibly_equivalent_to | preinfarction syndrome
mi | same_as | myocardial infarction"""


MEANING_EXPL = """isa and inverse_isa are exact inverses of each other.
* isa - points up the hierarchy: “Diabetic retinopathy” isa “Retinal disease.” (specific → general)
* inverse_isa - points down the hierarchy: “Retinal disease” inverse_isa “Diabetic retinopathy.”

* plays_role - links a concept that acts in a particular functional capacity to the concept that names that capacity or role.
The relation is non-hierarchical (not broader / narrower) and is often used for chemicals and gene products that serve a pharmacological or biological role.

* has_clinical_course - attaches a disorder to its typical time-course descriptor (acute, chronic, relapsing, cyclic, etc.). Inverse: clinical_course_of.

* cause_of - directional link where the source concept is understood to directly or indirectly produce, trigger, or give rise to the target concept.

* due_to - causal link – the subject condition, finding, or situation results from the object. Inverse: cause_of.

* associated_with - non‐directional link indicating that two concepts are statistically or clinically linked without asserting a clear cause-and-effect direction. 

* has_associated_morphology -links a pathological or clinical entity (typically a disease, syndrome or injury) to the characteristic structural change (“morphology”) it produces. Concretely: source: a disorder concept; target: a Morphologic Abnormality concept (e.g., “Necrosis”, “Hyperplasia”, “Fibrosis”). Inverse: associated_morphology_of.

* associated_finding_of - it reads:
    “X associated_finding_of Y”
    “Finding X is the clinical finding for which procedure Y is performed.”

* interprets - a relationship where a diagnostic or evaluative concept (e.g., a lab test, imaging study) is used to explain or make sense of a clinical finding or observation."""

#------


SYSTEM_CONTEXT_TEMPLATE = """You are a medical intelligence with academic knowledge in diabetes and comorbidities.
We are building a diabetes knowledge graph of triples <head, relation, tail>.
Given:
 • a sequence including biomedical context
 • a list of heads
return for each head all relations chosen from the list below that could form a plausible KG triple and are supported by the sequence.

Allowed relations:
###
{ALLOWED_RELATIONS}
###

The relations are taken from UMLS and have the same meaning as in UMLS. Examples:
-------------
{REL_USAGE_EXAMPLES}
-------------

Note the meaning of some relations in UMLS:
{MEANING_EXPL}

-------------   

The input format:
sequence

heads: [head1, head2, ...]

Output format:
{{
    "head 1": ["relation 1", "relation 2", ...],
    "head 2": [...],
}}

##
Steps
Step 1: Understand Input
    Clearly understand the biomedical context from the sequence.
    For each head, find explicit mentions in the text.
    Check if each head is explicitly linked to other concepts or relations.

Step 2: Use the list of allowed relations. Evaluate each head individually. Don't abuse the relation "associated_with" -- use it only if it's appropriate.

Step 3: For each head, list only plausible and supported relations. Return [] if none apply.
##

Think concisely within <think></think>. Immediately after, output JSON."""


SYSTEM_CONTEXT = SYSTEM_CONTEXT_TEMPLATE.format(ALLOWED_RELATIONS=ALLOWED_RELATIONS, REL_USAGE_EXAMPLES=REL_USAGE_EXAMPLES, MEANING_EXPL=MEANING_EXPL)


# ---- Examples for prompt ----

example_user_1 = f"""diabetes mellitus encompasses two principal subtypes—type 1 diabetes mellitus, produced by autoimmune β-cell destruction, and type 2 diabetes mellitus, driven mainly by insulin resistance.  Persistently elevated glucose accelerates atherosclerosis and greatly increases the likelihood of myocardial infarction.  Pathologically, an acute myocardial infarction is defined as coagulative necrosis located in the ventricular myocardium; the injury involves full-thickness loss of viable cardiac muscle fibres.  Cigarette smoking continues to be the leading modifiable risk factor for malignancy, with epidemiological evidence showing that roughly eighty-five percent of primary lung-cancer deaths are attributable to chronic tobacco exposure through oxidative DNA damage in bronchial epithelium.

heads: ["diabetes mellitus", "myocardial infarction"]

Output:"""

example_assistant_1 = """{
    "diabetes mellitus": ["inverse_isa", "associated_with"],
    "myocardial infarction": ["has_associated_morphology", "has_finding_site", "due_to", "associated_with"],
}
"""

example_explanation_1 = """The rationale behind this output:
diabetes mellitus:

"type 1 diabetes mellitus" and "type 2 diabetes mellitus" are subtypes of the broader class "diabetes mellitus" — precisely the inverse_isa hierarchical relation.
associated clearly with myocardial infarction via accelerating atherosclerosis.


myocardial infarction:
        has_associated_morphology "coagulative necrosis"
        has_finding_site "ventricular myocardium"

    Clearly mentioned to occur due to accelerated atherosclerosis caused by elevated glucose levels:
        "myocardial infarction" due_to "atherosclerosis"

    Additionally, myocardial infarction is clearly associated with diabetes mellitus
"""

# -----

example_user_2 = """Input:
related osteonecrosis of the jaw, nonsterile or sterile osteomyelitis of the jaw, and osteoporosis. in this review, we present a framework for understanding the role and mechanism of canonical and noncanonical inflammasomes in the pathogenesis and development of etiologically diverse diseases associated with alveolar bone loss. inappropriate inflammasome activation may drive alveolar osteolysis by regulating cellular players, including osteoclasts, osteoblasts, osteocytes, periodontal ligament cells, macrophages, monocytes, neutrophils, and adaptive immune cells, such as t helper 17 cells, causing increased osteoclast activity, decreased osteoblast activity, and enhanced periodontium inflammation by creating a pro - inflammatory milieu in a context -

heads: ['osteomyelitis of the jaw', 'osteoporosis', 'inflammasomes', 'alveolar bone loss', 'osteoclasts']"""

example_assistant_2 = """{
    "osteomyelitis of the jaw": ["has_associated_morphology"],
    "osteoporosis": ["cause_of", "has_associated_morphology"],
    "inflammasomes": ["cause_of", "has_disposition"],
    "alveolar bone loss": ["due_to", "associated_morphology_of"],
    "osteoclasts": ["plays_role"],
    "osteoblasts": ["plays_role"],
}
"""

example_explanation_2 = """The rationale behind this output:
    osteomyelitis of the jaw → has_associated_morphology – Likewise listed among diseases linked to alveolar bone loss, warranting the same disorder-to-morphology relation.

    osteoporosis → cause_of – Osteoporosis is named as a condition whose presence can lead to alveolar bone loss (osteolysis), making it a direct cause_of that outcome.
    osteoporosis → has_associated_morphology – Alveolar bone loss is also a typical skeletal change in osteoporosis, justifying the morphology link.

    inflammasomes → cause_of – “Inappropriate inflammasome activation may drive alveolar osteolysis,” i.e., it triggers the bone-loss process; hence cause_of.
    inflammasomes → has_disposition – Inflammasomes inherently possess the functional capacity to initiate inflammatory cascades; UMLS captures such intrinsic functional properties with has_disposition (entity → pro-inflammatory mediator role).

    alveolar bone loss → due_to – The bone loss results from upstream inflammasome activation, so the abnormality is due_to that causal mechanism.
    alveolar bone loss → associated_morphology_of – Conversely, it serves as the characteristic morphology associated_morphology_of the listed jaw disorders and osteoporosis.

    osteoclasts / osteoblasts - these cellular players are described as effectors whose altered activity or presence drives the osteolytic, pro-inflammatory milieu. Each cell type acts in a specific functional capacity within the pathological process, matching plays_role.

No additional relations are warranted:

    Although the diseases and bone-loss abnormality are “associated,” more precise morphological (has_associated_morphology / associated_morphology_of) and causal (cause_of / due_to) relations apply, so associated_with is omitted.

    The passage does not specify anatomical sites, findings, or procedures"""

# -----
example_user_3 = """Input:
interleukin ‐ 1 r6, and receptor activator of nuclear factor kappa ‐ b ( rank ). together, proteomic data suggest the targeting of several key regulators of inflammation, bone, and adipose turnover, via transforming growth factor ‐ beta / smad, and wingless ‐ related integration site / be ‐ catenin signaling pathways. to the best of the knowledge, this is first evidence of an intervention that drives against bone loss via rank. metatranscriptomic analyses of the gut microbiota show p7c3 increased porphyromonadaceae bacterium, candidatus melainabacteria, and ruminococcaceae bacterium abundance, potentially contributing to the favorable inflammatory

heads: ['interleukin ‐ 1 r6', 'receptor activator of nuclear factor kappa ‐ b', 'transforming growth factor ‐ beta']

Output:"""

example_assistant_3 = """{
    "interleukin ‐ 1 r6": ["associated_with"],
    "receptor activator of nuclear factor kappa ‐ b": ["cause_of"],
    "transforming growth factor ‐ beta": ["part_of"],
}"""

example_explanation_3 = """The rationale behind this output:

    interleukin - 1 r6 → associated_with – Named as a “key regulator of inflammation,” which links it to the inflammatory process without stating direction or hierarchy, so the non-causal associated_with relation fits best.

    receptor activator of nuclear factor κ-B (RANK) → cause_of – The text says the intervention prevents bone loss via RANK, implying that RANK signalling produces or drives bone loss; therefore cause_of is appropriate.

    transforming growth factor-β → part_of – Explicitly mentioned within the “TGF-β/Smad signalling pathway,” so it is a constituent component (part_of) that pathway.

No additional relations are warranted
"""
# -----

example_user_4 = """Input:
subretinal fluid ( srf ) resolution and their correlation with the final anatomical and functional outcome in cases of treatment naive acute central serous chorioretinopathy ( cscr ). methods and analysis we retrospectively studied 93 eyes of 93 patients diagnosed with treatment naive acute cscr presenting within 30 days of onset of symptoms. the eyes were divided into two groups ( 1 and 2 ) based on the duration of srf resolution ; which was ≤3 months in group 1 and > 3 months in group 2. demographic and medical history, and spectral domain optical coherence tomography features were noted and their association with duration for srf resolution, final central macular thickness ( cmt ) and

heads: ['subretinal fluid', 'central serous chorioretinopathy', 'spectral domain optical coherence tomography', 'central macular thickness']"""

example_assistant_4 = """{
    "subretinal fluid": ["associated_morphology_of"],
    "central serous chorioretinopathy": ["has_associated_morphology", "interprets"],
    "spectral domain optical coherence tomography": ["method_of"],
    "central macular thickness": ["is_interpreted_by"]
}"""

example_explanation_4 = """The rationale behind this output:
    subretinal fluid → associated_morphology_of – SRF is the hallmark morphological finding in acute CSCR; the fluid accumulation is a specific tissue morphology linked to that disease, so the morphology concept subretinal fluid is associated_morphology_of the disorder.

    central serous chorioretinopathy → has_associated_morphology – Conversely, CSCR has_associated_morphology SRF, reflecting the disease-to-morphology direction.
    central serous chorioretinopathy → interprets – CSCR is assessed through quantitative retinal metrics such as central macular thickness; the disease status provides the clinical interpretation of that measurement, matching interprets.

    spectral domain optical coherence tomography → method_of – SD-OCT is the imaging technique used to obtain retinal features (e.g., CMT) in this study; the modality is thus the method_of those diagnostic measurements.

    central macular thickness → is_interpreted_by – CMT values are interpreted in the context of CSCR severity and outcome, so the measurement concept central macular thickness is_interpreted_by the disease condition.

No additional relations are warranted:

    The text does not specify causal (cause_of / due_to), anatomical site (has_finding_site), or role-based (plays_role / has_disposition) links for these heads.

    associated_with is avoided in favor of the more precise morphological and interpretive relations explicitly supported by the context."""

# -----
example_user_5 = """Input:
partial or complete blindness if left untreated. as such, there is considerable interest in identifying more effective therapeutic options and preventive agents. one such agent is vitamin d, known to have a range of anti - cancer, anti - angiogenic, anti - inflammatory and anti - oxidative properties, and whose deficiency is linked to the pathogenesis of a range of cardiovascular, cancer, and inflammatory diseases. this review presents the current stage of knowledge concerning the link between vitamin d and its receptor and the occurrence of eye disease, as well as the influence of analogues of calcitriol, an active metabolite of vitamin d. generally, patients affected by various ocular disorders have

heads: ['blindness', 'vitamin d', 'calcitriol', 'eye disease']"""

example_assistant_5 = """{
    "blindness": ["occurs_after"],
    "vitamin d": ["associated_with", "plays_role"],
    "calcitriol": ["isa", "associated_with"],
    "eye disease": ["cause_of", "associated_with"]
}"""

example_explanation_5 = """blindness → occurs_after – The text notes that untreated ocular disorders can lead to “partial or complete blindness,” so blindness manifests after the underlying eye disease, satisfying occurs_after.

    vitamin d → associated_with – The review discusses “the link between vitamin D … and the occurrence of eye disease,” indicating a non-directional clinical association; thus associated_with fits.
    vitamin d → plays_role – Vitamin D is said to possess anti-inflammatory, anti-angiogenic, and other functional properties, meaning it acts in specific biological roles; plays_role links the molecule to such roles.

    calcitriol → isa – Calcitriol is described as “an active metabolite of vitamin D,” making it a more specific form within the vitamin D class; therefore isa (specific → general) is appropriate.
    calcitriol → associated_with – Analogues of calcitriol are discussed for their influence on ocular disorders, implying a clinical link; associated_with applies.

    eye disease → cause_of – Untreated ocular disease is implied to lead to blindness, making eye disease a direct cause of that outcome; hence cause_of.
    eye disease → associated_with – The stated “link between vitamin D … and the occurrence of eye disease” shows a bidirectional clinical association; associated_with is justified.

No additional relations are warranted:

    Hierarchical inverses, part-of, or procedural links are not explicitly supported in the passage.

    associated_with is used only where a clear but non-causal clinical link is stated; more specific causal or hierarchical relations are applied wherever the text justifies them."""

# ============
# For expanded KG as the seed: removed examples of relations that are not in the expanded seed KG

expanded_kg_example_user_4 = """Input:
subretinal fluid ( srf ) resolution and their correlation with the final anatomical and functional outcome in cases of treatment naive acute central serous chorioretinopathy ( cscr ). methods and analysis we retrospectively studied 93 eyes of 93 patients diagnosed with treatment naive acute cscr presenting within 30 days of onset of symptoms. the eyes were divided into two groups ( 1 and 2 ) based on the duration of srf resolution ; which was ≤3 months in group 1 and > 3 months in group 2. demographic and medical history, and spectral domain optical coherence tomography features were noted and their association with duration for srf resolution, final central macular thickness ( cmt ) and

heads: ['subretinal fluid', 'central serous chorioretinopathy', 'spectral domain optical coherence tomography', 'central macular thickness']"""

expanded_kg_assistant_4 = """{
    "subretinal fluid": ["associated_morphology_of"],
    "central serous chorioretinopathy": ["has_associated_morphology", "interprets"],
    "central macular thickness": ["is_interpreted_by"]
}"""

expanded_kg_explanation_4 = """The rationale behind this output:
    subretinal fluid → associated_morphology_of – SRF is the hallmark morphological finding in acute CSCR; the fluid accumulation is a specific tissue morphology linked to that disease, so the morphology concept subretinal fluid is associated_morphology_of the disorder.

    central serous chorioretinopathy → has_associated_morphology – Conversely, CSCR has_associated_morphology SRF, reflecting the disease-to-morphology direction.
    central serous chorioretinopathy → interprets – CSCR is assessed through quantitative retinal metrics such as central macular thickness; the disease status provides the clinical interpretation of that measurement, matching interprets.

    central macular thickness → is_interpreted_by – CMT values are interpreted in the context of CSCR severity and outcome, so the measurement concept central macular thickness is_interpreted_by the disease condition.

No additional relations are warranted:

    The text does not specify causal (cause_of / due_to), anatomical site (has_finding_site), or role-based (plays_role / has_disposition) links for these heads.

    associated_with is avoided in favor of the more precise morphological and interpretive relations explicitly supported by the context."""


expanded_kg_example_user_5 = """Input:
partial or complete blindness if left untreated. as such, there is considerable interest in identifying more effective therapeutic options and preventive agents. one such agent is vitamin d, known to have a range of anti - cancer, anti - angiogenic, anti - inflammatory and anti - oxidative properties, and whose deficiency is linked to the pathogenesis of a range of cardiovascular, cancer, and inflammatory diseases. this review presents the current stage of knowledge concerning the link between vitamin d and its receptor and the occurrence of eye disease, as well as the influence of analogues of calcitriol, an active metabolite of vitamin d. generally, patients affected by various ocular disorders have

heads: ['vitamin d', 'calcitriol', 'eye disease']"""

expanded_kg_assistant_5 = """{
    "vitamin d": ["associated_with", "plays_role"],
    "calcitriol": ["isa", "associated_with"],
    "eye disease": ["cause_of", "associated_with"]
}"""

expanded_kg_explanation_5 = """vitamin d → associated_with – The review discusses “the link between vitamin D … and the occurrence of eye disease,” indicating a non-directional clinical association; thus associated_with fits.
    vitamin d → plays_role – Vitamin D is said to possess anti-inflammatory, anti-angiogenic, and other functional properties, meaning it acts in specific biological roles; plays_role links the molecule to such roles.

    calcitriol → isa – Calcitriol is described as “an active metabolite of vitamin D,” making it a more specific form within the vitamin D class; therefore isa (specific → general) is appropriate.
    calcitriol → associated_with – Analogues of calcitriol are discussed for their influence on ocular disorders, implying a clinical link; associated_with applies.

    eye disease → cause_of – Untreated ocular disease is implied to lead to blindness, making eye disease a direct cause of that outcome; hence cause_of.
    eye disease → associated_with – The stated “link between vitamin D … and the occurrence of eye disease” shows a bidirectional clinical association; associated_with is justified.

No additional relations are warranted:

    Hierarchical inverses, part-of, or procedural links are not explicitly supported in the passage.

    associated_with is used only where a clear but non-causal clinical link is stated; more specific causal or hierarchical relations are applied wherever the text justifies them."""




positive_examples = [
    (example_assistant_1, example_assistant_1, example_explanation_1),
    (example_user_2, example_assistant_2, example_explanation_2),
    (example_user_3, example_assistant_3, example_explanation_3),
    (expanded_kg_example_user_4, expanded_kg_assistant_4, expanded_kg_explanation_4),
    (expanded_kg_example_user_5, expanded_kg_assistant_5, expanded_kg_explanation_5),
]

# the largeset list of relations examples
# EXAMPLES_gemini_score45 = """carotid artery stenosis | associated_finding_of | history of carotid artery stenosis
# fibrosis | associated_morphology_of | endomyocardial fibrosis
# cancer | associated_with | anemia in malignant neoplastic disease
# Mycobacterium tuberculosis | causative_agent_of | Tuberculosis
# diabetes mellitus | cause_of | diabetic foot
# angiotensin - converting enzyme inhibitor | disposition_of | zofenopril
# hypoglycemic alcoholic ketoacidosis | due_to | acute alcohol intoxication
# chemotherapy - induced peripheral neuropathy| due_to | administration of antineoplastic agent
# kidney | direct_procedure_site_of | xenograft renal transplant
# adipose tissue | finding_site_of | lipoatrophy
# renal failure | focus_of | emergency hemofiltration
# hepatitis a | has_associated_morphology | Hepatocellular necrosis
# influenza | has_causative_agent | Influenza A virus
# acute myocardial infarction | has_clinical_course | Abrupt onset
# fasting triglyceride | has_component | triacylglycerol
# tumor necrosis factor | has_disposition | immunomodulator
# pancreaticoduodenectomy | has_direct_procedure_site | abdomen
# subcutaneous insulin | has_direct_substance | insulin
# postmenopausal | has_occurrence | postmenopausal osteopenia
# melanoma | has_finding_site | skin
# vertebroplasty | has_focus | Vertebral fracture
# bariatric surgery | has_method | surgical action
# glucagon | has_modification | glucagon hydrochloride
# nephron | has_part | glomerulus
# alzheimer disease  | has_pathological_process | neurofibrillary tangle formation
# overweight | interprets | body weight measure
# adiponectin | inverse_isa | high molecular weight adiponectin
# blood eosinophil counts | is_interpreted_by | asthmatic pulmonary eosinophilia
# empagliflozin | isa | sodium glucose cotransporter subtype 2 inhibitor
# esomeprazole magnesium hydrate | is_modification_of | esomeprazole magnesium
# surgical closure | method_of | resuture of perineal wound
# acute chest syndrome ( acs ) | occurs_after | sickle cell - hemoglobin ss disease
# mastectomy | occurs_before | post - mastectomy chronic pain syndrome
# cardiac amyloidosis | occurs_in | old age
# postpartum anxiety status | occurs_in | maternal postpartum
# cellular senescence | part_of | cellular response to stress
# agomelatine | plays_role | antidepressant therapeutic role
# coronary syndrome | possibly_equivalent_to | preinfarction syndrome
# mi | same_as | myocardial infarction
# premacular fibrosis | direct_morphology_of | preretinal membranectomy
# ganglion cell layer | has_entire_anatomy_structure | entire ganglion cell layer of retina
# renal arteries | indirect_procedure_site_of | endovascular renal denervation
# autoimmune | pathological_process_of | myasthenia gravis with exacerbation
# islets of langerhans | procedure_site_of | procedure on endocrine pancreas"""

#------