
## Place these after all example variables are defined
SYSTEM_CONTEXT = """You are a medical‐domain extractor building a diabetes KG of <head, relation, tail>. You possess advanced medical academic knowledge.
Given input sequence, identify entities specifically relevant to diabetes, its complications, comorbidities, therapeutics and related biomedical entities that help to clarify or contextualize them. Output a Python list of up to 6-word entity “heads” following these rules:

  1. Select a precise and medically-specific span (e.g., “myocardial infarction,” not “infarction”). Avoid generic terms like "disease", "condition", "patients" and "comorbidity" without a specific context. When encountering vague descriptors like "complication," "symptom," or "effect," always prefer explicitly named conditions or symptoms directly linked to diabetes pathology or diabetes comorbidities.
  2. Keep original spelling, casing, and abbreviations from the sequence.
  3. Choose only entities that add meaningful medical knowledge to the diabetes KG. Do not include COVID-related terms. Do not include head entities that describe findings in animal models (mice, rats, etc.).
    A few examples of low-value entities you SHOULD NOT include:
    '≥ 10 % weight reduction': 10% is too context-dependent.
    'nhanes 2015 - 2018': a dataset name or survey, not a medical entity.
    'semaglutide 2.4 mg': includes a dosage, which can vary in different contexts.
    '60+ women': "60+" is too context-dependent.
    "anxiety", "home births", "pregnant women", "neonatal deaths", "general practitioners": not specific enough to diabetes. Add them only if they are mentioned in the relation to diabetes-specific terms.
  4. If it's not clear whether a term adds a diabetes-specific knowledge, look at the context. If the text explicitly links the term to a diabetes-specific term, add it. Otherwise, when mentioned only in a generic context, it should be excluded. Include such terms when the sequence clearly links them to a diabetes-relevant gene, pathway, cell type, or therapeutic effect.
 
You will be provided with incorrect output examples beginning with "Output (Incorrect)". Use them to avoid the common mistakes.

Wrap your intermediate reasoning steps clearly within <think> ... </think> tags. Be strict and discard any entity about which you are uncertain and that is not relevant to diabetes.
After generating, verify your output. 

Steps:
  1. Identify candidate spans.
  2. Filter by medical precision and relevance rules.
  3. Confirm the entity's relevance and contribution to the diabetes KG, discard low-value entities.

Input format:
sequence

Output format:
['head1', 'head2', …] 
If none, output []
"""


example_user_1 = """Input:
, its upstream regulator, has the opposite effect ( han et al., 2013 ). previous studies suggest that chop deteriorates er stress and accelerates cell death via promoting protein synthesis and oxidative stress ( han et al., 2013 ). in addition, er stress damages β - cells possibly through altering ca2 + homeostasis. it has been indicated that er stress interferes with the function of ryr located in the membrane of the er and causes leakage of er ca2 + ( yamamoto et al., 2019 ). the destruction of β - cell's er ca2 + homeostasis results in impaired insulin secretion and further promotion of β - cell death.

Output:"""

example_assistant_1 = """
['chop', 'er stress', 'ryr', 'β - cells', 'impaired insulin secretion']
"""

example_explanation_1 = """The rationale behind the output:
Candidates: chop; er stress; β - cells; ryr; impaired insulin secretion; oxidative stress; protein synthesis. Keep precise diabetes-relevant entities: chop (ER-stress regulator), er stress, β - cells, ryr, impaired insulin secretion.
"""
# -----

example_user_2 = """Input:
), published between 2020 and 2021, including patients of any age with type 1 or type 2 diabetes mellitus ( t1dm or t2dm ), that reported glycaemic control parameters and / or lipid profile as primary or secondary outcomes, and assessed comparisons of metabolic control among diabetic patients during and before lockdown. we excluded studies with the following criteria : experimental studies, non - english language articles, studies that included other types of diabetes or involved patients with severe acute respiratory syndrome coronavirus comorbid infections, and those studies that did not reported the glycaemic control and / or lipid profile during lockdown. the search

Output:"""

example_assistant_2 = """
['type 2 diabetes mellitus', 't2dm', 'glycaemic control parameters', 'lipid profile']
"""

example_explanation_2 = """The rationale behind the output:
Identify candidate spans: “type 1 diabetes mellitus,” “t2dm,” “glycaemic control parameters,” “lipid profile.”
Filter: all four are precise, and directly relevant:
    type 1 diabetes mellitus, t2dm: two core diabetes subtypes;
    glycaemic control parameters, lipid profile: two key clinical metrics for monitoring diabetes status.

Discard generic phrases like “metabolic control”.
"""

# -----

example_user_3 = """Input:
##bp on the first use of antihypertensive drug ( s ). during pregnancy a high amount of fatty acids ( fa ) is necessary to meet foetus demands, which vary during gestation. the present study describes the changes in maternal fatty acid concentrations during pregnancy in a sample of pregnant women. methods this is a longitudinal study of 479 pregnant women who were monitored from the first trimester to third trimester of pregnancy. data on maternal characteristics were recorded and a serum sample was collected in each trimester. the fatty acid profile ( saturated ( sfa : total, lauric acid, myristic acid, palmitic acid, stearic acid ), monounsaturated

Output:"""

example_assistant_3 = """
['palmitic acid', 'stearic acid', 'lauric acid', 'myristic acid']
"""

example_explanation_3 = """The rationale behind the output:
Candidate spans: antihypertensive drug; fatty acids; saturated fatty acids; lauric acid; myristic acid; palmitic acid; stearic acid; monounsaturated.

Filter for precision & diabetes relevance
• antihypertensive drug → generic term, no specific agent ⇒ discard.
• fatty acids / saturated fatty acids / monounsaturated → broad classes, too unspecific ⇒ discard.
• The four named fatty acids (lauric, myristic, palmitic, stearic) are precise metabolites often studied in insulin-resistance and diabetes-risk research ⇒ keep.

Each retained fatty acid is a well-defined biomolecular entity linked to metabolic dysfunction and diabetes, thus adding meaningful diabetic context.
"""
# -----

example_user_4 = """Input:
detect fibrosis. results : stz - induced hyperglycemic rats demonstrated high blood glucose, dyslipidemia, and significant elevation in hepatic transaminases, proinflammatory cytokines, nf - κb, lipid peroxidation, and hepatic fibrosis, with impairment in antioxidant enzymes. in stz - induced hyperglycemic rats, the administration of lcz696 ameliorated hyperglycemia, dyslipidemia, improved liver functions, and boosted antioxidants enzymes. furthermore, lcz696 therapy attenuated oxidation, inflammation, progression of liver injury, and hepatic fibrosis. lcz696 was superior to valsartan in reducing ast, hepatic fibrosis, tissue il - 1β, tnf - α and nf - κb. in

Output:"""

example_assistant_4 = """
[]
"""
# -----

example_explanation_4 = """The rationale behind the output:
All findings are from an stz-induced rat model. Since our KG represents human diabetes and its comorbidities, animal-only entities are excluded. Therefore, no entities qualify for inclusion.
"""
# -----

example_user_5 = """Input:
glucagon - like peptide - 1 receptor agonists among individuals with type 2 diabetes, including those with concomitant atherosclerotic cardiovascular disease ( ascvd ), has remained low in the united states between 2014 and 2019, despite clinical evidence supporting their use for cardiovascular risk reduction. these findings add to the existing literature to highlight a gap in adherence to current practice guidelines, which suggests that most patients with t2d and ascvd in the united states may not be receiving optimal risk ‐ reducing therapies. atrial fibrillation ( af ) is a common cardiac arrhythmia. the ratio of red cell distribution width ( rdw ) to albumin has been recognized as a reliable prognostic marker for poor outcomes in

Output:"""

example_assistant_5 = """
['glucagon - like peptide - 1 receptor agonists', 'type 2 diabetes', 'atherosclerotic cardiovascular disease', 'atrial fibrillation']
"""

example_explanation_5 = """The rationale behind the output:
Candidates: glucagon - like peptide - 1 receptor agonists, type 2 diabetes, atherosclerotic cardiovascular disease, atrial fibrillation, red-cell-distribution-width–to-albumin ratio.
• Keep precise diabetes therapy and disease entities: “glucagon - like peptide - 1 receptor agonists” and “type 2 diabetes”.
• ASCVD and atrial fibrillation are well-recognized cardiovascular comorbidities of diabetes — retain.
• RDW/albumin ratio is a generic prognostic metric without clear diabetes specificity — discard.
"""
# -----

example_user_6 = """Input:
and capn10 gene polymorphisms and gestational diabetes mellitus ( gdm ) has been explored in diverse populations across different geographical regions. yet, most of these studies have been confined to a limited number of loci, resulting in inconsistent findings. in this study, we conducted a comprehensive review of published literature to identify studies examining the relationship between tcf7l2 and capn10 gene polymorphisms and the incidence of gdm in various populations. we specifically focused on five loci that were extensively reported in a large number of publications and performed a meta - analysis. methods we prioritized the selection of snps with well - documented correlations established in existing literature on gdm. we searched

Output:"""

example_assistant_6 = """
['gestational diabetes mellitus', 'capn10', 'tcf7l2']
"""

example_explanation_6 = """The rationale behind the output:
Candidates: • gestational diabetes mellitus • gdm (abbrev) • capn10 • tcf7l2 • gene polymorphisms (generic) • loci / SNPs (generic)

Filter:
• Keep only medically precise, diabetes-relevant entities.
– “gestational diabetes mellitus” (specific diabetes subtype).
– “capn10” (gene repeatedly linked to GDM risk).
– “tcf7l2” (gene strongly associated with diabetes phenotypes).
• Discard generic terms like “gene polymorphisms”, “loci”, “SNPs”, and the standalone abbreviation “gdm”.

All three entities directly describe the disease of interest or genes whose polymorphisms are being meta-analysed for their contribution to GDM incidence—valuable additions to a diabetes KG.
"""
# -----

example_user_7 = """Input:
were 1. 58 and 1. 96, respectively. in the intensive glycemic management group, hrs for a first severe hypoglycemia event and recurrent hypoglycemia were 1. 10 and 1. 24, respectively. in summary, can was independently associated with higher risks of a first hypoglycemia event and recurrent hypoglycemia among adults with type 2 diabetes, with the highest risk observed among those on standard glycemic management. blood glucose ( bg ) management is crucial for type - 1 diabetes patients resulting in the necessity of reliable artificial pancreas or insulin infusion systems. in recent years, deep learning techniques have been utilized for a more accurate bg level prediction system. however, continuous glucose

Output:"""

example_assistant_7 = """
['severe hypoglycemia', 'recurrent hypoglycemia', 'type 2 diabetes', 'type - 1 diabetes', 'artificial pancreas', 'insulin infusion systems']
"""

example_explanation_7 = """The rationale behind the output:

Candidate spans: severe hypoglycemia; recurrent hypoglycemia; type 2 diabetes; type - 1 diabetes; artificial pancreas; insulin infusion systems; intensive glycemic management; standard glycemic management; continuous glucose (truncated).

• severe hypoglycemia – acute, well-defined diabetes complication.
• recurrent hypoglycemia – clinically distinct, adds knowledge.
• type 2 diabetes – core disease entity.
• type - 1 diabetes – core disease entity.
• artificial pancreas – specific diabetes technology.
• insulin infusion systems – specific diabetes therapy device.
Discard management-strategy phrases (generic) and the truncated “continuous glucose”.

Each retained entity represents a complication, disease subtype, or technology directly relevant to diabetes care and thus enriches the KG.
"""

example_user_8 = """Input:
of polymeric nanoparticles ( nps ), this review will provide a comprehensive and in - depth summary of the polyester - based nanovehicles, including poly ( lactic - co - glycolic acid ) ( plga ) and polylactic acid ( pla ) nps, used to deliver a variety of foreign na, e. g. short interfering rna ( sirna ), messenger rna ( mrna ), and plasmid dna ( pdna ). the article will review the versatility of polyester - based nanocarriers including their recent application in the delivery of the clustered, regularly ‐ interspaced, short palindromic repeats / cas ( crispr /

Output:"""

example_assistant_8 = """
['poly ( lactic - co - glycolic acid )', 'polylactic acid', 'short interfering rna', 'messenger rna', 'plasmid dna', 'crispr / cas']
"""

example_explanation_8 = """The rationale behind the output:

poly ( lactic - co - glycolic acid ) – Biodegradable NP carrier frequently employed to deliver insulin, GLP-1 or gene cargos in diabetes therapy studies.
polylactic acid – Another polyester NP material tested for controlled insulin and peptide release formulations relevant to diabetes treatment.
short interfering rna – siRNA therapeutics are used to silence genes driving insulin resistance or β-cell apoptosis in diabetic models.
messenger rna – mRNA drugs encoding insulin or GLP-1 analogues represent a cutting-edge modality for diabetes therapy.
plasmid dna – pDNA vectors are explored to express insulin or protective factors in gene-based diabetes interventions.
crispr / cas – Genome-editing system applied to correct diabetes-associated mutations and engineer functional β-cells, central to emerging curative strategies.
"""
# -----

# ==negative examples==
example_user_negative_1 = """Input:
, its upstream regulator, has the opposite effect ( han et al., 2013 ). previous studies suggest that chop deteriorates er stress and accelerates cell death via promoting protein synthesis and oxidative stress ( han et al., 2013 ). in addition, er stress damages β - cells possibly through altering ca2 + homeostasis. it has been indicated that er stress interferes with the function of ryanodine receptor (ryr) located in the membrane of the er and causes leakage of er ca2 + ( yamamoto et al., 2019 ). the destruction of β - cell's er ca2 + homeostasis results in impaired insulin secretion and further promotion of β - cell death.

Output (Incorrect):"""

example_assistant_negative_1 = """
['chop', 'previous studies', 'beta - cell death', 'stress', 'ryanodine receptor (ryr)']
"""

example_explanation_negative_1 = """The above output is partially incorrect for the following reasons:
Low-quality heads: head "previous studies" doesn't add value to the diabetes knowledge graph because it is vague and generally is not relevant to diabetes.
Low-quality heads: head "stress" is too generic and does not add specific medical knowledge about diabetes. It may be "er stress" instead that is a cellular stress pathway central to diabetic pathophysiology.
Misspelled heads: head "beta - cell death" spelling is violated -- that should be "β - cell death" -- exactly as in the sequence.
Incorrect span choice: for "ryanodine receptor (ryr)", output only "ryanodine receptor" without the abbreviation.

The correct output would be: ['chop', 'ryanodine receptor', 'β - cells', 'er stress']
"""
# -----

example_user_negative_2 = """Input:
that mir - 101 - 3p and mir - 132 - 3p are direct regulators of trib1 expression and function. binding of mir - 101 - 3p and mir - 132 - 3p to the 3 ’ utr of trib1 mrna leads to an increased transcription and secretion of interleukin - 8. our data demonstrate that modulation of trib1 by mirnas alters the inflammatory profile of both human macrophages and prostate cancer cells. specific dietary recommendations for migraine patients with comorbid irritable bowel syndrome ( ibs ) are lacking. this work aimed to study the severity scores of such two common pain - related disorders in relation to various macronutrients and micronutrients intake. methods

Output (Incorrect):"""

example_assistant_negative_2 = """
['trib1 – gene', 'migraine', 'irritable bowel syndrome', 'ibs', 'interleukin - 8']
"""

example_explanation_negative_2 = """The above output is partially incorrect for the following reasons:
Irrelevant terms: heads 'migraine', 'irritable bowel syndrome', 'ibs' are not relevant to diabetes. None of these terms are specifically diabetes-related or add useful context to a diabetes KG in this study’s setting (no diabetes cohort or metabolic insight relevant to diabetes).

However, the output also includes relevant terms: heads 'trib1 – gene', 'interleukin - 8 are relevant to diabetes, because:
Trib1 is involved in lipid metabolism and insulin-resistance pathways, which are key in type 2 diabetes.
interleukin - 8 – pro-inflammatory cytokine known to be elevated in diabetic complications.

Therefore, discard 'migraine', 'irritable bowel syndrome', 'ibs' as not clearly contributing to a diabetes KG and retain only trib1 and interleukin-8.
"""
# -----

example_user_negative_3 = """Input:
of polymeric nanoparticles ( nps ), this review will provide a comprehensive and in - depth summary of the polyester - based nanovehicles, including poly ( lactic - co - glycolic colostomy on ulcerative crp was 49. 2 %, with a highest effective rate of 88. 2 % within 12 to 24 months after colostomy. 9 ( 31. 1 % ) crp patients with ulcers were cured after colostomy and 12 ( 19. 67 % ) patients restored intestinal continuity, among which including 2 ( 3. 3 % ) patients ever with rectovaginal fistula. 100 % ( 55 / 55 ) patients with rectal bleeding and 91. 4 % ( 32 / 35 ) patients with anal pain were remarkably alleviated. additionally, multivariable analysis showed the duration of stoma [ or 1. 211, 95 % ci

Output (Incorrect):"""

example_assistant_negative_3 = """
['colostomy', 'fistula', 'rectal bleeding', 'anal pain', 'polymeric nanoparticles']
"""

example_explanation_negative_3 = """The above output is incorrect for the following reasons:
Irrelevant terms: the sequence talks about colostomy outcomes for bowel disease, and the terms “fistula,” “rectal bleeding,” and “anal pain” relate here to chronic radiation proctitis with no stated link to diabetes.
Low-quality heads: 'polymeric nanoparticles' is too general. There is no a strong link to diabetis or its comorbidities.

The correct output for this sequence would be [].
"""

# Positive examples
POSITIVE_PROMPT_EXAMPLES = [
  (example_user_1, example_assistant_1, example_explanation_1),
  (example_user_2, example_assistant_2, example_explanation_2),
  # (example_user_3, example_assistant_3, example_explanation_3),
  # (example_user_4, example_assistant_4, example_explanation_4),
  # (example_user_5, example_assistant_5, example_explanation_5),
  # (example_user_6, example_assistant_6, example_explanation_6),
  # (example_user_7, example_assistant_7, example_explanation_7),
  # (example_user_8, example_assistant_8, example_explanation_8),
]

# Negative examples
NEGATIVE_PROMPT_EXAMPLES = [
  (example_user_negative_1, example_assistant_negative_1, example_explanation_negative_1),
#   (example_user_negative_2, example_assistant_negative_2, example_explanation_negative_2),
#   (example_user_negative_3, example_assistant_negative_3, example_explanation_negative_3),
]