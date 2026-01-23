# Literature Buckets: Harms and Costs of Over-escalation (References + Keywords)

This file is a starting point for collecting references that justify assigning a non-zero cost to over-escalation, in two channels:

1) **Direct harms to the escalated patient**
2) **Indirect harms to other patients via crowding / capacity externalities**

The goal is not to claim a single “true” number, but to justify **weight profiles** (e.g., patient-first vs resource-constrained) with traceable evidence.

## A) Indirect harms: crowding, boarding, delays (harm to other patients)

**What to support**
- Increased ED/urgent capacity load leads to longer waits, delays to treatment, worse quality metrics, and (in many studies) higher mortality.
- Over-escalation is an upstream contributor to load, so its marginal cost includes externalities.

**Search terms**
- “emergency department crowding mortality systematic review”
- “ED boarding mortality”
- “time to antibiotics sepsis delay crowding”
- “waiting time outcomes emergency department”
- “queueing theory healthcare externalities crowding”

**Types of sources to cite**
- National Academies / Institute of Medicine reports on ED crowding/boarding and hospital-based emergency care
- Systematic reviews/meta-analyses on ED crowding and mortality/clinical outcomes
- Policy statements summarizing consensus (ACEP, etc.)

**Key references (starting set)**
- Institute of Medicine (US) Committee on the Future of Emergency Care in the United States Health System. (2007). *Hospital-Based Emergency Care: At the Breaking Point*. Washington, DC: The National Academies Press. https://doi.org/10.17226/11621
- Hoot, N. R., & Aronsky, D. (2008). Systematic review of emergency department crowding: Causes, effects, and solutions. *Annals of Emergency Medicine, 52*(2), 126–136.e1. https://doi.org/10.1016/j.annemergmed.2008.03.014
- Morley, C., Unwin, M., Peterson, G. M., Stankovich, J., & Kinsman, L. (2018). Emergency department crowding: A systematic review of causes, consequences and solutions. *PLOS ONE, 13*(8), e0203316. https://doi.org/10.1371/journal.pone.0203316
- Stang, A. S., Crotts, J., Johnson, D. W., Hartling, L., & Guttmann, A. (2015). Crowding measures associated with the quality of emergency department care: A systematic review. *Academic Emergency Medicine, 22*(6), 643–656. https://doi.org/10.1111/acem.12682
- Woodworth, L. (2020). Swamped: Emergency department crowding and patient mortality. *Journal of Health Economics, 70*, 102279. https://doi.org/10.1016/j.jhealeco.2019.102279

## B) Direct harms: low-value care, diagnostic cascades, incidental findings (harm to the escalated patient)

**What to support**
- Unnecessary “urgent” routing often triggers additional workups (labs, imaging, consults).
- False positives and incidentalomas create cascades: follow-ups, procedures, iatrogenic injury, anxiety, and financial toxicity.

**Search terms**
- “diagnostic cascade harms incidental findings”
- “incidentaloma follow-up harms anxiety”
- “low-value testing emergency department harms”
- “overdiagnosis overtreatment harms”
- “Choosing Wisely emergency medicine imaging”

**Types of sources to cite**
- Systematic reviews on overtesting/low-value care in ED/urgent contexts
- “cascades of care” literature (e.g., incidental findings leading to downstream testing)
- Overdiagnosis/overtreatment literature (conceptual grounding for “false positive” costs)
- Guideline/consensus sources (Choosing Wisely) for common low-value tests

**Key references (starting set)**
- Ganguli, I., Simpkin, A. L., Lupo, C., Weissman, A., Mainor, A. J., Orav, E. J., et al. (2019). Cascades of care after incidental findings in a US national survey of physicians. *JAMA Network Open, 2*(10), e1913325. https://doi.org/10.1001/jamanetworkopen.2019.13325
- Ganguli, I., Simpkin, A. L., Colla, C. H., Weissman, A., Mainor, A. J., Rosenthal, M. B., et al. (2019). Why do physicians pursue cascades of care after incidental findings? A national survey. *Journal of General Internal Medicine, 35*(4), 1352–1354. https://doi.org/10.1007/s11606-019-05213-1
- Brenner, D. J., & Hall, E. J. (2007). Computed tomography—an increasing source of radiation exposure. *New England Journal of Medicine, 357*(22), 2277–2284. https://doi.org/10.1056/NEJMra072149
- Choosing Wisely. (n.d.). *Choosing Wisely recommendations* (campaign overview and specialty lists; includes emergency medicine imaging/testing “don’t” items). https://www.choosingwisely.org/

## C) Explicit precedent for the acceptable over/under-triage tradeoff

**Why it’s useful**
- Trauma systems and field triage literature explicitly discusses acceptable rates of over-triage to avoid under-triage (patient-first with resource awareness).
- These sources help justify using a high weight on missed escalation, while still assigning non-zero cost to over-escalation.

**Search terms**
- “acceptable overtriage undertriage trauma”
- “ACS COT overtriage 25 35 undertriage 5”
- “field triage guidelines injured patients overtriage”

**Key references (starting set)**
- Newgard, C. D., Fischer, P. E., Gestring, M., Michaels, H. N., Jurkovich, G. J., Lerner, E. B., et al. (2022). National guideline for the field triage of injured patients: Recommendations of the National Expert Panel on Field Triage, 2021. *Journal of Trauma and Acute Care Surgery, 93*(2), e49–e60. https://doi.org/10.1097/TA.0000000000003627
- American College of Surgeons, Committee on Trauma. (2014). *Resources for Optimal Care of the Injured Patient*. Chicago, IL: American College of Surgeons.

## D) Turning literature into benchmark weights (practical method)

Rather than attempting a single “correct” conversion to dollars/QALYs, use literature to justify a *range*:

- Choose a unit baseline: `over_escalation_patient = 1`.
- Use crowding literature to justify `over_escalation_system` being non-zero and potentially > 1 under constrained capacity.
- Use triage precedent to justify `missed_escalation` being 50–500× larger than `over_escalation_patient`.

Then publish multiple profiles and report all scores.

### Optional: show “anchoring calculations” (transparent but approximate)

If you want to demonstrate how specific papers informed weight *ranges*, one transparent approach is to report a back-of-the-envelope mapping:

- Use cascade-of-care studies to justify that an unnecessary escalation has a non-trivial probability of downstream testing and psychological/financial burden (supports `over_escalation_patient > 0`).
- Use crowding studies to justify that increased load/delay correlates with worse outcomes, meaning over-escalation has system externalities (supports `over_escalation_system > 0`, potentially larger in constrained settings).
- Use triage guideline targets (tolerated over-triage vs strict under-triage) to justify an order-of-magnitude or larger asymmetry between missed escalation and over-escalation (supports `missed_escalation >> over_escalation_*`).

This benchmark intentionally avoids claiming a single universally-valid conversion from these studies to a numeric harm unit; instead it makes the assumptions explicit and supports reviewer-supplied weight profiles.
