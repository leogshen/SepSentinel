# Email drafts — data requests

Drafted 2026-09-07. **Nothing here has been sent.** Leo sends these himself
from his own address; they go out under his name.

## Rules I followed in writing these

- **He is a high-school student and every draft says so.** Not buried, not
  finessed. Researchers are generally glad to help a student who has clearly
  done the work; nobody is glad to discover halfway through that they were
  misled about who they were talking to.
- **Short.** A busy PI decides in about ten seconds whether to reply. Every
  one of these is under 200 words.
- **One specific ask per email**, answerable in a sentence or two.
- **Evidence of homework** — each names the exact dataset, accession, or
  table, so it is obvious he read the paper rather than mass-mailing.
- **No email is sent for data that can just be downloaded.** See below.

## DO NOT SEND: Mount Sinai / Icahn (SDY1662)

The Del Valle *et al.* (Nat Med 2020) cytokine data is **publicly deposited
on ImmPort as SDY1662**, and ImmPort registration is free, instant and
self-service — no DUA, no IRB, no CITI, no PI signature.

**Action instead:** register at https://www.immport.org with a personal
address, accept the click-through agreement, download SDY1662 and SDY1655.
Same day. Emailing Icahn for this would ask a busy lab to do something Leo
can do himself in fifteen minutes.

*Contingency only:* if the deposit turns out to lack the serial timepoints
(our two recon passes disagree — one found ~3,075 IL-6 results at ~2 draws
per patient, the earlier sweep found ~1.3 draws per patient with only n=244
having repeats), then email Sacha Gnjatic <sacha.gnjatic@mssm.edu>, and only
then. Draft 4 below covers that case.

---

## Draft 1 — Humanitas sepsis cohort (Zenodo 7612571)

**Why this one first: it is the only true Sepsis-3 cohort with IL-6 we
found, and the access path is a one-click request.** 178 patients, IL-6 /
IL-8 / IL-10 / TNF-alpha / PTX3 at ED admission and day 5.

**To:** corresponding author of Davoudian *et al.* (verify the current
address on the paper before sending)
**Subject:** Request to use Zenodo dataset 7612571 (sepsis cytokines) for a
student project

> Dear Dr. [Name],
>
> I'm a high school student working on a sepsis early-warning project with a
> sponsoring faculty advisor. I'm building a model that predicts sepsis from
> ICU vital signs, and I'm trying to understand how much a biomarker like
> IL-6 would add if it could be measured continuously rather than once or
> twice.
>
> I'd like to request access to your Zenodo dataset 7612571 from the
> Humanitas sepsis cohort. As far as I can tell it's the only publicly
> available dataset with IL-6 measured in patients diagnosed by Sepsis-3
> criteria, which is the definition I'm using. The two timepoints (admission
> and day 5) are exactly what I need to see how the marker moves.
>
> I'd use it only for this research project, wouldn't try to re-identify
> anyone, and would cite your paper in anything I write up. I'm happy to
> share what I find.
>
> Thank you for considering it,
> Leo Shen

---

## Draft 2 — Zigong Fourth People's Hospital ICU database

**Why: it may be the only routine-care serial IL-6 paired with a full ICU
time series. Costs one email to find out.** Ask before spending months on
credentialing.

**To:** corresponding author of the Zigong PhysioNet project
**Subject:** Question about lab variables in the Zigong ICU infection database

> Dear Dr. [Name],
>
> I'm a high school student working on sepsis prediction with a faculty
> advisor, and I have a quick question about your Zigong Fourth People's
> Hospital ICU infection database on PhysioNet.
>
> Before I go through the credentialing process, could I ask whether the
> laboratory table includes interleukin-6 or procalcitonin, and roughly how
> many patients have them measured more than once?
>
> I ask because I checked MIMIC-IV 3.1 directly and it has no interleukin
> items at all, and neither AmsterdamUMCdb nor HiRID has IL-6. If your
> database has serial IL-6 alongside the vital signs, it would be the only
> one I've found that does, and it would be worth the wait for access.
>
> Either way, thank you for making the database available.
>
> Best,
> Leo Shen

---

## Draft 3 — SICdb maintainer (IL-6 census)

**Why: the SICdb lab dictionary ships only inside the restricted download,
so this cannot be answered from outside. The maintainer answers GitHub
issues promptly.** Can be a GitHub issue instead of an email — arguably
better, since the answer then helps everyone.

**To:** Niklas Rodemund, or an issue on https://github.com/nrodemund/sicdb
**Subject:** Does d_references include Interleukin-6 or Procalcitonin?

> Hello,
>
> I'm a high school student working on a sepsis early-warning project, and
> I've been reading the SICdb documentation and import code on GitHub. Thank
> you for publishing the full schema openly — it let me understand the
> database structure before requesting access, which was really useful.
>
> One question I couldn't answer from the public materials: does the
> `d_references` table include Interleukin-6 or Procalcitonin among the 426
> laboratory parameters, and if so roughly how many cases have them?
>
> I'm asking because MIMIC-IV has no interleukin items at all, so a European
> ICU database that records IL-6 routinely would be valuable. Since
> `d_references` is only in the restricted download I can't check it myself.
>
> Thanks very much,
> Leo Shen

---

## Draft 4 — Mount Sinai, CONTINGENCY ONLY

Send **only** if SDY1662 turns out to lack serial timepoints after download.

**To:** Sacha Gnjatic <sacha.gnjatic@mssm.edu>
**Subject:** Serial timepoints in ImmPort SDY1662 (Del Valle et al., Nat Med 2020)

> Dear Dr. Gnjatic,
>
> I'm a high school student working on a sepsis early-warning project with a
> faculty advisor. I downloaded your cohort from ImmPort (SDY1662) and have
> been working with the cytokine and clinical tables.
>
> I wanted to ask about the sampling: the deposit looks like it has mostly a
> single admission draw per patient, with repeat measurements for a smaller
> subset. Is that the full extent of the serial sampling, or were additional
> timepoints collected that aren't in the ImmPort deposit?
>
> I'm studying how much of a biomarker's usefulness comes from its size
> versus how often it's measured, so the number of draws per patient matters
> more to me than the number of patients.
>
> Thank you for depositing the data publicly — it's the best paired
> cytokine-and-vitals dataset I've been able to find.
>
> Best regards,
> Leo Shen

---

## Before sending any of these

1. **Verify the current corresponding-author address on the paper itself.**
   Addresses in secondary sources go stale.
2. **Use a consistent signature** — name, school, and advisor's name and
   institution. Naming the advisor is what makes a student email credible;
   ask them first.
3. **Send one at a time**, not as a batch, and wait about two weeks before
   any follow-up.
4. **Do ImmPort first.** Having already worked with real data makes every
   one of these emails stronger, and it may make draft 1 unnecessary.
