# The American Statistician · Statistical Computing and Graphics

**Prepared on 21 November 2026 for author review. The current manuscript has an unresolved eligibility issue.** OpenAI Codex (GPT-6) drafted portions of the research text and interpretation. The publisher expressly restricts first drafts of manuscripts and sections by generative AI. Disclosure records the history and cannot by itself establish eligibility. Both authors should address this before using the files for an actual submission. One route is to ask the journal how its policy applies to the documented history. If the authors prepare their own manuscript from the results, they should retain and disclose the prior history as applicable. No editorial decision has been obtained. [Publisher's AI guidance](https://authorservices.taylorandfrancis.com/editorial-policies/using-ai-in-your-research-and-manuscript-preparations/)

## Files for author review

| File | Intended recipient and contents |
| --- | --- |
| [Anonymous manuscript](manuscript-anonymous.pdf) | Reviewer copy. Contains the figures, sensitivity table and declarations. The author list, contact details and project URL are omitted. |
| [Identified manuscript](manuscript-identified.pdf) | Editorial copy with both authors, shared affiliation, corresponding author and repository URL. |
| [Anonymous replication supplement](replication-anonymous.zip) | Reviewer copy with a one-page online appendix, a readme, data dictionaries, archived scores, replay code, figure and table generators, tests, pinned dependencies and file hashes. |
| [Identified replication supplement](replication-identified.zip) | Same scientific files, with author credit in the appendix and readme. |
| [English cover letter](cover-letter.pdf) and [editable text](cover-letter.txt) | Editor-only administrative working draft. It names the intended section, explains the result and discloses the manuscript's drafting history. It contains no personal attestation on behalf of the authors. |

In the anonymous files the UCI citations still identify the original data creators. The data sets are credited in both versions. The research manuscript was previously placed on GitHub, so a reader who independently searches its title may be able to identify it.

## Official requirements checked

- **Journal and section.** The ASA identifies Statistical Computing and Graphics as a section of TAS and links to ScholarOne. Use the ASA journal link to reach the live system when the authors elect to proceed. [ASA journal entry](https://www.amstat.org/publications/q-and-as/the-american-statistician-welcomes-submissions/)
- **Review format.** The ASA style guide specifies US letter, 12-point double-spaced review text (about 26 lines per full text page), an abstract of at most 200 words, and three to five keywords. It specifies author–year citations, an alphabetical reference list, and double-anonymous review. It asks for separate anonymous and identified supplementary materials. The prepared manuscripts are ten pages each, with four keywords and an abstract below the limit. Figure and table captions accompany the displays. The full grid of numerical results is in the online appendix. [ASA style and anonymization guide](https://files.taylorandfrancis.com/asa-style-guide.pdf)
- **Funding, competing interests and data.** Both versions state the confirmed funding and competing-interest information and explain how to obtain the UCI files and reproduce the figures. Direct identifiers and the author repository link were checked in the reviewer PDF text, metadata, bookmarks, link targets, and in the reviewer ZIP, including its embedded PDFs. [ASA style guide](https://files.taylorandfrancis.com/asa-style-guide.pdf)
- **Cover letter.** The editor's copy identifies the journal, section, article and corresponding author and explains its relevance. The publisher's advice also covers prior publication and concurrent review. Both authors must confirm the applicable facts before making those representations. [Publisher's cover-letter guidance](https://authorservices.taylorandfrancis.com/publishing-your-research/making-your-submission/writing-a-journal-article-cover-letter/)
- **Verification limit.** The journal-specific instructions endpoint returned HTTP 403. Section-specific length limits, file designations, required forms and account-only fields could not be verified there. Consult the journal's live instructions when preparing the actual submission. No portal record was created. [Journal instructions](https://www.tandfonline.com/action/authorSubmission?show=instructions&journalCode=utas20)

## Items for both authors

1. Resolve the AI first-draft issue above. Check the named tool, its version, use in analysis and drafting, and the authors' actual review of the code, figures, references, terms of use and interpretations. Only the authors can confirm responsibility and approval of the final version. The current PDFs disclose known use. The English administrative cover letter is also an AI-assisted draft for the corresponding author's review.
2. Check author order, corresponding author and institutional address. The identified copy uses the project-confirmed affiliation and correspondence email. Supply the author-confirmed postal code and any additional contact email required by the form.
3. Confirm originality, previous dissemination including the repository version, concurrent submission status, and approval by both authors before adding a declaration or signing a form. Recheck the live portal fields and assign editor-only and reviewer-visible files accordingly.

## Rebuild the packet

From the repository root, run `python submission/tas/build.py` with Tectonic on the path. The two review manuscripts and appendices share the scientific text and table rows of the archived research version. The supplement is assembled from an explicit list of replication files. Each ZIP includes a SHA-256 manifest and instructions for rebuilding the results independently from the released scores or the original UCI files.
