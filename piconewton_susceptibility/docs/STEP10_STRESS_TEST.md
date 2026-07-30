# Step 10 stress-test

## Decision

**IMPLEMENTATION PASS.** The archive builder and cold-run orchestration are complete. Final production closure requires executing the pinned notebook or hosted workflow against a clean runtime so the real Steps 2–10 archive is generated.

## Over-simplification

**PASS.** The builder exports all eight mandatory result groups through the six locked main figures, manuscript tables, supplementary robustness outputs and complete prior-step artifacts. It does not reduce the final deliverable to a notebook or a small set of plots.

## Overcomplication

**PASS.** Step 10 contains no new scientific model. It performs validation, figure/table rendering, provenance capture, packaging and checksumming only.

## Reproducibility

**PASS.** The notebook creates a unique Drive run directory and executes the entire chain from a clean clone. The hosted workflow independently executes the same publication chain on Python 3.10–3.12.

## Claim fidelity

**PASS.** `claim_lock.json` and `reduced_law.json` are copied byte-for-byte at the semantic JSON level. Gates explicitly require `scientific_claim_modified=false` and `new_scientific_fit_run=false`.

## Defects found and corrected

1. Step 2 initially relied only on downstream continuity. Direct Step 2 source, runtime, completion-gate and checksum validation was added.
2. Figure 1 initially omitted explicit Step 3–4 continuity. A dedicated continuity panel and source CSV were added.
3. Conditional reduced-law and robustness results were initially table-only. Three supplementary figures were added without changing the six main-figure contract.
4. The archive initially lacked an internal inventory and pre-archive checksum list. Both are now embedded.
5. Fixture execution is explicitly non-claim-bearing and cannot authorise publication closure.

## Residual external condition

Google Drive authentication and the final full-resolution runtime cannot be executed from the present connector environment. Any failure in cloning, installation, Drive writing, a prior scientific gate, figure generation or archive checksum stops the production notebook before a complete Step 10 manifest can be written.
