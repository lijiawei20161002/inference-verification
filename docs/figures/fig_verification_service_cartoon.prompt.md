# Verification service cartoon figure

Generated with the built-in imagegen tool. This figure illustrates the live-request path.

## Generation prompt

Use case: infographic-diagram.
Asset type: a polished cartoon infographic for the IVGym repository, illustrating the implemented independent verification service protocol.
Create one vivid, friendly, landscape figure with charming cartoon icons, crisp dark outlines, bright restrained colors, generous whitespace, and large readable English labels on a light background. Make it approachable to a nontechnical reader. Use a clean flat illustration style, not a screenshot or a technical flowchart.

Title: "Check an AI provider"
Subtitle: "Anyone can ask. An independent service checks."

Show THREE visibly separate parties across the page, with the independent service as the largest central panel:
LEFT: a friendly person with a laptop, labeled "Anyone".
CENTER: a clearly bounded panel labeled "Independent verification service", containing a friendly detective/magnifying glass icon and two check cards.
RIGHT: a separate colorful server illustration labeled "AI provider".

The flow must be accurate, with clear directed arrows and labels:
1. From Anyone to the service, label "1. Ask for a check", with a small "POST /v1/verify" tag beneath it.
2. From the service to the provider, label "2. Send test prompts".
3. From the provider back to the service, label "3. Return answers".
4. Inside the service, label "4. Run two checks".
5. A lower return route from the service back to Anyone ends at the laptop and is labeled "5. Get a report".
Keep the outgoing and returning arrows in separate lanes, with unmistakable arrowheads, no crossings through labels.

Inside the service panel:
A "Token check" card shows colorful word/token tiles and a magnifying glass. Its plain-language caption is "Compare with our own reference model".
A "Clock check" card shows a stopwatch and short versus long stacks of prompt cards. Its caption is "Compare arrival times for short and long prompts".
A small book/model icon plus baseline chart inside this same central boundary is labeled "Our reference model + trusted baseline".
A report sheet at the bottom of this panel has three simple rows: "No difference detected", "Difference detected", "Not enough evidence". Use neutral result symbols, not certification seals.

Use consistent warm, lively cartoon icon artwork across all components. Keep typography large and highly legible, concise labels, plenty of breathing room. The three parties must be visibly distinct: the service is operated by an independent third party; the provider does not own the verifier, reference model, baseline, or stopwatch. The stopwatch measures when streamed output arrives at the service, not the provider's GPU clock. This figure illustrates the live-request path. Do not depict cryptographic receipts, signatures, locks, blockchains, sample commitments, guaranteed correctness, or provider-run verification. Do not add extra text or logos.
