# GitHub presentation and launch notes

These are prepared maintainer settings and announcement copy. Editing this file
does not update GitHub metadata or publish an announcement.

## Repository About

Suggested description:

> Check what your LLM provider actually serves. Independent token and timing verification, a CPU demo, and reproducible inference-auditing research.

Suggested website:

```text
https://github.com/lijiawei20161002/inference-verification#try-it-without-a-gpu
```

Suggested topics:

```text
llm, inference, inference-verification, model-auditing, ai-safety,
machine-learning, evaluation, reproducible-research, pytorch, fastapi
```

Set these from the repository's About panel. GitHub topics make repositories
browsable by subject; see [GitHub's topic documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics).

## Before sharing

- Publish the repository changes and check the Actions result. The new badge
  reports real workflow status; it is not a claim of past hosted CI success.
- Confirm the project's intended license with the rights holder and include it
  in the repository before describing the project as open source.
- Run the README demo in a clean clone. Keep its explanation of recorded data.
- Choose one scoped task from [CONTRIBUTING.md](../CONTRIBUTING.md#places-to-start)
  as a first contribution opportunity. Apply labels to real issues when opened.

## Announcement draft

**Is your LLM provider serving the model and settings it claims?**

I built IVGym to study what returned tokens and timing can tell us about that
question. It includes an independent verification API, a NumPy-only demo that
replays real H100 timing traces, and reproducible attack/detector experiments.

The demo shows how the timing gap changes when a provider's context is truncated.
The repository also keeps the difficult results visible: a small pilot detected
a temperature override but flagged none of its real NF4 blocks. Cross-hardware
calibration and deployment error rates remain open research questions.

I'm looking for feedback on the API, provider integrations, and replications on
other hardware.

[Try the demo](https://github.com/lijiawei20161002/inference-verification#try-it-without-a-gpu)

## After sharing

Use relevant inference and ML systems communities where project sharing is
welcome. Lead with a reproducible example and a concrete request for feedback.
Track whether visitors complete the demo, report useful issues, and contribute
replications alongside stars. Fix repeated onboarding problems first.
