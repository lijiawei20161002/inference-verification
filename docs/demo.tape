# ivgym — adding custom attack/defense strategies and running them on a backend.
# Render with:  vhs docs/demo.tape   (run from the inference-verification/ dir)
Output docs/demo.gif
Output docs/demo.mp4

Set Shell zsh
Set FontSize 15
Set Width 1280
Set Height 800
Set Theme "Catppuccin Mocha"
Set TypingSpeed 25ms
Set Margin 16
Set MarginFill "#1e1e2e"
Set BorderRadius 8

# --- silent setup ---
Hide
Type "cd /Users/lijiawei/Desktop/inference-verification" Enter
Type "setopt interactivecomments" Enter
Type "alias py='python3'" Enter
Type "clear" Enter
Show

# 1. What ships built in
Type "#  ivgym: cheating providers (attacks)  vs  verifiers (defenses), scored by detection AUC."
Sleep 1500ms Enter
Type "#  Everything registered out of the box:"
Sleep 800ms Enter
Type "py -m experiments.run --list" Enter
Sleep 3s

Type "clear" Enter
Sleep 400ms

# 2. Add your own — a plain file, two @register decorators
Type "#  Add your OWN strategies in a file. No edits to the library: just @register a subclass."
Sleep 1500ms Enter
Type "sed -n '28,72p' examples/custom_strategies.py" Enter
Sleep 6s

Type "clear" Enter
Sleep 400ms

# 3. They are now first-class, picked up everywhere
Type "#  Point --strategies at the file. logit_spike + topk_overlap now join the registries:"
Sleep 1200ms Enter
Type "py -m experiments.run --strategies examples/custom_strategies.py --list" Enter
Sleep 3s

# 4. Run the sweep — custom attack/defense scored against the built-ins
Type "#  Run the sweep: detection AUC for each attack x each defense (the new ones included)."
Sleep 1500ms Enter
Type "py -m experiments.run --strategies examples/custom_strategies.py \" Enter
Type "   --attacks logit_spike quant_4bit temp_1.1 \" Enter
Type "   --defenses token_difr cross_entropy topk_overlap \" Enter
Type "   --prompts 30 --tokens 128 --batch 1000 --n-batches 200" Enter
Sleep 9s

Type "clear" Enter
Sleep 400ms

# 5. Different backends — same strategies, same command
Type "#  Same strategies, different ARENA. The backend is one flag:"
Sleep 1200ms Enter
Type "py -m experiments.run --help | grep -- '--backend'" Enter
Sleep 3s
Type "#  --backend synthetic  -> runs here, no GPU (what we just did)."
Sleep 1200ms Enter
Type "#  --backend hf_gpu     -> the IDENTICAL attacks/defenses/harness on a real model on a GPU:"
Sleep 1200ms Enter
Type "#  py -m experiments.run --backend hf_gpu --strategies examples/custom_strategies.py"
Sleep 2500ms Enter
Type "#  Write a strategy once; run it on the synthetic CPU model and on a real GPU model unchanged."
Sleep 3s Enter
