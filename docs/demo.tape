# ivgym — the inference-verification game, told as a story:
#   a cheating PROVIDER (attack) vs an auditing VERIFIER (defense), scored by detection AUC.
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
Type "cd /root/inference-verification" Enter
Type "setopt interactivecomments" Enter
Type "alias py='python3'" Enter
Type "clear" Enter
Show

# 1. The setup: the con
Type "#  You rent an LLM API. You pay for full-precision Qwen3, temperature 0.7."
Sleep 1800ms Enter
Type "#  THE ATTACKER (a cheating provider) quietly deviates to cut its bill:"
Sleep 1200ms Enter
Type "#    serve a 4-bit model, crank the temperature, reuse a seed, spike a logit..."
Sleep 1200ms Enter
Type "#    ...then hand back tokens and swear it ran exactly your spec."
Sleep 1500ms Enter
Type "#  THE DEFENDER (a verifier) re-runs your prompt under the CLAIMED spec,"
Sleep 1200ms Enter
Type "#    scores each token's divergence S, and flags the batch when S > tau."
Sleep 1500ms Enter
Type "#  Detection AUC = how reliably the cheat gets caught.  Here is who's on the board:"
Sleep 1500ms Enter
Type "py -m experiments.run --list" Enter
Sleep 3500ms

Type "clear" Enter
Sleep 400ms

# 2. One attacker and one defender, in plain Python
Type "#  Both sides are plain Python you @register -- no edits to the library."
Sleep 1500ms Enter
Type "#  THE ATTACK 'logit_spike': 3% of tokens, secretly shove +6 onto a few logits."
Sleep 1200ms Enter
Type "#  THE DEFENSE 'topk_overlap': how far down the verifier's ranking the claimed token sits."
Sleep 1500ms Enter
Type "sed -n '28,72p' examples/custom_strategies.py" Enter
Sleep 6500ms

Type "clear" Enter
Sleep 400ms

# 3. They join the game as first-class players
Type "#  Point --strategies at the file -- the new attacker and defender join the registries:"
Sleep 1500ms Enter
Type "py -m experiments.run --strategies examples/custom_strategies.py --list" Enter
Sleep 3s

# 4. The showdown: every attacker vs every defender
Type "#  THE SHOWDOWN. Each cell is detection AUC:  1.0 = caught every time,  0.5 = invisible cheat."
Sleep 1800ms Enter
Type "py -m experiments.run --strategies examples/custom_strategies.py \" Enter
Type "   --attacks logit_spike quant_4bit temp_1.1 \" Enter
Type "   --defenses token_difr cross_entropy topk_overlap \" Enter
Type "   --prompts 30 --tokens 128 --batch 1000 --n-batches 200" Enter
Sleep 9s
Type "#  Read it like a scoreboard: some cheats are nailed by every verifier; one slips past."
Sleep 3500ms

Type "clear" Enter
Sleep 400ms

# 5. Same con, a bigger arena: the backends
Type "#  Same con, bigger ARENA. The backend the game runs on is one flag:"
Sleep 1500ms Enter
Type "py -m experiments.run --help | grep -- '--backend'" Enter
Sleep 3s
Type "#  --backend synthetic  -> the duel we just ran here, no GPU."
Sleep 1500ms Enter
Type "#  --backend hf_gpu     -> the IDENTICAL attackers/defenders/scoring on a real model on a GPU:"
Sleep 1500ms Enter
Type "#  py -m experiments.run --backend hf_gpu --strategies examples/custom_strategies.py"
Sleep 2500ms Enter
Type "#  Write a cheat or a catcher once; run it on the synthetic CPU model AND a real GPU model unchanged."
Sleep 3500ms Enter
