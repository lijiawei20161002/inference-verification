# ivgym — inference verification, told as an audit:
#   the verifier re-scores a CLAIMED run token-by-token and labels each
#   SAFE (green) / SUSPICIOUS (yellow) / DANGEROUS (red), then renders a verdict.
# Render with:  vhs demo/demo.tape   (run from the inference-verification/ dir)
Output demo/demo.gif
Output demo/demo.mp4

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
Type "#  You rent an LLM API: you pay for full-precision Qwen3 at the spec you agreed on."
Sleep 1800ms Enter
Type "#  A cheating provider can quietly deviate -- 4-bit weights, a wrong temperature,"
Sleep 1200ms Enter
Type "#  a reused seed, a spiked logit -- then hand back tokens and swear it ran your spec."
Sleep 1500ms Enter
Type "#  The VERIFIER re-scores every claimed token and labels it safe / suspicious / dangerous."
Sleep 2000ms Enter
Type "clear" Enter
Sleep 400ms

# 2. The honest baseline -- a clean board
Type "#  First, an HONEST provider. The verifier should wave it through:"
Sleep 1500ms Enter
Type "py -m experiments.verify_demo --claimed honest" Enter
Sleep 5500ms
Type "#  Nearly all green, and the batch statistic S stays under the 1%-FPR threshold: CONSISTENT."
Sleep 3500ms Enter
Type "clear" Enter
Sleep 400ms

# 3. The cheat -- caught
Type "#  Now the SAME audit against a provider secretly serving a 4-bit model:"
Sleep 1500ms Enter
Type "py -m experiments.verify_demo --claimed quant_4bit" Enter
Sleep 5500ms
Type "#  Yellow and red bloom across the stream, S blows past tau:  DEVIATION DETECTED."
Sleep 3500ms Enter
Type "clear" Enter
Sleep 400ms

# 4. The whole board, in one matrix
Type "#  That was one cheat vs one verifier. The full game is every attack x every defense,"
Sleep 1500ms Enter
Type "#  scored by detection AUC  (1.0 = caught every time,  0.5 = an invisible cheat):"
Sleep 1800ms Enter
Type "py -m experiments.run --attacks honest quant_4bit temp_1.1 adv_quant_temp \" Enter
Type "   --defenses token_difr cross_entropy activation_difr \" Enter
Type "   --prompts 30 --tokens 128 --batch 1000 --n-batches 200" Enter
Sleep 9s
Type "#  Some cheats are nailed by every verifier; the adversarial one drives cross_entropy to ~0 -- invisible to that defense."
Sleep 3500ms Enter
Type "clear" Enter
Sleep 400ms

# 5. Same audit, a bigger arena
Type "#  Attacks and defenses are pluggable; the backend is one flag:"
Sleep 1500ms Enter
Type "py -m experiments.run --help | grep -- '--backend'" Enter
Sleep 3s
Type "#  --backend synthetic -> the audit you just watched, no GPU."
Sleep 1500ms Enter
Type "#  --backend hf_gpu    -> the IDENTICAL verifier on a real model on a real GPU."
Sleep 3000ms Enter
