"""Generate all five published figures."""
from figures import fig1_circuit
from figures import fig2_saturation
from figures import fig3_gap
from figures import fig4_scatter
from figures import fig5_noise
from figures.style import OUTPUT_DIR, print_color_map

print("Model identity map (fixed across all figures):")
print_color_map()

for mod in (fig1_circuit, fig2_saturation, fig3_gap, fig4_scatter, fig5_noise):
    print(f"\n=== {mod.__name__} ===")
    mod.main()

print("\nOutputs in", OUTPUT_DIR)
for f in sorted(OUTPUT_DIR.glob("*.pdf")):
    print(" ", f)
