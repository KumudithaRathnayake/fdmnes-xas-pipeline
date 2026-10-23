# Example structures and configs

| File | What it is |
|---|---|
| `structures/demo_MO6.cif` | Synthetic, invented P1 CIF: a single `Ni` atom octahedrally coordinated by 6 `O` atoms (Ni-O = 2.1 Å). Not a real or published material -- built only so the K-edge/CIF example runs out of the box. |
| `structures/demo_Cu_hexaaqua.xyz` | Synthetic, invented idealized octahedral `Cu(H2O)6` (Cu-O = 2.10 Å, O-H = 0.96 Å, 104.5° H-O-H angle). Not a real or published structure. Also used by the test suite's input-format regression test. |
| `structures/demo_Zn_hexaaqua.xyz` | Same geometry as above, with `Zn` in place of `Cu`. |
| `jobs.json` | K-edge/CIF batch config: the `Demo_Ni_MO6` job runs as-is against `demo_MO6.cif`; `Your_Structure_Here` is a template to duplicate for your own CIF. |
| `jobs_ledge.json` | L-edge/XYZ batch config: both jobs run as-is against the synthetic hexaaqua structures above. |

None of the structures here come from unpublished research data -- they
were generated purely geometrically (ideal octahedral coordination) to
give this repository a small, runnable, citation-free example. For real
work, swap in your own CIF/XYZ (or a public structure, e.g. from the
[Crystallography Open Database](http://www.crystallography.net/cod/)) and
point `structure_file`/`xyz_file` at it -- see the main README's "Two
structure input modes" section.
