# Axis-labelled publication figures

These five editable SVGs contain the quantitative plots and axis labels supplied at the author's proof stage for Figures 2, 3, 4, 5 and 7. Filenames retain the article's numbering. Each is a self-contained vector graphic that can be opened in a browser or Inkscape. The source geometry preserves the estimates, individual points, uncertainty intervals and numeric labels.

Figure 4 expresses changes in Dice in percentage points. In Figure 5, panels A and B count examinations, C reports mean false-positive enhancing-tumor volume in mm³ and D gives component positive predictive value. The planning comparison in Figure 7 is development-only minus historical Dice.

For the inputs and code underlying the quantitative figures, see the analysis directory. These editable files preserve the final axis typography and panel layout.

The accepted vector sources are in `../analysis/editable_figures/`. To regenerate the five updated SVGs, install the Python package `lxml` and run:

```sh
python apply_axis_labels.py --output rebuilt
```
