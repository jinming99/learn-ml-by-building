# Third-party notices

## kaobook

This folder includes `kaohandt.cls` and `kao.sty` from the [kaobook project](https://github.com/fmarotta/kaobook), licensed under the LaTeX Project Public License (LPPL), version 1.3 or later. The license text and upstream manifest are included as `KAOBOOK-LICENSE` and `KAOBOOK-MANIFEST.md`.

The local `kao.sty` adaptation changes the monospaced font to Libertinus Mono, removes the obsolete `usenames` option from `xcolor`, and omits unused optional integrations for PDF inclusion, subfiles, margin todos, algorithms, Creative Commons icons, glossaries, and nomenclature. These changes are marked in the source.

## tcolorbox

`vendor/tcolorbox/` contains version 6.2.0 of `tcolorbox`, by Thomas F. Sturm, pinned for compatibility with the LaTeX kernel used to build these notes. It is distributed under the LPPL, version 1.3 or later; each vendored source file retains its upstream copyright and license header.
