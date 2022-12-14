(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fontenc" "T1")))
   (TeX-run-style-hooks
    "latex2e"
    "File_Setup"
    "scrartcl"
    "scrartcl10"
    "fontenc"
    "listings"
    "xcolor")
   (LaTeX-add-labels
    "sec:va"))
 :latex)

