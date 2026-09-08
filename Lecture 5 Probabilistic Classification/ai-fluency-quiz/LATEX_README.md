# AI Fluency quiz LaTeX package

The two `.tex` files are self-contained and use only standard TeX Live packages.

- `quiz-05-ai-fluency-sample.tex` is the editable quiz template and rendered sample. Change the metadata commands near the top, then edit the five `\AIClaim{...}{...}` entries and the evidence packet.
- `ai-fluency-quiz-student-guide.tex` explains the format to students.
- `canvas-ai-fluency-announcement.md` is ready to paste into Canvas.

Compile from this directory with:

```bash
latexmk -pdf quiz-05-ai-fluency-sample.tex
latexmk -pdf ai-fluency-quiz-student-guide.tex
```

Because the course is cross-listed, the quiz separates the registrar fields:

```text
Course: [ ] ECE 4424  [ ] CS 4824     Section: ______
```

`ECE` or `CS` identifies the student's enrollment prefix; it should not replace the section field.

## TA handoff for each new quiz

Ask the TA to provide one small source package for every quiz:

- the editable `.tex` source;
- the compiled student-facing PDF;
- a separate answer key or brief grading notes for course staff; and
- any figures, data excerpts, or source notes needed to rebuild the PDF.

The student guide is course-wide and should not be copied into every lecture package. Revise it only when the quiz format or grading policy changes.

For the public course site, post only the released student PDF. Add it to the central quiz page and link it from the matching lecture page. Keep unreleased quizzes and staff grading materials outside the public site until they are safe to publish.

For a released quiz, use the following website handoff:

1. Copy the student PDF to `website/quizzes/files/quiz-XX-short-topic.pdf`.
2. Add one row to `website/quizzes/index.qmd`.
3. Add a short quiz link to the matching file in `website/lectures/`.

This gives students one predictable quiz archive while keeping each quiz visible beside the lecture concepts it assesses.
