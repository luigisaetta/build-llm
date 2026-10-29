# Contributing

build-llm is a personal learning project: a collection of what I explore and
build while studying how Large Language Models work. Questions, corrections,
and ideas from other learners are very welcome.

## Ask a question

Open an issue with the **Question** template. Good questions are the ones
you had while reading the code or the book: "why is this done this way?",
"what happens if I change this value?".

Questions with general interest, with a short answer, may be added to
[Questions and answers](docs/questions-and-answers.md), organized by chapter.

## Report a problem

Open an issue with the **Bug report** template, and include:

* the chapter and the file (script or notebook);
* the command you ran, and the full error message;
* your operating system and machine (for example macOS on Apple Silicon),
  and the output of `python -m chap0.check_environment`.

## Propose a change

Small fixes (typos, unclear explanations, broken links) can go directly in a
pull request. For larger changes, such as a new chapter or a new demo, open
an issue first, so that we can agree on the approach.

Before opening a pull request:

1. Follow the [Quickstart](QUICKSTART.md) to set up the `build-llm`
   environment.
2. Follow the coding rules in [AGENTS.md](AGENTS.md): clarity first, English
   documentation, the standard module header, and tests for new code.
3. From the repository root, run the checks, and make them pass:

   ```bash
   black .
   ```

   ```bash
   pylint build_llm chap* tests
   ```

   ```bash
   pytest
   ```

   For changes to a web UI, also run `npm run lint` and `npm run build` in
   its folder.

4. If you add material derived from other sources, state its origin and
   license, and update [NOTICE](NOTICE).

## Licensing of contributions

By contributing, you agree that your contribution is released under the
[MIT License](LICENSE), like the original code of this repository.
Changes to the notebooks derived from LLMs-from-scratch are released under
their [Apache License 2.0](LICENSES/Apache-2.0-LLMs-from-scratch.txt), as
explained in [NOTICE](NOTICE).
