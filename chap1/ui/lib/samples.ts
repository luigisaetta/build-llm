/**
 * Short sample texts that show how tokenizers treat different kinds of text.
 */

export type Sample = {
  label: string;
  text: string;
};

export const SAMPLES: Sample[] = [
  {
    label: "English",
    text:
      "Large Language Models do not read words: they read tokens. " +
      "A tokenizer splits the text into pieces, and maps each piece to an integer ID.",
  },
  {
    label: "Italian",
    text:
      "I modelli linguistici non leggono parole, ma token. " +
      "Le lingue diverse dall'inglese richiedono spesso più token per la stessa frase.",
  },
  {
    label: "Python code",
    text:
      "def greet(name: str) -> str:\n" +
      '    """Return a friendly greeting."""\n' +
      '    return f"Hello, {name}!"\n',
  },
  {
    label: "Numbers",
    text: "In 2026, 1234567 + 89 = 1234656, and pi is about 3.14159265.",
  },
  {
    label: "Emoji",
    text: "Tokenizers 🤖 split emoji 😀🎉 into byte pieces, and <|endoftext|> is plain text here.",
  },
];
