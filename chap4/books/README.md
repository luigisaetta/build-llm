# Chapter 4 training books

These are modified plain-text conversions of public-domain literary works
from [Standard Ebooks](https://standardebooks.org/). Copyright: the literary
texts are in the public domain; Standard Ebooks contributors dedicate their
contributions under [CC0 1.0](../../LICENSES/CC0-1.0.txt).

The corpus has **31 complete books and 10,056,985 GPT-2 tokens**. The count
is the sum of each UTF-8 text encoded with `tiktoken.get_encoding("gpt2")`,
before adding the document separators used by training.

Only bodymatter XHTML is retained, in EPUB spine order. Chapter headings,
paragraphs, and explicit verse line breaks are kept. Front and back matter,
notes, note references, and Gutenberg headers/footers are excluded; invisible
formatting characters and typographic spaces are normalized. Title and author
are retained. Source and license metadata stay in this README, the manifest,
and NOTICE rather than being repeated inside the training text.

The original twelve texts are retained unchanged. The nineteen added texts
are downloaded data and are ignored by Git. Restore them from the repository
root, in the `build-llm` environment:

```bash
python -m chap4.prepare_books
```

The [manifest](../books_manifest.json) records each new file's edition,
author, translator where applicable, immutable source revision, retained
chapter paths, modifications, token count, and SHA-256 of the converted text.
The command verifies existing files and refuses to overwrite modified texts.
The original twelve editions and their conversion details are in
[NOTICE](../../NOTICE).

| File | Book / author | GPT-2 tokens |
| --- | --- | ---: |
| `the-time-machine.txt` | "The Time Machine", H. G. Wells, 1895 | 41,578 |
| `ethan-frome.txt` | "Ethan Frome", Edith Wharton, 1911 | 48,728 |
| `heart-of-darkness.txt` | "Heart of Darkness", Joseph Conrad, 1899 | 51,678 |
| `the-war-of-the-worlds.txt` | "The War of the Worlds", H. G. Wells, 1898 | 78,344 |
| `dr-jekyll-and-mr-hyde.txt` | "The Strange Case of Dr. Jekyll and Mr. Hyde", Robert Louis Stevenson, 1886 | 34,798 |
| `a-christmas-carol.txt` | "A Christmas Carol", Charles Dickens, 1843 | 42,162 |
| `pride-and-prejudice.txt` | "Pride and Prejudice", Jane Austen, 1813 | 161,830 |
| `frankenstein.txt` | "Frankenstein", Mary Shelley, 1818 | 95,169 |
| `dracula.txt` | "Dracula", Bram Stoker, 1897 | 209,600 |
| `the-picture-of-dorian-gray.txt` | "The Picture of Dorian Gray", Oscar Wilde, 1890 | 109,030 |
| `the-hound-of-the-baskervilles.txt` | "The Hound of the Baskervilles", Arthur Conan Doyle, 1902 | 80,861 |
| `treasure-island.txt` | "Treasure Island", Robert Louis Stevenson, 1883 | 97,767 |
| `war-and-peace.txt` | [War and Peace](https://standardebooks.org/ebooks/leo-tolstoy/war-and-peace/louise-maude_aylmer-maude), Leo Tolstoy; translated by Louise and Aylmer Maude | 796,437 |
| `les-miserables.txt` | [Les Misérables](https://standardebooks.org/ebooks/victor-hugo/les-miserables/isabel-f-hapgood), Victor Hugo; translated by Isabel F. Hapgood | 801,594 |
| `bleak-house.txt` | [Bleak House](https://standardebooks.org/ebooks/charles-dickens/bleak-house), Charles Dickens | 504,743 |
| `david-copperfield.txt` | [David Copperfield](https://standardebooks.org/ebooks/charles-dickens/david-copperfield), Charles Dickens | 514,791 |
| `dombey-and-son.txt` | [Dombey and Son](https://standardebooks.org/ebooks/charles-dickens/dombey-and-son), Charles Dickens | 509,986 |
| `little-dorrit.txt` | [Little Dorrit](https://standardebooks.org/ebooks/charles-dickens/little-dorrit), Charles Dickens | 481,483 |
| `our-mutual-friend.txt` | [Our Mutual Friend](https://standardebooks.org/ebooks/charles-dickens/our-mutual-friend), Charles Dickens | 495,666 |
| `nicholas-nickleby.txt` | [Nicholas Nickleby](https://standardebooks.org/ebooks/charles-dickens/nicholas-nickleby), Charles Dickens | 482,732 |
| `martin-chuzzlewit.txt` | [Martin Chuzzlewit](https://standardebooks.org/ebooks/charles-dickens/martin-chuzzlewit), Charles Dickens | 498,913 |
| `the-pickwick-papers.txt` | [The Pickwick Papers](https://standardebooks.org/ebooks/charles-dickens/the-pickwick-papers), Charles Dickens | 474,373 |
| `middlemarch.txt` | [Middlemarch](https://standardebooks.org/ebooks/george-eliot/middlemarch), George Eliot | 436,282 |
| `vanity-fair.txt` | [Vanity Fair](https://standardebooks.org/ebooks/william-makepeace-thackeray/vanity-fair), William Makepeace Thackeray | 420,170 |
| `moby-dick.txt` | [Moby Dick](https://standardebooks.org/ebooks/herman-melville/moby-dick), Herman Melville | 293,165 |
| `jane-eyre.txt` | [Jane Eyre](https://standardebooks.org/ebooks/charlotte-bronte/jane-eyre), Charlotte Brontë | 262,302 |
| `the-brothers-karamazov.txt` | [The Brothers Karamazov](https://standardebooks.org/ebooks/fyodor-dostoevsky/the-brothers-karamazov/constance-garnett), Fyodor Dostoevsky; translated by Constance Garnett | 508,682 |
| `the-count-of-monte-cristo.txt` | [The Count of Monte Cristo](https://standardebooks.org/ebooks/alexandre-dumas/the-count-of-monte-cristo/chapman-and-hall), Alexandre Dumas; translated by Chapman and Hall | 700,226 |
| `the-portrait-of-a-lady.txt` | [The Portrait of a Lady](https://standardebooks.org/ebooks/henry-james/the-portrait-of-a-lady), Henry James | 326,281 |
| `great-expectations.txt` | [Great Expectations](https://standardebooks.org/ebooks/charles-dickens/great-expectations), Charles Dickens | 261,531 |
| `oliver-twist.txt` | [Oliver Twist](https://standardebooks.org/ebooks/charles-dickens/oliver-twist), Charles Dickens | 236,083 |
| **Total** | **31 books** | **10,056,985** |
